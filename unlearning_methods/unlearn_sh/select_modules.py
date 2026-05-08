"""Stage 0-M: Forget/Retain Gradient Ratio를 이용한 **모듈** 선택 스크립트.

select_layers.py가 레이어 단위(layer 8, 9, ...)로 집계하는 것과 달리,
이 스크립트는 개별 모듈 단위(model.layers.8.self_attn.q_proj, model.layers.10.mlp.up_proj, ...)
로 점수를 계산하여 Top-K 모듈을 선택합니다.

알고리즘:
  1. Forget 데이터 NPO loss gradient 계산 → g_f_i (모듈 i)
  2. Retain 데이터 CE loss gradient 계산  → g_r_i (모듈 i)
  3. Module-wise RCP Projection:
       각 모듈 i별로 독립적으로 dot(g_f_i, g_r_i) < 0 인지 판단
       충돌하면 g_proj_i = g_f_i - λ * (dot_i / ||g_r_i||²) * g_r_i
       충돌 없으면 g_proj_i = g_f_i
     (global projection과 달리, 모듈마다 독립적으로 적용하여
      모듈별 forget_score가 해당 모듈의 충돌 상황을 정확히 반영)
  4. 개별 모듈별 forget_score / retain_score 계산 (파라미터 수 정규화)
  5. Forget score가 median 이상인 모듈만 필터링 (Noise 방지)
  6. Ratio (Forget/Retain) 기준 Top-K 모듈 선택
  7. 결과를 json으로 저장 및 config 적용 안내

사용법:
    python unlearning_methods/unlearn_sh/select_modules.py \
        --config_name config \
        --top_k 40 \
        --num_batches 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_methods.unlearn_sh.dataloader import RCPForgetDataset, rcp_collator
from unlearning_methods.unlearn_sh.loss import compute_dpo_loss, compute_retain_loss
from utils import get_model_identifiers_from_yaml


def extract_layer_idx(name):
    """'model.layers.14.mlp.up_proj.weight' -> 14"""
    if "layers." not in name:
        return None
    try:
        return int(name.split("layers.")[1].split(".")[0])
    except Exception:
        return None


def extract_module_key(name):
    """'model.layers.14.mlp.up_proj.weight' -> 'model.layers.14.mlp.up_proj'

    .weight suffix를 제거하여 모듈 이름만 반환.
    """
    if name.endswith(".weight"):
        return name[: -len(".weight")]
    return name


def extract_module_type(name):
    """'model.layers.14.mlp.up_proj.weight' -> 'up_proj'

    모듈의 leaf 이름(종류)만 반환.
    """
    key = extract_module_key(name)
    return key.split(".")[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Forget/Retain gradient ratio 기반 개별 모듈 선택"
    )
    parser.add_argument("--config_name", default="config", help="Hydra config name")
    parser.add_argument("--top_k", type=int, default=40, help="Number of modules to select")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches to estimate gradients")
    parser.add_argument("--output_file", default="./outputs/module_analysis.json", help="Path to save result")
    args = parser.parse_args()

    # Hydra config 로드
    from hydra import compose, initialize
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name=args.config_name)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    model_path = cfg.model_path if cfg.model_path else model_cfg["ft_model_path"]
    model_path = str(PROJECT_ROOT / model_path)

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 모델 로드 ──────────────────────────────────────────
    device = f"cuda:{cfg.gpu_train}"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if model_cfg.get("flash_attention2") == "true" else None,
        torch_dtype=torch.bfloat16 if getattr(cfg, "bf16", False) else torch.float16,
        trust_remote_code=True,
    ).to(device)

    # oracle 로드 (NPO 계산용)
    oracle_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if model_cfg.get("flash_attention2") == "true" else None,
        torch_dtype=torch.bfloat16 if getattr(cfg, "bf16", False) else torch.float16,
        trust_remote_code=True,
    ).to(f"cuda:{cfg.gpu_oracle}")
    oracle_model.eval()

    # ── 데이터 로드 ────────────────────────────────────────
    dataset = RCPForgetDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=500,
        split=cfg.split,
        language=cfg.language,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=rcp_collator
    )

    # 타겟 파라미터 식별 (Linear weight만, lm_head 제외)
    target_params = []
    param_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            if hasattr(module, "weight"):
                target_params.append(module.weight)
                param_names.append(name + ".weight")
                module.weight.requires_grad = True

    # ── 모듈 단위 점수 누적 딕셔너리 ───────────────────────
    # key: 모듈 전체 이름 (예: "model.layers.14.mlp.up_proj")
    forget_scores = {}       # projected forget score (배치 합산)
    retain_scores = {}       # retain score (배치 합산)
    raw_forget_scores = {}   # projection 전 raw forget score (배치 합산)
    dot_sums = {}            # dot_i 합산 (부호 포함)
    projection_counts = {}   # dot_i < 0 이었던 배치 수

    print(f"Estimating gradients over {args.num_batches} batches...")
    model.eval()

    batch_idx = 0
    for inputs in dataloader:
        if batch_idx >= args.num_batches:
            break

        forget_inputs, retain_inputs = inputs

        # 1. Forget NPO gradient
        forget_loss, _ = compute_dpo_loss(
            model=model,
            ref_model=oracle_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=cfg.beta,
        )
        g_f = torch.autograd.grad(forget_loss, target_params, retain_graph=False, allow_unused=True)

        # 2. Retain CE gradient
        retain_loss = compute_retain_loss(model, retain_inputs)
        g_r = torch.autograd.grad(retain_loss, target_params, retain_graph=False, allow_unused=True)

        # 3 & 4. Module-wise RCP Projection + 점수 누적
        # global projection과 달리, 모듈(파라미터) i별로 독립적으로 충돌을 판단.
        # dot_i = g_f_i · g_r_i  (스칼라)
        # if dot_i < 0: g_proj_i = g_f_i - λ * (dot_i / ||g_r_i||²) * g_r_i
        # else:         g_proj_i = g_f_i
        # 이렇게 해야 모듈별 forget_score가 해당 모듈의 충돌 상황을 반영.
        eps_proj = float(cfg.get("projection_eps", 1e-12))
        lamb = float(cfg.get("projection_lambda", 1.0))

        for gf, gr, name, p in zip(g_f, g_r, param_names, target_params):
            if extract_layer_idx(name) is None:
                continue  # layers 외부 (embedding 등) 제외

            mod_key = extract_module_key(name)

            # None 처리
            if gf is None:
                gf = torch.zeros_like(p)
            if gr is None:
                gr = torch.zeros_like(p)

            # module-wise dot & norm
            dot_i = (gf * gr).sum()
            norm_r_i = (gr * gr).sum() + eps_proj

            # module-wise projection
            if dot_i < 0:
                g_proj_i = gf - lamb * (dot_i / norm_r_i) * gr
            else:
                g_proj_i = gf

            # 점수 누적
            raw_score = (gf.norm().item() ** 2) / p.numel()   # projection 전
            proj_score = (g_proj_i.norm().item() ** 2) / p.numel()  # projection 후
            forget_scores[mod_key]      = forget_scores.get(mod_key, 0.0)      + proj_score
            retain_scores[mod_key]      = retain_scores.get(mod_key, 0.0)      + (gr.norm().item() ** 2) / p.numel()
            raw_forget_scores[mod_key]  = raw_forget_scores.get(mod_key, 0.0)  + raw_score
            dot_sums[mod_key]           = dot_sums.get(mod_key, 0.0)           + dot_i.item()
            projection_counts[mod_key]  = projection_counts.get(mod_key, 0)    + (1 if dot_i < 0 else 0)

        batch_idx += 1
        print(f"  Batch {batch_idx}/{args.num_batches} processed.")

    # 평균
    if batch_idx > 0:
        for mod_key in forget_scores:
            forget_scores[mod_key]     /= batch_idx
            retain_scores[mod_key]     /= batch_idx
            raw_forget_scores[mod_key] /= batch_idx
            dot_sums[mod_key]          /= batch_idx  # dot_mean

    # ── 필터링 및 점수 계산 ──────────────────────────────────
    eps = 1e-8
    ratios = {m: forget_scores[m] / (retain_scores[m] + eps) for m in forget_scores}

    # Forget score median 필터링
    f_vals = list(forget_scores.values())
    if len(f_vals) == 0:
        print("Error: No module gradients collected.")
        return

    median_f = np.median(f_vals)
    candidates = {m: r for m, r in ratios.items() if forget_scores[m] >= median_f}

    # Top-K 선택
    selected_modules = sorted(candidates, key=candidates.get, reverse=True)[: args.top_k]

    # ── 결과 출력 ────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"Module Analysis Results (Top {args.top_k})")
    print(f"Total modules analyzed: {len(forget_scores)}")
    print(f"Median Forget Score Threshold: {median_f:.8f}")
    print(f"Candidates after filtering: {len(candidates)}")
    print(f"{'='*90}")
    print(f"{'Module':<50} | {'Forget':<12} | {'Retain':<12} | {'Ratio':<10} | {'Sel'}")
    print("-" * 90)

    all_modules_sorted = sorted(forget_scores.keys())
    for m in all_modules_sorted:
        f_s = forget_scores[m]
        r_s = retain_scores[m]
        ratio = ratios[m]
        if m in selected_modules:
            mark = "✓"
        elif f_s < median_f:
            mark = "(Filtered)"
        else:
            mark = ""
        print(f"{m:<50} | {f_s:<12.8f} | {r_s:<12.8f} | {ratio:<10.4f} | {mark}")

    selected_modules.sort()
    print("-" * 90)
    print(f"\nSelected Modules ({len(selected_modules)}):")
    for m in selected_modules:
        layer_idx = extract_layer_idx(m)
        mod_type = extract_module_type(m)
        print(f"  Layer {layer_idx:>2} | {mod_type:<12} | {m}")

    # ── 모듈 타입별 선택 통계 ──────────────────────────────
    type_counts = {}
    for m in selected_modules:
        mt = extract_module_type(m)
        type_counts[mt] = type_counts.get(mt, 0) + 1
    print(f"\nModule type distribution:")
    for mt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {mt:<15}: {cnt}")

    # ── 레이어별 선택 통계 ─────────────────────────────────
    layer_counts = {}
    for m in selected_modules:
        li = extract_layer_idx(m)
        layer_counts[li] = layer_counts.get(li, 0) + 1
    print(f"\nLayer distribution:")
    for li in sorted(layer_counts.keys()):
        cnt = layer_counts[li]
        print(f"  Layer {li:>2}: {cnt} modules")

    # ── 결과 저장 ──────────────────────────────────────────
    output_path = Path(PROJECT_ROOT / args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "top_k": args.top_k,
                "num_batches": args.num_batches,
                "selected_modules": selected_modules,
                "module_stats": {
                    m: {
                        # ── 최종 점수 ──────────────────────────────
                        "projected_forget_score": forget_scores[m],
                        "raw_forget_score":       raw_forget_scores[m],
                        "retain_score":           retain_scores[m],
                        "ratio":                  ratios[m],
                        # ── RCP projection 진단 ────────────────────
                        "dot_mean":                   dot_sums[m],
                        "projection_applied_count":   projection_counts[m],
                        "projection_applied_ratio":   projection_counts[m] / batch_idx if batch_idx > 0 else 0.0,
                        # ── 메타 ───────────────────────────────────
                        "layer_idx":   extract_layer_idx(m),
                        "module_type": extract_module_type(m),
                    }
                    for m in forget_scores
                },
            },
            f,
            indent=4,
        )

    print(f"\nAnalysis saved to {output_path}")
    print(f"\n[NEXT STEP] config.yaml에 다음을 업데이트하세요:")
    print(f"lora_target_modules:")
    for m in selected_modules:
        print(f"  - {m}")


if __name__ == "__main__":
    main()

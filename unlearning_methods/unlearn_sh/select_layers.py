"""Stage 0: Author-noise Forget/Retain Gradient Ratio 레이어 선택 스크립트.

특징:
- Forget 데이터 author-noised NPO loss gradient 계산
- Retain 데이터 CE loss gradient 계산
- 파라미터 수로 정규화 (mean gradient squared)
- Forget score가 median 이상인 레이어만 필터링 (Noise 방지)
- Ratio (Forget/Retain) 계산 후 Top-K 레이어 선택
- 결과를 json으로 저장 및 config 적용 안내

사용법:
    python unlearning_methods/unlearn_sh/select_layers.py \
        --config_name config \
        --top_k 12 \
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

from unlearning_methods.unlearn_author_noise_npo.dataloader import (
    AuthorNoiseNPODataset,
    author_noise_npo_collator,
)
from unlearning_methods.unlearn_author_noise_npo.loss import compute_noisy_forget_npo_loss
from unlearning_methods.unlearn_sh.loss import compute_retain_loss
from utils import get_model_identifiers_from_yaml


def extract_layer_idx(name):
    """'model.layers.14.mlp.up_proj.weight' -> 14"""
    if "layers." not in name:
        return None
    try:
        return int(name.split("layers.")[1].split(".")[0])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="config", help="Hydra config name")
    parser.add_argument("--top_k", type=int, default=12, help="Number of layers to select")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches to estimate gradients")
    parser.add_argument("--output_file", default="./outputs/layer_analysis.json", help="Path to save result")
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
    dataset = AuthorNoiseNPODataset(
        cfg.data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=500,
        split=cfg.split,
        language=cfg.language,
        authors=list(cfg.get("authors", [])),
        skip_missing_trigger=True,
    )
    print(f"Trigger stats: {dataset.trigger_stats()}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=author_noise_npo_collator,
    )

    # 타겟 파라미터 식별 (Linear weight만)
    target_params = []
    param_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            if hasattr(module, "weight"):
                target_params.append(module.weight)
                param_names.append(name + ".weight")
                module.weight.requires_grad = True

    forget_scores = {}
    retain_scores = {}

    print(f"Estimating gradients over {args.num_batches} batches...")
    model.eval()
    
    batch_idx = 0
    for inputs in dataloader:
        if batch_idx >= args.num_batches:
            break
            
        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]

        # 1. Author-noised forget NPO gradient
        forget_loss, _, trigger_ratio = compute_noisy_forget_npo_loss(
            model=model,
            ref_model=oracle_model,
            forget_inputs=forget_inputs,
            trigger_mask=inputs["trigger_mask"],
            has_trigger=inputs["has_trigger"],
            beta=cfg.beta,
            noise_sigma=float(cfg.get("noise_sigma", 0.1)),
        )
        g_f = torch.autograd.grad(forget_loss, target_params, retain_graph=False, allow_unused=True)
        
        # 2. Retain CE gradient
        retain_loss = compute_retain_loss(model, retain_inputs)
        g_r = torch.autograd.grad(retain_loss, target_params, retain_graph=False, allow_unused=True)

        # 3. RCP Projection (충돌 시 retain 방향 제거)
        eps_proj = float(cfg.get("projection_eps", 1e-12))
        lamb = float(cfg.get("projection_lambda", 1.0))
        
        dot = sum((gf * gr).sum() for gf, gr in zip(g_f, g_r) if gf is not None and gr is not None)
        norm_r = sum((gr * gr).sum() for gr in g_r if gr is not None) + eps_proj
        
        g_proj_list = []
        if dot < 0:
            for gf, gr in zip(g_f, g_r):
                if gf is not None and gr is not None:
                    g_proj_list.append(gf - lamb * (dot / norm_r) * gr)
                else:
                    g_proj_list.append(gf)
        else:
            g_proj_list = list(g_f)

        # 4. 점수 누적 (forget score = ||g_proj||^2, retain score = ||g_r||^2)
        for g_p, name, p in zip(g_proj_list, param_names, target_params):
            if g_p is not None:
                l_idx = extract_layer_idx(name)
                if l_idx is not None:
                    forget_scores[l_idx] = forget_scores.get(l_idx, 0.0) + (g_p.norm().item() ** 2) / p.numel()

        for gr_val, name, p in zip(g_r, param_names, target_params):
            if gr_val is not None:
                l_idx = extract_layer_idx(name)
                if l_idx is not None:
                    retain_scores[l_idx] = retain_scores.get(l_idx, 0.0) + (gr_val.norm().item() ** 2) / p.numel()

        batch_idx += 1
        print(f"  Batch {batch_idx}/{args.num_batches} processed. trigger_ratio={trigger_ratio.item():.3f}")

    # 평균
    if batch_idx > 0:
        for l_idx in forget_scores:
            forget_scores[l_idx] /= batch_idx
            retain_scores[l_idx] /= batch_idx

    # ── 필터링 및 점수 계산 ──────────────────────────────────
    eps = 1e-8
    ratios = {l: forget_scores[l] / (retain_scores[l] + eps) for l in forget_scores}
    
    # Forget score median 필터링
    f_vals = list(forget_scores.values())
    if len(f_vals) == 0:
        print("Error: No layer gradients collected.")
        return
        
    median_f = np.median(f_vals)
    candidates = {l: r for l, r in ratios.items() if forget_scores[l] >= median_f}
    
    print(f"\n--- Layer Analysis Results (Top {args.top_k}) ---")
    print(f"Median Forget Score Threshold: {median_f:.6f}")
    print(f"{'Layer':<6} | {'Forget_Score':<15} | {'Retain_Score':<15} | {'Ratio':<10} | {'Selected'}")
    print("-" * 65)

    selected_layers = sorted(candidates, key=candidates.get, reverse=True)[:args.top_k]
    
    for l in sorted(forget_scores.keys()):
        f_s = forget_scores[l]
        r_s = retain_scores[l]
        ratio = ratios[l]
        is_sel = "✓" if l in selected_layers else ""
        if f_s < median_f:
            is_sel = "(Filtered)"
        print(f"{l:<6} | {f_s:<15.6f} | {r_s:<15.6f} | {ratio:<10.4f} | {is_sel}")

    selected_layers.sort()
    print("-" * 65)
    print(f"Selected Layers: {selected_layers}")

    # 결과 저장
    output_path = Path(PROJECT_ROOT / args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "top_k": args.top_k,
            "num_batches": args.num_batches,
            "selection_signal": "author_noised_npo_gradient_ratio",
            "noise_sigma": float(cfg.get("noise_sigma", 0.1)),
            "authors": list(cfg.get("authors", [])),
            "selected_layers": selected_layers,
            "layer_stats": {
                l: {
                    "forget_score": forget_scores[l],
                    "retain_score": retain_scores[l],
                    "ratio": ratios[l]
                } for l in forget_scores
            }
        }, f, indent=4)
        
    print(f"\nAnalysis saved to {output_path}")
    print(f"\n[NEXT STEP] config.yaml에 다음을 업데이트하세요:")
    print(f"lora_target_layers: {selected_layers}")


if __name__ == "__main__":
    main()

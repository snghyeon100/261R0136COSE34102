"""Apply a saved UNLEARN task matrix to a clean HuggingFace checkpoint."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_methods.unlearn_UNLEARN.low_rank import apply_task_matrices
from utils import get_model_identifiers_from_yaml


def resolve_project_path(path):
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((PROJECT_ROOT / path).resolve())
    return path


def _cli_cfg():
    cfg = OmegaConf.create(
        {
            "model_family": "qwen3_5_2b",
            "model_path": None,
            "task_matrix_dir": None,
            "output_dir": None,
            "tokenizer_path": None,
            "unlearn_scale": 1.0,
            "include_alpha": True,
            "bf16": True,
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.from_cli())


def apply_unlearn_checkpoint(cfg):
    os.chdir(PROJECT_ROOT)
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_path = resolve_project_path(cfg.model_path or model_cfg["ft_model_path"])
    task_matrix_dir = resolve_project_path(cfg.task_matrix_dir)
    output_dir = resolve_project_path(cfg.output_dir)
    if task_matrix_dir is None or output_dir is None:
        raise ValueError("task_matrix_dir and output_dir are required.")

    dtype = torch.bfloat16 if bool(cfg.bf16) and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    changed = apply_task_matrices(
        model,
        task_matrix_dir,
        unlearn_scale=float(cfg.unlearn_scale),
        dtype=torch.float32,
        include_alpha=bool(cfg.include_alpha),
    )

    tokenizer_source = cfg.tokenizer_path or (model_path if Path(model_path, "tokenizer_config.json").exists() else model_cfg["hf_key"])
    tokenizer_source = resolve_project_path(tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return changed


def main():
    cfg = _cli_cfg()
    changed = apply_unlearn_checkpoint(cfg)
    print(f"Applied UNLEARN task matrices to {changed} modules.")
    print(f"Saved unlearned model to {resolve_project_path(cfg.output_dir)}")


if __name__ == "__main__":
    main()

"""End-to-end LingTea source-language unlearning runner.

Run from the NLP_rice root:
    CUDA_VISIBLE_DEVICES=0,1 python unlearning_methods/unlearn_LingTea/train.py
"""

import json
import math
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
import transformers
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed

from evaluate_util import evaluate_languages
from unlearning_methods.unlearn_LingTea.dataloader import LingTeaDataset, lingtea_collator
from unlearning_methods.unlearn_LingTea.loss import compute_lingtea_loss
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


def pick_device(index):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    index = int(index)
    if index >= torch.cuda.device_count():
        raise ValueError(f"Requested cuda:{index}, but only {torch.cuda.device_count()} visible CUDA devices exist.")
    return torch.device(f"cuda:{index}")


def pick_dtype(cfg):
    if not torch.cuda.is_available():
        return torch.float32
    if bool(cfg.bf16):
        return torch.bfloat16
    if bool(cfg.fp16):
        return torch.float16
    return torch.float32


def ensure_save_dir(cfg):
    save_dir = Path(cfg.save_dir)
    if save_dir.exists() and any(save_dir.iterdir()) and not bool(cfg.overwrite_dir):
        raise FileExistsError(
            f"save_dir already exists and is not empty: {save_dir}\n"
            "Set overwrite_dir=true or choose a new save_dir."
        )
    if save_dir.exists() and bool(cfg.overwrite_dir):
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


def load_tokenizer(cfg, model_id):
    tokenizer_path = cfg.get("tokenizer_path", None)
    if tokenizer_path is not None:
        source = resolve_project_path(tokenizer_path)
    elif bool(cfg.get("prefer_checkpoint_tokenizer", True)):
        model_path = resolve_project_path(cfg.model_path)
        source = model_path if Path(model_path, "tokenizer_config.json").exists() else model_id
    else:
        source = model_id
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_path, model_cfg, dtype, device):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)


def build_training_args(cfg, max_steps, steps_per_epoch):
    optimizer_steps_per_epoch = max(1, math.ceil(steps_per_epoch / int(cfg.gradient_accumulation_steps)))
    warmup_steps = max(0, int(float(cfg.warmup_ratio) * optimizer_steps_per_epoch * float(cfg.num_epochs)))
    use_cuda_precision = torch.cuda.is_available()
    return transformers.TrainingArguments(
        per_device_train_batch_size=int(cfg.batch_size),
        per_device_eval_batch_size=int(cfg.batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        warmup_steps=warmup_steps,
        max_steps=int(max_steps),
        learning_rate=float(cfg.lr),
        bf16=bool(cfg.bf16) and use_cuda_precision,
        fp16=bool(cfg.fp16) and use_cuda_precision,
        bf16_full_eval=bool(cfg.bf16) and use_cuda_precision,
        fp16_full_eval=bool(cfg.fp16) and use_cuda_precision,
        logging_steps=max(1, int(cfg.log_steps)),
        logging_dir=str(Path(cfg.save_dir) / "logs"),
        output_dir=str(cfg.save_dir),
        optim=str(cfg.optim),
        save_strategy="steps" if bool(cfg.save_model) and not bool(cfg.eval_only) else "no",
        save_steps=max(1, steps_per_epoch),
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=float(cfg.weight_decay),
        max_grad_norm=float(cfg.max_grad_norm),
        eval_steps=max(1, steps_per_epoch),
        eval_strategy="steps" if bool(cfg.eval_while_train) else "no",
        seed=int(cfg.seed),
        report_to=[],
        remove_unused_columns=False,
    )


def make_eval_cfg(cfg):
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg.eval, resolve=True))
    languages = [str(cfg.source_language)]
    for language in list(cfg.target_languages):
        language = str(language)
        if language not in languages:
            languages.append(language)
    eval_cfg.languages = languages
    eval_cfg.model_family = cfg.model_family
    eval_cfg.model_path = cfg.save_dir
    eval_cfg.save_dir = str(Path(cfg.save_dir) / "eval_results")
    eval_cfg.tokenizer_path = cfg.get("tokenizer_path", None)
    return eval_cfg


class LingTeaTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, cfg=None, **kwargs):
        self.teacher_model = teacher_model
        self.cfg = cfg
        super().__init__(*args, **kwargs)
        if self.teacher_model is None:
            raise ValueError("LingTeaTrainer requires a teacher/reference model.")
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs, logs = compute_lingtea_loss(model, self.teacher_model, inputs, self.cfg)
        if model.training:
            self.log({key: float(value.detach().float().cpu()) for key, value in logs.items()})
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        source_inputs = inputs["forget_source"] if isinstance(inputs, dict) else inputs
        input_ids, labels, attention_mask = (tensor.to(model.device) for tensor in source_inputs)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, use_cache=False)
            if prediction_loss_only:
                return outputs.loss, None, None
        return outputs.loss, outputs.logits, labels


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.environ["WANDB_DISABLED"] = "true"
    os.chdir(PROJECT_ROOT)
    set_seed(int(cfg.seed))

    if cfg.model_path is None:
        cfg.model_path = get_model_identifiers_from_yaml(cfg.model_family)["ft_model_path"]
    cfg.model_path = resolve_project_path(cfg.model_path)
    cfg.save_dir = resolve_project_path(cfg.save_dir)
    ensure_save_dir(cfg)

    with open(Path(cfg.save_dir) / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = load_tokenizer(cfg, model_id)

    dataset = LingTeaDataset(cfg, tokenizer=tokenizer, project_root=PROJECT_ROOT)
    steps_per_epoch = max(1, math.ceil(len(dataset) / int(cfg.batch_size)))
    if cfg.max_steps is None:
        max_steps = max(1, math.ceil(float(cfg.num_epochs) * steps_per_epoch))
    else:
        max_steps = int(cfg.max_steps)

    print(f"dataset size:    {len(dataset)}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"max_steps:       {max_steps}")
    print(f"save_dir:        {cfg.save_dir}")

    dtype = pick_dtype(cfg)
    train_device = pick_device(cfg.gpu_train)
    try:
        teacher_device = pick_device(cfg.gpu_teacher)
    except ValueError:
        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            teacher_device = train_device
            print("Only one CUDA device is visible; placing teacher on the training GPU.")
        else:
            raise
    if train_device == teacher_device and torch.cuda.is_available():
        print("Student and teacher are on the same GPU; memory use will be high.")

    model = build_model(cfg.model_path, model_cfg, dtype, train_device)
    teacher_model = build_model(cfg.model_path, model_cfg, dtype, teacher_device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    model.generation_config.do_sample = True
    if bool(cfg.gradient_checkpointing) or model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    training_args = build_training_args(cfg, max_steps=max_steps, steps_per_epoch=steps_per_epoch)
    trainer = LingTeaTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=training_args,
        data_collator=lingtea_collator,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        cfg=cfg,
    )

    if bool(cfg.eval_only):
        print("eval_only=true: skipping training.")
    else:
        trainer.train()

    if bool(cfg.save_model) and not bool(cfg.eval_only):
        model.save_pretrained(cfg.save_dir, safe_serialization=True)
        tokenizer.save_pretrained(cfg.save_dir)

    if bool(cfg.run_eval_after_train):
        eval_cfg = make_eval_cfg(cfg)
        Path(eval_cfg.save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(eval_cfg.save_dir) / "resolved_eval_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(eval_cfg, resolve=True))
        model.eval()
        evaluate_languages(model, tokenizer, eval_cfg)

    with open(Path(cfg.save_dir) / "training_done.json", "w") as f:
        json.dump({"max_steps": max_steps, "dataset_size": len(dataset)}, f, indent=2)


if __name__ == "__main__":
    main()

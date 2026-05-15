"""End-to-end UNLEARN task-matrix training runner."""

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
import transformers
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed

from unlearning_methods.unlearn_UNLEARN.dataloader import UNLEARNDataset, unlearn_collator
from unlearning_methods.unlearn_UNLEARN.low_rank import (
    apply_task_matrices,
    assert_only_task_matrices_trainable,
    save_task_matrices,
    task_parameter_count,
    wrap_task_matrices,
)
from unlearning_methods.unlearn_UNLEARN.loss import compute_unlearn_loss
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


def get_rank_info():
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def configure_cuda_for_rank():
    _, local_rank, _ = get_rank_info()
    if torch.cuda.is_available() and os.environ.get("LOCAL_RANK") is not None:
        torch.cuda.set_device(local_rank)
    return local_rank


def pick_device(index):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    index = int(index)
    if index >= torch.cuda.device_count():
        raise ValueError(f"Requested cuda:{index}, but only {torch.cuda.device_count()} visible CUDA devices exist.")
    return torch.device(f"cuda:{index}")


def is_deepspeed_enabled(cfg):
    return bool(cfg.get("use_deepspeed", False)) and cfg.get("deepspeed_config", None) is not None


def pick_dtype(cfg):
    if not torch.cuda.is_available():
        return torch.float32
    if bool(cfg.bf16):
        return torch.bfloat16
    if bool(cfg.fp16):
        return torch.float16
    return torch.float32


def ensure_save_dir(cfg, rank):
    save_dir = Path(cfg.save_dir)
    setup_marker = save_dir / ".setup_done"
    if rank == 0:
        if save_dir.exists() and any(save_dir.iterdir()) and not bool(cfg.overwrite_dir):
            raise FileExistsError(
                f"save_dir already exists and is not empty: {save_dir}\n"
                "Set overwrite_dir=true or choose a new save_dir."
            )
        if save_dir.exists() and bool(cfg.overwrite_dir):
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        setup_marker.write_text("ok\n")
    else:
        while not setup_marker.exists():
            time.sleep(0.25)


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


def build_model(model_path, model_cfg, dtype, device=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if device is not None:
        model = model.to(device)
    return model


def resolve_teacher_device(cfg, local_rank, train_device):
    value = str(cfg.get("teacher_device", "auto")).lower()
    if value == "auto":
        if is_deepspeed_enabled(cfg) and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            return pick_device(local_rank)
        if torch.cuda.is_available() and torch.cuda.device_count() > int(cfg.gpu_teacher):
            return pick_device(cfg.gpu_teacher)
        return train_device
    if value == "local_rank":
        return pick_device(local_rank)
    if value == "same_as_student":
        return train_device
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda:"):
        return pick_device(int(value.split(":", 1)[1]))
    return pick_device(value)


def build_training_args(cfg, max_steps, optimizer_steps_per_epoch, world_size):
    warmup_steps = max(0, int(float(cfg.warmup_ratio) * optimizer_steps_per_epoch * float(cfg.num_epochs)))
    use_cuda_precision = torch.cuda.is_available()
    deepspeed_config = resolve_project_path(cfg.deepspeed_config) if is_deepspeed_enabled(cfg) else None
    per_device_batch = max(1, int(cfg.batch_size) // max(1, int(world_size)))
    return transformers.TrainingArguments(
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
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
        output_dir=str(Path(cfg.save_dir) / "trainer_checkpoints"),
        optim=str(cfg.optim),
        save_strategy="no",
        ddp_find_unused_parameters=False,
        weight_decay=float(cfg.weight_decay),
        max_grad_norm=float(cfg.max_grad_norm),
        eval_strategy="no",
        seed=int(cfg.seed),
        report_to=[],
        remove_unused_columns=False,
        deepspeed=deepspeed_config,
    )


class UNLEARNTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, cfg=None, **kwargs):
        self.teacher_model = teacher_model
        self.cfg = cfg
        super().__init__(*args, **kwargs)
        if bool(self.cfg.use_retain_regularization):
            if self.teacher_model is None:
                raise ValueError("use_retain_regularization=true requires a teacher model.")
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs, logs = compute_unlearn_loss(model, self.teacher_model, inputs, self.cfg)
        if model.training:
            self.log({key: float(value.detach().float().cpu()) for key, value in logs.items()})
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        forget_inputs = inputs["forget"] if isinstance(inputs, dict) else inputs
        device = next(model.parameters()).device
        input_ids, labels, attention_mask = (tensor.to(device) for tensor in forget_inputs)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, use_cache=False)
        return outputs.loss, None if prediction_loss_only else outputs.logits, labels


def save_final_unlearned_model(cfg, tokenizer, model_cfg):
    clean_model = build_model(resolve_project_path(cfg.model_path), model_cfg, pick_dtype(cfg), device=None)
    changed = apply_task_matrices(
        clean_model,
        resolve_project_path(cfg.save_task_matrix_dir),
        unlearn_scale=float(cfg.unlearn_scale),
        dtype=torch.float32,
    )
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    clean_model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)
    return changed


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.environ["WANDB_DISABLED"] = "true"
    os.chdir(PROJECT_ROOT)
    local_rank = configure_cuda_for_rank()
    rank, _, world_size = get_rank_info()
    set_seed(int(cfg.seed))

    if cfg.model_path is None:
        cfg.model_path = get_model_identifiers_from_yaml(cfg.model_family)["ft_model_path"]
    cfg.model_path = resolve_project_path(cfg.model_path)
    cfg.teacher_path = resolve_project_path(cfg.teacher_path)
    cfg.save_dir = resolve_project_path(cfg.save_dir)
    cfg.save_task_matrix_dir = resolve_project_path(cfg.save_task_matrix_dir)
    ensure_save_dir(cfg, rank)

    if rank == 0:
        with open(Path(cfg.save_dir) / "resolved_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    tokenizer = load_tokenizer(cfg, model_cfg["hf_key"])
    dataset = UNLEARNDataset(cfg, tokenizer=tokenizer, project_root=PROJECT_ROOT)

    micro_steps_per_epoch = max(1, math.ceil(len(dataset) / (int(cfg.batch_size) * max(1, world_size))))
    optimizer_steps_per_epoch = max(1, math.ceil(micro_steps_per_epoch / int(cfg.gradient_accumulation_steps)))
    max_steps = int(cfg.max_steps) if cfg.max_steps is not None else max(1, math.ceil(float(cfg.num_epochs) * optimizer_steps_per_epoch))

    dtype = pick_dtype(cfg)
    train_device = pick_device(local_rank if is_deepspeed_enabled(cfg) and world_size > 1 else cfg.gpu_train)
    student_device = None if is_deepspeed_enabled(cfg) else train_device
    model = build_model(cfg.model_path, model_cfg, dtype, student_device)
    model.config.use_cache = False
    if bool(cfg.gradient_checkpointing) or model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    wrapped_modules = wrap_task_matrices(
        model,
        target_modules=list(cfg.target_modules),
        rank=int(cfg.rank),
        alpha=float(cfg.alpha),
        target_layer_start=cfg.target_layer_start,
        target_layer_end=cfg.target_layer_end,
    )
    trainable_names = assert_only_task_matrices_trainable(model)
    if not wrapped_modules:
        raise ValueError("No target modules were wrapped. Check target_modules and model architecture.")

    teacher_model = None
    if bool(cfg.use_retain_regularization):
        teacher_device = resolve_teacher_device(cfg, local_rank, train_device)
        teacher_model = build_model(cfg.teacher_path, model_cfg, dtype, teacher_device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    if rank == 0:
        print(f"world_size:                {world_size}")
        print(f"dataset size:              {len(dataset)}")
        print(f"optimizer_steps_per_epoch: {optimizer_steps_per_epoch}")
        print(f"max_steps:                 {max_steps}")
        print(f"wrapped modules:           {len(wrapped_modules)}")
        print(f"trainable params:          {task_parameter_count(model)}")
        print(f"trainable tensors:         {len(trainable_names)}")
        print(f"save_dir:                  {cfg.save_dir}")
        with open(Path(cfg.save_dir) / "wrapped_modules.json", "w") as f:
            json.dump(wrapped_modules, f, indent=2)

    training_args = build_training_args(
        cfg,
        max_steps=max_steps,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        world_size=world_size,
    )
    trainer = UNLEARNTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=training_args,
        data_collator=unlearn_collator,
        teacher_model=teacher_model,
        cfg=cfg,
    )

    if bool(cfg.eval_only):
        print("eval_only=true: skipping training.")
    else:
        trainer.train()

    if bool(cfg.save_model) and not bool(cfg.eval_only):
        trainer.accelerator.wait_for_everyone()
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        if trainer.is_world_process_zero():
            save_task_matrices(unwrapped, cfg.save_task_matrix_dir)
            changed = save_final_unlearned_model(cfg, tokenizer, model_cfg)
            with open(Path(cfg.save_dir) / "training_done.json", "w") as f:
                json.dump(
                    {
                        "max_steps": max_steps,
                        "dataset_size": len(dataset),
                        "world_size": world_size,
                        "wrapped_modules": len(wrapped_modules),
                        "task_parameter_count": task_parameter_count(unwrapped),
                        "merged_modules": changed,
                    },
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    main()

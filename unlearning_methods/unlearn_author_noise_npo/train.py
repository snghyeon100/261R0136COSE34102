"""End-to-end author-name-noise NPO unlearning runner.

Run from the repo root with:
    python unlearning_methods/unlearn_author_noise_npo/train.py
"""

import copy
import csv
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import deepspeed
import hydra
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed
from transformers.integrations.deepspeed import deepspeed_init

from evaluate_util import get_all_evals, get_dataloader
from unlearning_methods.unlearn_author_noise_npo.dataloader import AuthorNoiseNPODataset, author_noise_npo_collator
from unlearning_methods.unlearn_author_noise_npo.loss import (
    compute_author_noise_npo_loss,
    compute_batch_nll,
    compute_dpo_loss,
    compute_dpo_loss_with_ref_nll,
    compute_noisy_forget_npo_loss,
    compute_noisy_forget_npo_loss_with_ref_nll,
)
from utils import get_forget_quality, get_model_identifiers_from_yaml, get_model_utility, merge_dicts


def resolve_project_path(path):
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((PROJECT_ROOT / path).resolve())
    return path


class AuthorNoiseNPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        oracle_model=None,
        eval_cfg=None,
        tokenizer=None,
        beta=1.0,
        gamma=1.0,
        alpha=1.0,
        clean_npo_weight=1.0,
        noisy_npo_weight=1.0,
        noise_sigma=0.1,
        language="en",
        **kwargs,
    ):
        self.oracle_model = oracle_model
        self.eval_cfg = eval_cfg
        self.tokenizer = tokenizer
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.clean_npo_weight = clean_npo_weight
        self.noisy_npo_weight = noisy_npo_weight
        self.noise_sigma = noise_sigma
        self.language = language
        super().__init__(*args, **kwargs)
        if self.oracle_model is not None:
            self.oracle_model.eval()

    def _wrap_model(self, model, training=True, dataloader=None):
        return model

    def e_prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None and hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs, logs = compute_author_noise_npo_loss(
            model,
            self.oracle_model,
            inputs,
            beta=self.beta,
            gamma=self.gamma,
            alpha=self.alpha,
            clean_npo_weight=self.clean_npo_weight,
            noisy_npo_weight=self.noisy_npo_weight,
            noise_sigma=self.noise_sigma,
        )
        if model.training:
            self.log({key: float(value.detach().float().cpu()) for key, value in logs.items()})
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Backward each loss term separately to reduce peak activation memory."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        grad_accum = max(1, int(self.args.gradient_accumulation_steps))
        logs = {}
        total_detached = None

        def _accumulate(name, weighted_loss):
            nonlocal total_detached
            loss_for_backward = weighted_loss / grad_accum
            self.accelerator.backward(loss_for_backward)
            detached = weighted_loss.detach()
            total_detached = detached if total_detached is None else total_detached + detached
            logs[name] = float(detached.float().cpu())

        forget_inputs = inputs["forget"]
        has_precomputed_ref = "reference_nll" in inputs and torch.isfinite(inputs["reference_nll"]).all()

        with self.compute_loss_context_manager():
            if float(self.clean_npo_weight) != 0:
                if has_precomputed_ref:
                    clean_loss, _ = compute_dpo_loss_with_ref_nll(
                        model=model,
                        lose_inputs=forget_inputs,
                        ref_nll=inputs["reference_nll"],
                        beta=self.beta,
                    )
                else:
                    clean_loss, _ = compute_dpo_loss(
                        model=model,
                        ref_model=self.oracle_model,
                        win_inputs=None,
                        lose_inputs=forget_inputs,
                        beta=self.beta,
                    )
                logs["loss_clean_npo_raw"] = float(clean_loss.detach().float().cpu())
                _accumulate("loss_clean_npo", float(self.gamma) * float(self.clean_npo_weight) * clean_loss)

            if float(self.noisy_npo_weight) != 0:
                if has_precomputed_ref:
                    noisy_loss, _, trigger_ratio = compute_noisy_forget_npo_loss_with_ref_nll(
                        model=model,
                        forget_inputs=forget_inputs,
                        trigger_mask=inputs["trigger_mask"],
                        has_trigger=inputs["has_trigger"],
                        ref_nll=inputs["reference_nll"],
                        beta=self.beta,
                        noise_sigma=self.noise_sigma,
                    )
                else:
                    noisy_loss, _, trigger_ratio = compute_noisy_forget_npo_loss(
                        model=model,
                        ref_model=self.oracle_model,
                        forget_inputs=forget_inputs,
                        trigger_mask=inputs["trigger_mask"],
                        has_trigger=inputs["has_trigger"],
                        beta=self.beta,
                        noise_sigma=self.noise_sigma,
                    )
                logs["loss_noisy_npo_raw"] = float(noisy_loss.detach().float().cpu())
                logs["trigger_batch_ratio"] = float(trigger_ratio.detach().float().cpu())
                _accumulate("loss_noisy_npo", float(self.gamma) * float(self.noisy_npo_weight) * noisy_loss)

            if float(self.alpha) != 0:
                retain_input_ids, retain_labels, retain_attention_mask = (
                    tensor.to(next(model.parameters()).device)
                    for tensor in inputs["retain"]
                )
                retain_outputs = model(
                    retain_input_ids,
                    labels=retain_labels,
                    attention_mask=retain_attention_mask,
                    use_cache=False,
                )
                logs["loss_retain_raw"] = float(retain_outputs.loss.detach().float().cpu())
                _accumulate("loss_retain", float(self.alpha) * retain_outputs.loss)

        if total_detached is None:
            total_detached = next(model.parameters()).sum().detach() * 0.0
        logs["loss_total"] = float(total_detached.float().cpu())
        self.log(logs)
        return total_detached / grad_accum

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list, self.eval_cfg.split)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            if self.is_fsdp_enabled:
                self.model = model
            if model is not self.model:
                self.model_wrapped = model
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()

        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg
        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        aggregated_eval_logs = {}

        with torch.no_grad():
            for folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key in zip(
                eval_cfg.data_path,
                eval_cfg.split_list,
                eval_cfg.question_key,
                eval_cfg.answer_key,
                eval_cfg.eval_task,
                eval_cfg.base_answer_key,
                eval_cfg.perturbed_answer_key,
            ):
                world_size = self.accelerator.num_processes
                if eval_task == "eval_log_forget":
                    split = eval_cfg.split
                print(f"Working on eval task {eval_task} with split {split}")
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                save_filename = (
                    save_filename
                    if world_size == 1
                    else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                )
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
                    eval_cfg,
                    eval_task,
                    self.tokenizer,
                    folder,
                    split,
                    question_key,
                    answer_key,
                    base_answer_key,
                    perturbed_answer_key,
                    language=self.language,
                )
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)

                eval_logs = get_all_evals(
                    eval_cfg,
                    model,
                    self.tokenizer,
                    eval_task,
                    eval_dataloader,
                    base_eval_dataloader,
                    perturb_dataloader,
                    normalize_gt=False,
                )
                with open(save_filename, "w") as f:
                    json.dump(eval_logs, f, indent=4)
                if world_size == 1:
                    aggregated_eval_logs[f"{eval_task}.json"] = eval_logs

            self.accelerator.wait_for_everyone()
            world_size = self.accelerator.num_processes
            if world_size > 1 and self.accelerator.is_local_main_process:
                for eval_task in eval_cfg.eval_task:
                    eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))
                    for i in range(1, world_size):
                        filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                        eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                    aggregated_eval_logs[f"{eval_task}.json"] = eval_logs

                    new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                    with open(new_save_filename, "w") as f:
                        json.dump(eval_logs, f, indent=4)
                    for i in range(world_size):
                        os.remove(os.path.join(curr_save_dir, f"{eval_task}_{i}.json"))

            if self.accelerator.is_local_main_process:
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")
                with open(aggregated_eval_log_filename, "w") as f:
                    json.dump(aggregated_eval_logs, f, indent=4)

                if eval_cfg.retain_result is not None:
                    model_utility = get_model_utility(aggregated_eval_logs)
                    retain_result = json.load(open(eval_cfg.retain_result, "r"))
                    forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
                    aggregate_stat = {**model_utility, **forget_quality}

                    with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), "w") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=list(aggregate_stat.keys()))
                        writer.writeheader()
                        writer.writerow(aggregate_stat)


def build_training_args(cfg, max_steps, steps_per_epoch, batch_size):
    num_cuda_devices = max(1, torch.cuda.device_count())
    return transformers.TrainingArguments(
        per_device_train_batch_size=max(1, batch_size // num_cuda_devices),
        per_device_eval_batch_size=max(1, batch_size // num_cuda_devices),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=use_bf16(cfg),
        fp16=use_fp16(cfg),
        bf16_full_eval=use_bf16(cfg),
        fp16_full_eval=use_fp16(cfg),
        logging_steps=max(1, max_steps // 20),
        logging_dir=str(Path(cfg.save_dir) / "logs"),
        output_dir=cfg.save_dir,
        optim=str(cfg.get("optim", "paged_adamw_8bit")),
        save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
        save_steps=steps_per_epoch,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=cfg.weight_decay,
        eval_steps=steps_per_epoch,
        eval_strategy="steps" if cfg.eval_while_train else "no",
        seed=cfg.seed,
        remove_unused_columns=False,
    )


def resolve_torch_device(value):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    value = str(value)
    if value == "auto":
        return torch.device("cuda:0")
    return torch.device(value)


def model_dtype(cfg):
    if not torch.cuda.is_available():
        return torch.float32
    if bool(cfg.get("bf16", True)) and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if bool(cfg.get("fp16", False)):
        return torch.float16
    if bool(cfg.get("bf16", True)) and not torch.cuda.is_bf16_supported():
        print("bf16 requested but unsupported on this GPU; using fp16.")
        return torch.float16
    return torch.float32


def use_bf16(cfg):
    return bool(cfg.get("bf16", True)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def use_fp16(cfg):
    return bool(cfg.get("fp16", False)) and torch.cuda.is_available() or (
        bool(cfg.get("bf16", True)) and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    )


def parse_torch_dtype(value):
    value = str(value).lower()
    if value in {"float32", "fp32"}:
        return torch.float32
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def collect_lora_targets(model, target_leaves):
    if target_leaves is None:
        target_leaves = sorted(
            {
                name.split(".")[-1]
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and name.split(".")[-1] != "lm_head"
            }
        )
    target_leaves = set(str(name) for name in target_leaves)
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.split(".")[-1] in target_leaves and "lm_head" not in name:
            targets.append(name)
    return targets


def cast_trainable_parameters(model, dtype):
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=dtype)
            if param.grad is not None:
                param.grad = param.grad.to(dtype=dtype)


def count_trainable_parameters(model):
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return {
        "trainable": trainable,
        "total": total,
        "trainable_fraction": trainable / max(1, total),
    }


def attach_lora_if_enabled(model, cfg):
    if not bool(cfg.get("use_lora", True)):
        print("LoRA disabled: full model parameters remain trainable.")
        return model, None

    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = collect_lora_targets(model, cfg.get("lora_target_modules", None))
    if not target_modules:
        raise ValueError("No LoRA target modules found.")
    print(f"LoRA target module count: {len(target_modules)}")
    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        target_modules=target_modules,
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    cast_trainable_parameters(model, parse_torch_dtype(cfg.get("trainable_param_dtype", "float32")))
    model.print_trainable_parameters()
    return model, target_modules


def precompute_reference_nlls(cfg, dataset, model_cfg):
    device = resolve_torch_device(cfg.get("reference_device", "cuda:0"))
    dtype = model_dtype(cfg)
    print(f"Precomputing reference NLLs on {device} and then freeing oracle model.")
    oracle_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    oracle_model.eval()
    for param in oracle_model.parameters():
        param.requires_grad = False

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("reference_batch_size", 1)),
        shuffle=False,
        collate_fn=author_noise_npo_collator,
    )
    reference_nlls = [None] * len(dataset)
    with torch.inference_mode():
        for batch in loader:
            nll, _ = compute_batch_nll(oracle_model, batch["forget"])
            for idx, value in zip(batch["index"].tolist(), nll.detach().float().cpu().tolist()):
                reference_nlls[int(idx)] = float(value)

    del oracle_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    missing = [idx for idx, value in enumerate(reference_nlls) if value is None]
    if missing:
        raise RuntimeError(f"Missing reference NLLs for indices: {missing}")
    dataset.set_reference_nlls(reference_nlls)
    print("Reference NLL precompute complete.")
    return reference_nlls


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    print(f"num_devices: {num_devices}")

    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]
    cfg.model_path = resolve_project_path(cfg.model_path)
    cfg.save_dir = resolve_project_path(cfg.save_dir)

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = AuthorNoiseNPODataset(
        cfg.data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=500,
        split=cfg.split,
        language=cfg.language,
        authors=list(cfg.authors),
        skip_missing_trigger=bool(cfg.skip_missing_trigger),
    )
    stats = dataset.trigger_stats()
    print(f"trigger stats: {stats}")
    batch_size = cfg.batch_size
    steps_per_epoch = max(1, len(dataset) // (batch_size * cfg.gradient_accumulation_steps * num_devices))
    if cfg.get("max_steps", None) is None:
        max_steps = max(1, int(cfg.num_epochs * len(dataset)) // (batch_size * cfg.gradient_accumulation_steps))
    else:
        max_steps = int(cfg.max_steps)
    print(f"max_steps: {max_steps}")
    print("batch_size:", batch_size)

    oracle_model = None
    if bool(cfg.get("precompute_reference_nll", True)):
        precompute_reference_nlls(cfg, dataset, model_cfg)
    else:
        print("precompute_reference_nll=false: keeping oracle model in memory during training.")
        oracle_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
            torch_dtype=model_dtype(cfg),
            trust_remote_code=True,
        ).to(resolve_torch_device(cfg.get("reference_device", "cuda:1")))
        oracle_model.eval()
        for param in oracle_model.parameters():
            param.requires_grad = False

    training_args = build_training_args(cfg, max_steps, steps_per_epoch, batch_size)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=model_dtype(cfg),
        trust_remote_code=True,
    ).to("cuda:0")

    model.generation_config.do_sample = True
    if bool(cfg.get("gradient_checkpointing", False)) or model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model, lora_targets = attach_lora_if_enabled(model, cfg)
    if bool(cfg.get("gradient_checkpointing", False)) or model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False

    trainable_info = count_trainable_parameters(model)
    trainable_info["lora_targets"] = lora_targets
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.save_dir) / "trainable_parameters.json", "w") as f:
        json.dump(trainable_info, f, indent=2)

    trainer = AuthorNoiseNPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=author_noise_npo_collator,
        oracle_model=oracle_model,
        eval_cfg=cfg.eval,
        beta=cfg.beta,
        gamma=cfg.gamma,
        alpha=cfg.alpha,
        clean_npo_weight=cfg.clean_npo_weight,
        noisy_npo_weight=cfg.noisy_npo_weight,
        noise_sigma=cfg.noise_sigma,
        language=cfg.language,
    )

    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()

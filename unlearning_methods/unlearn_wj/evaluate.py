"""Evaluate Paired Counterfactual Equivalence ICUL without training.

Run from the NLP_rice root:
    python unlearning_methods/unlearn_wj/evaluate.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for entry in list(sys.path):
    if Path(entry or ".").resolve() == SCRIPT_DIR:
        sys.path.remove(entry)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from omegaconf import OmegaConf

from evaluate_util import (
    build_summary_accumulator,
    build_summary_row_from_accumulator,
    cfg_get,
    get_all_evals,
    language_eval_cfg,
    list_cfg,
    load_eval_model,
    load_eval_tokenizer,
    merge_summary_accumulators,
    reinitialize_weights,
    resolve_project_path,
    write_case_study_files,
    write_summary_files,
)
from unlearning_methods.unlearn_wj.dataset import get_in_context_dataloader
from unlearning_methods.unlearn_wj.prompt_builder import normalize_condition
from utils import get_forget_quality, get_model_identifiers_from_yaml, get_model_utility


METHOD_CFG_KEYS = [
    "condition",
    "source_language",
    "source_forget_data_path",
    "source_forget_split",
    "source_forget_language",
    "source_forget_from_end",
    "source_forget_question_key",
    "source_forget_answer_key",
    "forget_profile_size",
    "counterfactual_answer_policy",
    "counterfactual_perturb_fallback_policy",
    "require_perturb_counterfactuals",
    "source_perturbed_forget_data_path",
    "source_perturbed_forget_split",
    "source_perturbed_forget_language",
    "source_perturbed_forget_from_end",
    "source_perturbed_forget_answer_key",
    "source_perturbed_candidate_index",
    "pair_forget_by_index",
    "retrieval_gate",
    "retrieval_gate_mode",
    "retrieval_gate_threshold",
    "retain_anchor_data_path",
    "retain_anchor_split",
    "retain_anchor_source_language",
    "retain_anchor_question_key",
    "retain_anchor_answer_key",
    "num_extra_forget_cases",
    "num_unpaired_forget_cases",
    "num_retain_anchors",
    "demo_seed",
    "answer_policy",
    "include_instruction",
    "max_demo_chars",
    "save_prompt_samples",
    "prompt_sample_limit",
]


def infer_conditions(cfg) -> list[str]:
    conditions = list_cfg(cfg_get(cfg, "conditions", None))
    if conditions:
        return [normalize_condition(condition) for condition in conditions]
    return [normalize_condition(cfg_get(cfg, "condition", "pce_icul"))]


def infer_languages(cfg) -> list[str]:
    languages = list_cfg(cfg_get(cfg, "languages", None))
    if languages:
        return languages
    return [cfg_get(cfg, "language", "ko")]


def method_language_eval_cfg(cfg, language: str, condition: str):
    eval_cfg = language_eval_cfg(cfg, language)
    eval_cfg.condition = normalize_condition(condition)
    for key in METHOD_CFG_KEYS:
        if key == "condition":
            continue
        value = cfg_get(cfg, key, None)
        if value is not None:
            setattr(eval_cfg, key, value)
    return eval_cfg


def maybe_write_prompt_samples(save_dir: Path, eval_dataloader, cfg, eval_task: str):
    if eval_task != "eval_log_forget" or not bool(cfg_get(cfg, "save_prompt_samples", False)):
        return

    dataset = getattr(eval_dataloader, "dataset", None)
    if dataset is None or not hasattr(dataset, "prompt_sample"):
        return

    limit = min(int(cfg_get(cfg, "prompt_sample_limit", 5)), len(dataset))
    samples = [dataset.prompt_sample(idx) for idx in range(limit)]
    with open(save_dir / "prompts_sample.md", "w") as f:
        f.write(f"# Prompt Samples ({cfg.condition}, {cfg.language})\n\n")
        for sample in samples:
            f.write(f"## index {sample['index']}\n\n")
            f.write("Prompt question:\n\n")
            f.write("```text\n")
            f.write(sample["prompt_question"])
            f.write("\n```\n\n")
            f.write("Gold answer:\n\n")
            f.write("```text\n")
            f.write(sample["gold_answer"])
            f.write("\n```\n\n")


def evaluate_one_language_condition(model, tokenizer, cfg, condition: str, language: str, save_dir: Path):
    eval_cfg = method_language_eval_cfg(cfg, language, condition)
    if len(eval_cfg.data_path) != len(eval_cfg.split_list) or len(eval_cfg.data_path) != len(eval_cfg.eval_task):
        raise ValueError("data_path, split_list, and eval_task must have the same length.")

    save_dir.mkdir(parents=True, exist_ok=True)
    save_raw_logs = bool(cfg_get(eval_cfg, "save_raw_logs", False))
    save_legacy_stat = bool(cfg_get(eval_cfg, "save_legacy_aggregate_stat", False))
    aggregated_eval_logs = {} if (save_raw_logs or save_legacy_stat) else None
    summary_accumulator = {}

    for folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key in zip(
        eval_cfg.data_path,
        eval_cfg.split_list,
        eval_cfg.question_key,
        eval_cfg.answer_key,
        eval_cfg.eval_task,
        eval_cfg.base_answer_key,
        eval_cfg.perturbed_answer_key,
    ):
        if eval_task == "eval_log_forget":
            split = eval_cfg.split

        print(f"[{condition}/{language}] Working on eval task {eval_task} with split {split}")
        save_filename = save_dir / f"{eval_task}.json"
        if save_raw_logs and save_filename.exists() and not eval_cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            with open(save_filename, "r") as f:
                eval_logs = json.load(f)
            merge_summary_accumulators(summary_accumulator, build_summary_accumulator({f"{eval_task}.json": eval_logs}))
            if aggregated_eval_logs is not None:
                aggregated_eval_logs[f"{eval_task}.json"] = eval_logs
            continue

        eval_dataloader, base_eval_dataloader, perturb_dataloader = get_in_context_dataloader(
            eval_cfg,
            eval_task,
            tokenizer,
            resolve_project_path(folder),
            split,
            question_key,
            answer_key,
            base_answer_key,
            perturbed_answer_key,
            language=language,
        )

        maybe_write_prompt_samples(save_dir, eval_dataloader, eval_cfg, eval_task)

        normalize_gt = "eval_log" not in eval_task
        eval_logs = get_all_evals(
            eval_cfg,
            model,
            tokenizer,
            eval_task,
            eval_dataloader,
            base_eval_dataloader,
            perturb_dataloader,
            normalize_gt=normalize_gt,
            language=language,
        )

        if save_raw_logs:
            with open(save_filename, "w") as f:
                json.dump(eval_logs, f, indent=4, ensure_ascii=False)

        if bool(cfg_get(eval_cfg, "save_case_studies", False)) and eval_task == "eval_log_forget":
            write_case_study_files(
                save_dir,
                language,
                eval_logs.get("case_study_candidates", []),
                eval_cfg,
                dataloader=eval_dataloader,
                model=model,
                tokenizer=tokenizer,
            )
            if not save_raw_logs:
                eval_logs.pop("case_study_candidates", None)

        merge_summary_accumulators(summary_accumulator, build_summary_accumulator({f"{eval_task}.json": eval_logs}))
        if aggregated_eval_logs is not None:
            aggregated_eval_logs[f"{eval_task}.json"] = eval_logs
        del eval_logs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if save_raw_logs and aggregated_eval_logs is not None:
        with open(save_dir / "eval_log_aggregated.json", "w") as f:
            json.dump(aggregated_eval_logs, f, indent=4, ensure_ascii=False)

    if save_legacy_stat and eval_cfg.retain_result is not None and aggregated_eval_logs is not None:
        retain_result = json.load(open(resolve_project_path(eval_cfg.retain_result), "r"))
        aggregate_stat = {**get_model_utility(aggregated_eval_logs), **get_forget_quality(aggregated_eval_logs, retain_result)}
        with open(save_dir / "aggregate_stat.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate_stat.keys()))
            writer.writeheader()
            writer.writerow(aggregate_stat)

    return summary_accumulator, aggregated_eval_logs


def evaluate_condition(model, tokenizer, cfg, condition: str, root_save_dir: Path):
    languages = infer_languages(cfg)
    multilingual = cfg_get(cfg, "languages", None) is not None
    condition_save_dir = root_save_dir / condition
    condition_save_dir.mkdir(parents=True, exist_ok=True)
    with open(condition_save_dir / "resolved_eval_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    save_raw_logs = bool(cfg_get(cfg, "save_raw_logs", False))
    all_logs = {} if save_raw_logs else None
    summary_rows = []
    total_accumulator = {}

    for language in languages:
        save_dir = condition_save_dir / language if multilingual else condition_save_dir
        language_accumulator, language_logs = evaluate_one_language_condition(
            model,
            tokenizer,
            cfg,
            condition,
            language,
            save_dir,
        )
        merge_summary_accumulators(total_accumulator, language_accumulator)
        summary_rows.append(build_summary_row_from_accumulator(language, language_accumulator))
        if save_raw_logs and language_logs is not None:
            all_logs[language] = language_logs
        del language_logs

    if multilingual and save_raw_logs:
        with open(condition_save_dir / "multilingual_aggregated.json", "w") as f:
            json.dump(all_logs, f, indent=4, ensure_ascii=False)

    if len(languages) > 1:
        summary_rows.append(build_summary_row_from_accumulator("total", total_accumulator))

    write_summary_files(condition_save_dir, summary_rows)
    return summary_rows


def write_condition_metadata(cfg, root_save_dir: Path, conditions: list[str]):
    payload = {
        "method_name": cfg_get(cfg, "method_name", "pce_icul"),
        "conditions": conditions,
        "languages": infer_languages(cfg),
        "source_language": cfg_get(cfg, "source_language", "en"),
        "counterfactual_answer_policy": cfg_get(cfg, "counterfactual_answer_policy", "source_perturb_first"),
        "source_perturbed_forget_data_path": cfg_get(cfg, "source_perturbed_forget_data_path", None),
        "source_perturbed_forget_split": cfg_get(cfg, "source_perturbed_forget_split", None),
        "source_perturbed_candidate_index": int(cfg_get(cfg, "source_perturbed_candidate_index", 0)),
        "retrieval_gate": cfg_get(cfg, "retrieval_gate", True),
        "retrieval_gate_mode": cfg_get(cfg, "retrieval_gate_mode", "eval_task"),
        "retrieval_gate_threshold": float(cfg_get(cfg, "retrieval_gate_threshold", 0.35)),
        "num_extra_forget_cases": int(cfg_get(cfg, "num_extra_forget_cases", 0)),
        "num_unpaired_forget_cases": int(cfg_get(cfg, "num_unpaired_forget_cases", 0)),
        "num_retain_anchors": int(cfg_get(cfg, "num_retain_anchors", 1)),
        "answer_policy": cfg_get(cfg, "answer_policy", "unknown_short"),
        "metric_contract": "All metric values are produced by the root evaluate_util.py metric path.",
    }
    with open(root_save_dir / "condition_metadata.json", "w") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.chdir(PROJECT_ROOT)
    os.environ["WANDB_DISABLED"] = "true"

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = load_eval_tokenizer(cfg, model_id)
    model = load_eval_model(cfg, model_cfg, model_id)

    if cfg.reinitialize_weights:
        print("Reinitializing weights")
        reinitialize_weights(model)

    conditions = infer_conditions(cfg)
    root_save_dir = Path(resolve_project_path(cfg.save_dir))
    root_save_dir.mkdir(parents=True, exist_ok=True)
    write_condition_metadata(cfg, root_save_dir, conditions)

    for condition in conditions:
        evaluate_condition(model, tokenizer, cfg, condition, root_save_dir)


if __name__ == "__main__":
    main()

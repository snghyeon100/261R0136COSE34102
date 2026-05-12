"""Context-injected datasets for Paired Counterfactual Equivalence ICUL."""

from __future__ import annotations

import random
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import datasets
import torch
from torch.utils.data import DataLoader, Dataset

from data_module import (
    convert_raw_data_to_model_format,
    custom_data_collator_with_indices,
    normalize_eval_text,
)
from evaluate_util import cfg_get
from unlearning_methods.unlearn_wj.prompt_builder import (
    QAExample,
    RetainAnchor,
    build_prompt,
    normalize_condition,
    unknown_answer,
)
from utils import add_dataset_index, get_model_identifiers_from_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((PROJECT_ROOT / path).resolve())
    return path


def _load_dataset(data_path: str, split_name: str | None = None):
    resolved = resolve_project_path(data_path)
    if resolved is not None and Path(resolved).exists():
        loaded = datasets.load_from_disk(resolved)
        if isinstance(loaded, datasets.DatasetDict):
            split = split_name if split_name in loaded else "train"
            return loaded[split]
        return loaded

    if split_name in {None, "", "train"}:
        loaded = datasets.load_dataset(data_path)
    else:
        loaded = datasets.load_dataset(data_path, split_name)
    if isinstance(loaded, datasets.DatasetDict):
        return loaded["train"]
    return loaded


def _ensure_index(data):
    if "index" in data.column_names:
        return data
    return add_dataset_index(data)


def _filter_language(data, language: str | None):
    if language is None or "language" not in data.column_names:
        return data
    indices = [idx for idx, row in enumerate(data) if row.get("language") == language]
    return data.select(indices)


def _first_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)


def _format_template(path_template: str, language: str) -> str:
    return str(path_template).format(language=language)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return int(value)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", str(text).lower()))


def _jaccard(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _choose_answer_candidate(value: Any, candidate_index: int = 0) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        index = min(max(int(candidate_index), 0), len(value) - 1)
        return str(value[index])
    return str(value)


def infer_forget_slot(question: str) -> str:
    """Coarse TOFU question slot used to choose type-compatible false answers."""

    q = str(question).lower()
    if "full name" in q or "notable author born" in q or ("who is" in q and "author born" in q):
        return "identity"
    if "gender" in q or "identif" in q:
        return "gender"
    if ("city" in q and "country" in q and "born" in q) or "birthplace" in q or ("where" in q and "born" in q):
        return "birthplace"
    if "parent" in q or "father" in q or "mother" in q or "vocation" in q or "profession" in q:
        return "parents"
    if "genre" in q:
        return "genre"
    if "book" in q or "written by" in q or "works" in q:
        return "books"
    if "award" in q or "recognition" in q or "honor" in q:
        return "awards"
    if "writing style" in q or "style" in q or "prose" in q:
        return "style"
    if "writing process" in q or "motivates" in q or "motivation" in q:
        return "process"
    if "impact" in q or "contribution" in q or "readers" in q or "globally" in q:
        return "impact"
    if "theme" in q or "message" in q:
        return "themes"
    if "background" in q or "upbringing" in q or "heritage" in q or "native" in q or "influence" in q:
        return "background"
    if "align" in q or "combine" in q or "incorporate" in q:
        return "genre_alignment"
    return "other"


def _examples_from_rows(
    data,
    language: str,
    question_key: str,
    answer_key: str,
    limit_from_end: int | None = None,
    profile_size: int = 20,
) -> list[QAExample]:
    if limit_from_end is not None:
        limit = min(int(limit_from_end), len(data))
        start = len(data) - limit
        data = data.select(range(start, len(data)))

    examples = []
    for local_idx, row in enumerate(data):
        question = str(row[question_key])
        examples.append(
            QAExample(
                question=question,
                answer=_first_text(row[answer_key]),
                language=language,
                index=local_idx,
                source_index=row.get("__index_level_0__", row.get("index", local_idx)),
                profile_id=local_idx // profile_size if profile_size > 0 else None,
                slot=infer_forget_slot(question),
            )
        )
    return examples


def _counterfactual_candidate(
    example: QAExample,
    examples: list[QAExample],
    policy: str,
    profile_size: int,
    aligned_perturbed_answers: list[str] | None = None,
    perturb_fallback_policy: str = "source_slot_swap",
) -> str:
    if not examples:
        return unknown_answer(example.language)
    if len(examples) == 1:
        return unknown_answer(example.language)

    if policy in {"source_perturb_first", "english_perturb_first", "perturb_first"}:
        if aligned_perturbed_answers and example.index < len(aligned_perturbed_answers):
            answer = aligned_perturbed_answers[example.index].strip()
            if answer:
                return answer
        policy = perturb_fallback_policy

    if policy in {"unknown_short", "unknown_factual", "abstain"}:
        return unknown_answer(example.language, policy)

    if policy == "source_profile_swap":
        offset = profile_size if 0 < profile_size < len(examples) else 1
        return examples[(example.index + offset) % len(examples)].answer

    if policy == "other_forget_answer":
        return examples[(example.index + 1) % len(examples)].answer

    same_slot = [
        candidate
        for candidate in examples
        if candidate.index != example.index
        and candidate.slot == example.slot
        and (example.profile_id is None or candidate.profile_id != example.profile_id)
    ]
    if same_slot:
        return same_slot[0].answer

    other_profile = [
        candidate
        for candidate in examples
        if candidate.index != example.index
        and (example.profile_id is None or candidate.profile_id != example.profile_id)
    ]
    if other_profile:
        return other_profile[0].answer

    return examples[(example.index + 1) % len(examples)].answer


def _assign_counterfactual_answers(
    examples: list[QAExample],
    policy: str,
    profile_size: int,
    aligned_perturbed_answers: list[str] | None = None,
    perturb_fallback_policy: str = "source_slot_swap",
) -> list[QAExample]:
    return [
        replace(
            example,
            counterfactual_answer=_counterfactual_candidate(
                example,
                examples,
                policy,
                profile_size,
                aligned_perturbed_answers=aligned_perturbed_answers,
                perturb_fallback_policy=perturb_fallback_policy,
            ),
        )
        for example in examples
    ]


def load_source_perturbed_counterfactuals(cfg, expected_count: int | None = None) -> list[str]:
    data_path = cfg_get(cfg, "source_perturbed_forget_data_path", None)
    if data_path is None:
        if _as_bool(cfg_get(cfg, "require_perturb_counterfactuals", False)):
            raise ValueError("source_perturbed_forget_data_path is required for perturb-aligned counterfactuals.")
        return []

    try:
        data = _load_dataset(
            data_path,
            cfg_get(cfg, "source_perturbed_forget_split", "forget01_perturbed"),
        )
    except Exception as exc:
        if _as_bool(cfg_get(cfg, "require_perturb_counterfactuals", True)):
            raise RuntimeError(
                "Failed to load source perturbed forget data. "
                "Set source_perturbed_forget_data_path to a local English forget01_perturbed dataset, "
                "or set counterfactual_answer_policy=source_slot_swap to use the older fallback."
            ) from exc
        return []

    language = cfg_get(cfg, "source_perturbed_forget_language", cfg_get(cfg, "source_language", "en"))
    data = _filter_language(data, language)
    limit_from_end = _optional_int(cfg_get(cfg, "source_perturbed_forget_from_end", None))
    if limit_from_end is not None:
        limit = min(limit_from_end, len(data))
        data = data.select(range(len(data) - limit, len(data)))

    answer_key = cfg_get(cfg, "source_perturbed_forget_answer_key", "perturbed_answer")
    if answer_key not in data.column_names:
        if _as_bool(cfg_get(cfg, "require_perturb_counterfactuals", True)):
            raise KeyError(f"{answer_key!r} not found in source perturbed forget data columns: {data.column_names}")
        return []

    candidate_index = int(cfg_get(cfg, "source_perturbed_candidate_index", 0))
    answers = [_choose_answer_candidate(row[answer_key], candidate_index) for row in data]
    if expected_count is not None and len(answers) < expected_count and _as_bool(cfg_get(cfg, "require_perturb_counterfactuals", True)):
        raise ValueError(
            f"Expected at least {expected_count} perturb counterfactuals, but loaded {len(answers)}."
        )
    return answers


def load_source_forget_pool(cfg) -> list[QAExample]:
    language = cfg_get(cfg, "source_forget_language", cfg_get(cfg, "source_language", "en"))
    profile_size = int(cfg_get(cfg, "forget_profile_size", 20))
    counterfactual_policy = cfg_get(cfg, "counterfactual_answer_policy", "source_perturb_first")
    data = _load_dataset(
        cfg_get(cfg, "source_forget_data_path", "./dataset/full_merged_all_10_lang"),
        cfg_get(cfg, "source_forget_split", "train"),
    )
    data = _filter_language(data, language)
    examples = _examples_from_rows(
        data,
        language=language,
        question_key=cfg_get(cfg, "source_forget_question_key", "question"),
        answer_key=cfg_get(cfg, "source_forget_answer_key", "answer"),
        limit_from_end=_optional_int(cfg_get(cfg, "source_forget_from_end", 40)),
        profile_size=profile_size,
    )
    aligned_perturbed_answers = []
    if counterfactual_policy in {"source_perturb_first", "english_perturb_first", "perturb_first"}:
        aligned_perturbed_answers = load_source_perturbed_counterfactuals(cfg, expected_count=len(examples))
    return _assign_counterfactual_answers(
        examples,
        policy=counterfactual_policy,
        profile_size=profile_size,
        aligned_perturbed_answers=aligned_perturbed_answers,
        perturb_fallback_policy=cfg_get(cfg, "counterfactual_perturb_fallback_policy", "source_slot_swap"),
    )


def _rows_by_language(data, language: str, question_key: str, answer_key: str) -> list[tuple[str, str]]:
    rows = _filter_language(data, language)
    return [
        (str(row[question_key]), _first_text(row[answer_key]))
        for row in rows
    ]


def load_retain_anchor_pool(cfg, target_language: str) -> list[RetainAnchor]:
    data = _load_dataset(
        cfg_get(cfg, "retain_anchor_data_path", cfg_get(cfg, "bridge_data_path", "./dataset/retain99_merged_all_10_lang")),
        cfg_get(cfg, "retain_anchor_split", cfg_get(cfg, "bridge_split", "train")),
    )
    question_key = cfg_get(cfg, "retain_anchor_question_key", cfg_get(cfg, "bridge_question_key", "question"))
    answer_key = cfg_get(cfg, "retain_anchor_answer_key", cfg_get(cfg, "bridge_answer_key", "answer"))
    source_language = cfg_get(cfg, "retain_anchor_source_language", cfg_get(cfg, "source_language", "en"))

    source_rows = _rows_by_language(data, source_language, question_key, answer_key)
    target_rows = _rows_by_language(data, target_language, question_key, answer_key)
    limit = min(len(source_rows), len(target_rows))
    anchors = []
    for idx in range(limit):
        source_question, source_answer = source_rows[idx]
        target_question, target_answer = target_rows[idx]
        anchors.append(
            RetainAnchor(
                source_question=source_question,
                source_answer=source_answer,
                target_question=target_question,
                target_answer=target_answer,
                source_language=source_language,
                target_language=target_language,
                index=idx,
            )
        )
    return anchors


def _sample_pool(pool: list[Any], count: int, seed: int, exclude_index: int | None = None) -> list[Any]:
    if count <= 0 or not pool:
        return []
    candidates = [item for item in pool if exclude_index is None or getattr(item, "index", None) != exclude_index]
    if not candidates:
        candidates = list(pool)
    if len(candidates) <= count:
        return candidates[:count]
    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(candidates)), count)
    return [candidates[idx] for idx in selected_indices]


class InContextQAStatDataset(Dataset):
    """Drop-in replacement for `TextDatasetQAStat` with contextual questions."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_family,
        max_length=512,
        split=None,
        question_key="question",
        answer_key="answer",
        language="en",
        cfg=None,
        condition="no_context",
        eval_task="eval_log",
        unicode_normalization=None,
        normalize_languages=None,
        source_forget_pool: list[QAExample] | None = None,
        retain_anchor_pool: list[RetainAnchor] | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language = language
        self.cfg = cfg
        self.condition = normalize_condition(condition)
        self.eval_task = eval_task
        self.unicode_normalization = unicode_normalization
        self.normalize_languages = normalize_languages
        self.data = _ensure_index(_load_dataset(data_path, split))
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key
        self.source_forget_pool = source_forget_pool or []
        self.retain_anchor_pool = retain_anchor_pool or []
        self.source_forget_token_sets = [_token_set(example.question) for example in self.source_forget_pool]

    def __len__(self):
        return len(self.data)

    def _demo_seed(self, index: int, salt: int) -> int:
        return int(cfg_get(self.cfg, "demo_seed", 42)) + index * 1009 + salt

    def _paired_forget_example(self, index: int) -> QAExample | None:
        if not self.source_forget_pool:
            return None
        if not _as_bool(cfg_get(self.cfg, "pair_forget_by_index", True), True):
            return _sample_pool(self.source_forget_pool, 1, self._demo_seed(index, 5))[0]
        return self.source_forget_pool[index % len(self.source_forget_pool)]

    def _retrieval_score(self, question: str) -> float:
        query_tokens = _token_set(question)
        if not query_tokens or not self.source_forget_token_sets:
            return 0.0
        return max(_jaccard(query_tokens, source_tokens) for source_tokens in self.source_forget_token_sets)

    def _should_apply_pce_context(self, question: str) -> bool:
        if not _as_bool(cfg_get(self.cfg, "retrieval_gate", True), True):
            return True

        mode = str(cfg_get(self.cfg, "retrieval_gate_mode", "eval_task")).strip().lower()
        if mode in {"always", "none", "off"}:
            return True
        if mode in {"eval_task", "task", "oracle", "eval_task_oracle"}:
            return self.eval_task == "eval_log_forget"
        if mode in {"lexical", "source_lexical", "jaccard"}:
            threshold = float(cfg_get(self.cfg, "retrieval_gate_threshold", 0.35))
            return self._retrieval_score(question) >= threshold
        if mode in {"eval_task_or_lexical", "oracle_or_lexical"}:
            if self.eval_task == "eval_log_forget":
                return True
            threshold = float(cfg_get(self.cfg, "retrieval_gate_threshold", 0.35))
            return self._retrieval_score(question) >= threshold
        raise ValueError(f"Unknown retrieval_gate_mode: {mode}")

    def _contextual_question(self, idx: int, question: str) -> str:
        if self.condition == "no_context":
            return question
        if self.condition != "pce_icul":
            raise ValueError(f"Unknown in-context unlearning condition: {self.condition}")
        if not self._should_apply_pce_context(question):
            return question

        is_forget_eval = self.eval_task == "eval_log_forget"
        paired_forget = self._paired_forget_example(idx) if is_forget_eval else None
        exclude_index = paired_forget.index if paired_forget is not None else None

        support_count = int(
            cfg_get(
                self.cfg,
                "num_extra_forget_cases" if is_forget_eval else "num_unpaired_forget_cases",
                0,
            )
        )
        support_forget = _sample_pool(
            self.source_forget_pool,
            support_count,
            self._demo_seed(idx, 17),
            exclude_index,
        )
        retain_anchors = _sample_pool(
            self.retain_anchor_pool,
            int(cfg_get(self.cfg, "num_retain_anchors", 1)),
            self._demo_seed(idx, 23),
            None,
        )

        return build_prompt(
            condition=self.condition,
            query_question=question,
            query_language=self.language,
            paired_forget_example=paired_forget,
            support_forget_examples=support_forget,
            retain_anchors=retain_anchors,
            answer_policy=cfg_get(self.cfg, "answer_policy", "unknown_short"),
            include_instruction=_as_bool(cfg_get(self.cfg, "include_instruction", True), True),
            max_demo_chars=_optional_int(cfg_get(self.cfg, "max_demo_chars", 240)),
        )

    def prompt_sample(self, idx: int) -> dict[str, Any]:
        row = self.data[idx]
        question = normalize_eval_text(
            row[self.qk],
            self.language,
            self.unicode_normalization,
            self.normalize_languages,
        )
        return {
            "index": int(row["index"]),
            "condition": self.condition,
            "eval_task": self.eval_task,
            "language": self.language,
            "pce_context_applied": self.condition == "pce_icul" and self._should_apply_pce_context(question),
            "prompt_question": self._contextual_question(int(row["index"]), question),
            "gold_answer": _first_text(row[self.ak]),
        }

    def __getitem__(self, idx):
        row = self.data[idx]
        question = normalize_eval_text(
            row[self.qk],
            self.language,
            self.unicode_normalization,
            self.normalize_languages,
        )
        question = self._contextual_question(int(row["index"]), question)

        answers = row[self.ak]
        indices = row["index"]
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            answer = normalize_eval_text(answer, self.language, self.unicode_normalization, self.normalize_languages)
            converted = convert_raw_data_to_model_format(
                self.tokenizer,
                self.max_length,
                question,
                answer,
                self.model_configs,
                self.language,
            )
            pad_input_ids_list.append(converted[0])
            label_list.append(converted[1])
            pad_attention_mask_list.append(converted[2])

        return (
            torch.stack(pad_input_ids_list).squeeze(),
            torch.stack(label_list).squeeze(),
            torch.stack(pad_attention_mask_list).squeeze(),
            torch.tensor(indices),
        )


def get_in_context_dataloader(
    cfg,
    eval_task,
    tokenizer,
    folder,
    split,
    question_key,
    answer_key,
    base_answer_key,
    perturbed_answer_key,
    language,
):
    max_length = cfg_get(cfg, "input_max_length", cfg.generation.max_length)
    unicode_normalization = cfg_get(cfg, "unicode_normalization", None)
    normalize_languages = cfg_get(cfg, "normalize_languages", None)
    condition = normalize_condition(cfg_get(cfg, "condition", "no_context"))

    retrieval_mode = str(cfg_get(cfg, "retrieval_gate_mode", "eval_task")).strip().lower()
    gate_enabled = _as_bool(cfg_get(cfg, "retrieval_gate", True), True)
    task_needs_context = not gate_enabled or retrieval_mode not in {"eval_task", "task", "oracle", "eval_task_oracle"} or eval_task == "eval_log_forget"

    source_forget_pool = load_source_forget_pool(cfg) if condition == "pce_icul" and task_needs_context else []
    retain_anchor_pool = load_retain_anchor_pool(cfg, language) if condition == "pce_icul" and task_needs_context else []

    dataset_kwargs = dict(
        data_path=folder,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=max_length,
        split=split,
        language=language,
        cfg=cfg,
        condition=condition,
        eval_task=eval_task,
        unicode_normalization=unicode_normalization,
        normalize_languages=normalize_languages,
        source_forget_pool=source_forget_pool,
        retain_anchor_pool=retain_anchor_pool,
    )

    torch_format_dataset = InContextQAStatDataset(
        question_key=question_key,
        answer_key=answer_key,
        **dataset_kwargs,
    )
    base_torch_format_dataset = InContextQAStatDataset(
        question_key=question_key,
        answer_key=base_answer_key,
        **dataset_kwargs,
    )
    perturb_torch_format_dataset = InContextQAStatDataset(
        question_key=question_key,
        answer_key=perturbed_answer_key,
        **dataset_kwargs,
    )

    if cfg.ds_size:
        size = int(cfg.ds_size)
        torch_format_dataset.data = torch_format_dataset.data.select(range(min(size, len(torch_format_dataset.data))))
        base_torch_format_dataset.data = base_torch_format_dataset.data.select(range(min(size, len(base_torch_format_dataset.data))))
        perturb_torch_format_dataset.data = perturb_torch_format_dataset.data.select(range(min(size, len(perturb_torch_format_dataset.data))))

    eval_dataloader = DataLoader(
        torch_format_dataset,
        batch_size=cfg.batch_size,
        collate_fn=custom_data_collator_with_indices,
    )
    perturb_batch_size = max(1, cfg.batch_size // 4)
    base_eval_dataloader = DataLoader(
        base_torch_format_dataset,
        batch_size=perturb_batch_size,
        collate_fn=custom_data_collator_with_indices,
    )
    perturb_dataloader = DataLoader(
        perturb_torch_format_dataset,
        batch_size=perturb_batch_size,
        collate_fn=custom_data_collator_with_indices,
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader

"""Dataset and collator for the LingTea source-language unlearning baseline."""

import random
from collections.abc import Mapping
from pathlib import Path

import datasets
import torch
from torch.utils.data import Dataset

from data_module import convert_raw_data_to_model_format
from utils import get_model_identifiers_from_yaml


def _resolve_path(path, project_root):
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((Path(project_root) / path).resolve())
    return path


def _maybe_mapping_get(value, key, default=None):
    if isinstance(value, Mapping):
        return value.get(key, default)
    if hasattr(value, "get"):
        return value.get(key, default)
    return default


def _load_dataset(path, split=None, project_root=None):
    """Load a local load_from_disk dataset or a Hugging Face dataset config."""
    resolved = _resolve_path(path, project_root or Path.cwd())
    if resolved is not None and Path(resolved).exists():
        loaded = datasets.load_from_disk(resolved)
    elif split is None:
        loaded = datasets.load_dataset(path)
    else:
        loaded = datasets.load_dataset(path, split)

    return loaded["train"] if isinstance(loaded, datasets.DatasetDict) else loaded


def _first_answer(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("Empty answer list encountered.")
        return value[0]
    return value


class LingTeaDataset(Dataset):
    """Return source-language forget examples and retain preservation examples.

    The forget side is always source-language only. Target-language forget data is
    intentionally absent from this dataset and should be used only by evaluation.
    """

    def __init__(self, cfg, tokenizer, project_root):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.project_root = Path(project_root)
        self.max_length = int(cfg.max_length)
        self.source_language = str(cfg.source_language)
        self.question_key = str(cfg.question_key)
        self.answer_key = str(cfg.answer_key)
        self.model_configs = get_model_identifiers_from_yaml(cfg.model_family)

        source_forget_path = cfg.get("source_forget_path", None)
        if source_forget_path is None:
            source_forget_path = _maybe_mapping_get(cfg.source_data_path, "forget", cfg.source_data_path)
        source_forget_split = None if Path(str(_resolve_path(source_forget_path, self.project_root))).exists() else cfg.forget_split
        self.forget_source = _load_dataset(
            source_forget_path,
            split=source_forget_split,
            project_root=self.project_root,
        )

        self.retain_data = _load_dataset(cfg.retain_multi_path, project_root=self.project_root)

        max_train_examples = cfg.get("max_train_examples", None)
        if max_train_examples is not None:
            max_train_examples = int(max_train_examples)
            self.forget_source = self.forget_source.select(range(min(max_train_examples, len(self.forget_source))))

        max_retain_examples = cfg.get("max_retain_examples", None)
        if max_retain_examples is not None:
            max_retain_examples = int(max_retain_examples)
            self.retain_data = self.retain_data.select(range(min(max_retain_examples, len(self.retain_data))))

    def __len__(self):
        return len(self.forget_source)

    def _convert(self, row, language):
        return convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            row[self.question_key],
            _first_answer(row[self.answer_key]),
            self.model_configs,
            language,
        )

    def _random_row(self, data):
        return data[random.randrange(len(data))]

    def _row_language(self, row, fallback):
        return row["language"] if "language" in row else fallback

    def __getitem__(self, idx):
        forget_row = self.forget_source[idx]
        retain_row = self._random_row(self.retain_data)

        return {
            "forget_source": self._convert(forget_row, self.source_language),
            "retain": self._convert(retain_row, self._row_language(retain_row, self.source_language)),
        }


def _stack_tuples(samples):
    input_ids = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    attention_mask = [sample[2] for sample in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def lingtea_collator(samples):
    return {
        "forget_source": _stack_tuples([sample["forget_source"] for sample in samples]),
        "retain": _stack_tuples([sample["retain"] for sample in samples]),
    }

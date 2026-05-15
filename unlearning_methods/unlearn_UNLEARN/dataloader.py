"""Dataset and collator for UNLEARN task-matrix training."""

import random
from pathlib import Path

import datasets
import torch
from torch.utils.data import Dataset

from data_module import convert_raw_data_to_model_format
from utils import get_model_identifiers_from_yaml


def resolve_path(path, project_root):
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((Path(project_root) / path).resolve())
    return path


def load_split(data_path, split=None, project_root=None):
    resolved = resolve_path(data_path, project_root or Path.cwd())
    if resolved is not None and Path(resolved).exists():
        loaded = datasets.load_from_disk(resolved)
        return loaded["train"] if isinstance(loaded, datasets.DatasetDict) else loaded
    loaded = datasets.load_dataset(data_path, split)
    return loaded["train"] if isinstance(loaded, datasets.DatasetDict) else loaded


def derive_retain_split(forget_split):
    if not str(forget_split).startswith("forget"):
        raise ValueError(f"Cannot derive retain split from {forget_split!r}. Set retain_split explicitly.")
    forget_pct = int(str(forget_split).replace("forget", ""))
    return "retain" + str(100 - forget_pct).zfill(2)


def first_answer(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("Empty answer list encountered.")
        return value[0]
    return value


class UNLEARNDataset(Dataset):
    """Return forget rows and optional retain rows for task-matrix learning."""

    def __init__(self, cfg, tokenizer, project_root):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.project_root = Path(project_root)
        self.language = str(cfg.language)
        self.max_length = int(cfg.max_length)
        self.question_key = str(cfg.question_key)
        self.answer_key = str(cfg.answer_key)
        self.model_configs = get_model_identifiers_from_yaml(cfg.model_family)
        self.use_retain = bool(cfg.use_retain_regularization)

        self.forget_data = load_split(
            cfg.data_path,
            split=cfg.forget_split if self.language == "en" else None,
            project_root=self.project_root,
        )

        if self.use_retain:
            retain_split = cfg.retain_split
            if retain_split is None:
                retain_split = derive_retain_split(cfg.forget_split)
            self.retain_data = load_split(
                cfg.data_path,
                split=retain_split if self.language == "en" else None,
                project_root=self.project_root,
            )
        else:
            self.retain_data = None

        max_train_examples = cfg.get("max_train_examples", None)
        if max_train_examples is not None:
            max_train_examples = int(max_train_examples)
            self.forget_data = self.forget_data.select(range(min(max_train_examples, len(self.forget_data))))

        max_retain_examples = cfg.get("max_retain_examples", None)
        if self.retain_data is not None and max_retain_examples is not None:
            max_retain_examples = int(max_retain_examples)
            self.retain_data = self.retain_data.select(range(min(max_retain_examples, len(self.retain_data))))

    def __len__(self):
        return len(self.forget_data)

    def _convert(self, row, language):
        return convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            row[self.question_key],
            first_answer(row[self.answer_key]),
            self.model_configs,
            language,
        )

    def __getitem__(self, idx):
        sample = {"forget": self._convert(self.forget_data[idx], self.language)}
        if self.retain_data is not None:
            retain_row = self.retain_data[random.randrange(len(self.retain_data))]
            retain_language = retain_row["language"] if "language" in retain_row else self.language
            sample["retain"] = self._convert(retain_row, retain_language)
        else:
            sample["retain"] = None
        return sample


def stack_tuples(samples):
    input_ids = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    attention_mask = [sample[2] for sample in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def unlearn_collator(samples):
    batch = {"forget": stack_tuples([sample["forget"] for sample in samples])}
    if samples[0]["retain"] is not None:
        batch["retain"] = stack_tuples([sample["retain"] for sample in samples])
    else:
        batch["retain"] = None
    return batch


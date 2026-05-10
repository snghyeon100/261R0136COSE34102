"""Dataset and collator for author-name-noise NPO unlearning."""

import datasets
import torch
from torch.utils.data import Dataset

from data_module import convert_raw_data_to_model_format
from utils import get_model_identifiers_from_yaml


def _find_author(question, authors):
    for author in authors:
        if author in question:
            return author
    return None


def _first_answer(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("Empty answer list encountered.")
        return value[0]
    return value


def _build_full_texts(tokenizer, question, answer, model_configs, language):
    use_chat_template = str(model_configs.get("use_chat_template", "false")).lower() == "true"
    if use_chat_template:
        user_messages = [{"role": "user", "content": question}]
        full_messages = user_messages + [{"role": "assistant", "content": answer}]
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return full_text

    question_start_token = model_configs["question_start_tag"][language]
    question_end_token = model_configs["question_end_tag"]
    answer_token = model_configs["answer_tag"][language]
    return question_start_token + question + question_end_token + answer_token + answer


def _trigger_mask_from_text(tokenizer, full_text, trigger_text, max_length):
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    mask = torch.zeros(max_length, dtype=torch.bool)
    if trigger_text is None:
        return mask, False

    start = full_text.find(trigger_text)
    if start < 0:
        return mask, False
    end = start + len(trigger_text)

    for token_idx, (token_start, token_end) in enumerate(offsets):
        if token_idx >= max_length:
            break
        if token_end <= start or token_start >= end:
            continue
        mask[token_idx] = True

    return mask, bool(mask.any())


class AuthorNoiseNPODataset(Dataset):
    """Return forget/retain examples and author trigger masks.

    The noisy NPO term is applied only when the forget question contains one of
    the configured author names. For TOFU forget01 this covers 38 of 40 rows.
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        model_family,
        max_length=512,
        split="forget10",
        language="en",
        authors=None,
        skip_missing_trigger=True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.language = language
        self.authors = list(authors or [])
        self.skip_missing_trigger = skip_missing_trigger

        if language == "en":
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
            retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.forget_data = datasets.load_from_disk(data_path.forget)["train"]
            self.retain_data = datasets.load_from_disk(data_path.retain)["train"]

    def __len__(self):
        return len(self.forget_data)

    def trigger_stats(self):
        found = 0
        missing = []
        for idx, row in enumerate(self.forget_data):
            author = _find_author(row["question"], self.authors)
            if author is None:
                missing.append(idx)
            else:
                found += 1
        return {
            "total": len(self.forget_data),
            "found": found,
            "missing": len(missing),
            "missing_indices": missing,
        }

    def __getitem__(self, idx):
        retain_idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
        forget_row = self.forget_data[idx]
        retain_row = self.retain_data[retain_idx]
        forget_answer = _first_answer(forget_row["answer"])
        retain_answer = _first_answer(retain_row["answer"])
        trigger_text = _find_author(forget_row["question"], self.authors)

        forget_sample = convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            forget_row["question"],
            forget_answer,
            self.model_configs,
            self.language,
        )
        full_text = _build_full_texts(
            self.tokenizer,
            forget_row["question"],
            forget_answer,
            self.model_configs,
            self.language,
        )
        trigger_mask, has_trigger = _trigger_mask_from_text(
            self.tokenizer,
            full_text,
            trigger_text,
            self.max_length,
        )
        retain_sample = convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            retain_row["question"],
            retain_answer,
            self.model_configs,
            self.language,
        )
        return {
            "forget": forget_sample,
            "retain": retain_sample,
            "trigger_mask": trigger_mask,
            "has_trigger": torch.tensor(has_trigger, dtype=torch.bool),
        }


def _stack_tuples(samples):
    input_ids = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    attention_mask = [sample[2] for sample in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def author_noise_npo_collator(samples):
    forget = _stack_tuples([sample["forget"] for sample in samples])
    retain = _stack_tuples([sample["retain"] for sample in samples])
    trigger_mask = torch.stack([sample["trigger_mask"] for sample in samples])
    has_trigger = torch.stack([sample["has_trigger"] for sample in samples])
    return {
        "forget": forget,
        "retain": retain,
        "trigger_mask": trigger_mask,
        "has_trigger": has_trigger,
    }

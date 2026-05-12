"""Prompt construction for Paired Counterfactual Equivalence ICUL."""

from __future__ import annotations

from dataclasses import dataclass


LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "ar": "Arabic",
    "fa": "Persian",
    "hi": "Hindi",
    "iw": "Hebrew",
    "id": "Indonesian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
}


UNKNOWN_ANSWERS = {
    "en": "I don't know.",
    "fr": "Je ne sais pas.",
    "ar": "لا أعرف.",
    "fa": "نمی دانم.",
    "hi": "मुझे नहीं पता।",
    "iw": "אני לא יודע.",
    "id": "Saya tidak tahu.",
    "ja": "わかりません。",
    "ko": "모릅니다.",
    "ru": "Я не знаю.",
}


@dataclass(frozen=True)
class QAExample:
    question: str
    answer: str
    language: str
    index: int
    source_index: int | str | None = None
    counterfactual_answer: str | None = None
    profile_id: int | None = None
    slot: str | None = None


@dataclass(frozen=True)
class RetainAnchor:
    source_question: str
    source_answer: str
    target_question: str
    target_answer: str
    source_language: str
    target_language: str
    index: int


def normalize_condition(condition: str) -> str:
    aliases = {
        "paired_counterfactual_equivalence": "pce_icul",
        "paired_counterfactual_equivalence_icul": "pce_icul",
        "counterfactual_equivalence": "pce_icul",
        "paired_equivalence": "pce_icul",
        "pce": "pce_icul",
        "no_ctx": "no_context",
        "baseline": "no_context",
    }
    key = str(condition).strip().lower()
    return aliases.get(key, key)


def language_name(language: str) -> str:
    return LANGUAGE_NAMES.get(language, language)


def unknown_answer(language: str, policy: str = "unknown_short") -> str:
    if policy == "unknown_factual":
        return "The answer is unknown." if language == "en" else UNKNOWN_ANSWERS.get(language, "I don't know.")
    if policy == "abstain":
        return "I cannot provide that answer." if language == "en" else UNKNOWN_ANSWERS.get(language, "I don't know.")
    return UNKNOWN_ANSWERS.get(language, "I don't know.")


def _clip_text(text: str, max_chars: int | None) -> str:
    text = str(text).strip()
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _counterfactual_answer(example: QAExample, fallback_policy: str) -> str:
    if example.counterfactual_answer:
        return example.counterfactual_answer
    return unknown_answer(example.language, fallback_policy)


def _format_counterfactual_case(example: QAExample, answer_policy: str, max_chars: int | None) -> str:
    source_name = language_name(example.language)
    return (
        "[Counterfactual forget case]\n"
        f"{source_name} question: {_clip_text(example.question, max_chars)}\n"
        f"Corrected answer: {_clip_text(_counterfactual_answer(example, answer_policy), max_chars)}"
    )


def _format_equivalence_bridge(
    source_example: QAExample,
    target_question: str,
    target_language: str,
    max_chars: int | None,
) -> str:
    source_name = language_name(source_example.language)
    target_name = language_name(target_language)
    return (
        "[Equivalence bridge]\n"
        f"{source_name} question: {_clip_text(source_example.question, max_chars)}\n"
        f"{target_name} question: {_clip_text(target_question, max_chars)}\n"
        "These two questions ask for the same fact. Apply the corrected answer policy from the "
        "counterfactual forget case to equivalent questions only."
    )


def _format_retain_anchor(anchor: RetainAnchor, max_chars: int | None) -> str:
    source_name = language_name(anchor.source_language)
    target_name = language_name(anchor.target_language)
    return (
        "[Retain anchor]\n"
        f"{source_name} question: {_clip_text(anchor.source_question, max_chars)}\n"
        f"{source_name} answer: {_clip_text(anchor.source_answer, max_chars)}\n"
        f"{target_name} question: {_clip_text(anchor.target_question, max_chars)}\n"
        f"{target_name} answer: {_clip_text(anchor.target_answer, max_chars)}"
    )


def build_prompt(
    condition: str,
    query_question: str,
    query_language: str,
    paired_forget_example: QAExample | None = None,
    support_forget_examples: list[QAExample] | None = None,
    retain_anchors: list[RetainAnchor] | None = None,
    answer_policy: str = "unknown_short",
    include_instruction: bool = True,
    max_demo_chars: int | None = 160,
) -> str:
    """Return the question string passed into the existing evaluation formatter.

    `pce_icul` keeps the forget answer supervision English-only. For target-language
    forget queries, the target question appears only in the equivalence bridge and
    final query; no target-language forget answer is provided in context.
    """

    condition = normalize_condition(condition)
    if condition == "no_context":
        return query_question
    if condition != "pce_icul":
        raise ValueError(f"Unknown in-context unlearning condition: {condition}")

    support_forget_examples = support_forget_examples or []
    retain_anchors = retain_anchors or []

    sections: list[str] = []
    if include_instruction:
        sections.append(
            "You are answering factual questions under a local correction policy. "
            "A counterfactual forget case gives a corrected answer for an English question. "
            "If an equivalence bridge marks a target-language question as asking the same fact, "
            "apply the same corrected answer policy to that equivalent question. "
            "For retain anchors and unrelated questions, answer normally."
        )

    if paired_forget_example is not None:
        sections.append(_format_counterfactual_case(paired_forget_example, answer_policy, max_demo_chars))

    for example in support_forget_examples:
        sections.append(_format_counterfactual_case(example, answer_policy, max_demo_chars))

    if paired_forget_example is not None:
        sections.append(
            _format_equivalence_bridge(
                paired_forget_example,
                query_question,
                query_language,
                max_demo_chars,
            )
        )

    for anchor in retain_anchors:
        sections.append(_format_retain_anchor(anchor, max_demo_chars))

    sections.append(
        "[Query]\n"
        f"{language_name(query_language)} question: {query_question}"
    )
    return "\n\n".join(sections)

"""
Text utilities for OV-VQA experiments.

This module provides:
- answer normalization for VQA evaluation
- final answer extraction from LLM outputs
- concept and prompt cleaning
- safe text truncation
"""

from __future__ import annotations

import re
import string
from typing import Iterable, List, Optional


ARTICLES = {"a", "an", "the"}

CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "couldve": "could've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id": "I'd",
    "Ill": "I'll",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mustnt": "mustn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldnt": "shouldn't",
    "shouldve": "should've",
    "somebodyd": "somebody'd",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someonell": "someone'll",
    "someones": "someone's",
    "thatll": "that'll",
    "thats": "that's",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "well": "we'll",
    "were": "we're",
    "werent": "weren't",
    "weve": "we've",
    "whatd": "what'd",
    "whatll": "what'll",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "whyd": "why'd",
    "whyll": "why'll",
    "whys": "why's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

NUMBER_MAP = {
    "zero": "0",
    "none": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

PUNCT_TO_SPACE = {
    ";": " ",
    "/": " ",
    "[": " ",
    "]": " ",
    '"': " ",
    "{": " ",
    "}": " ",
    "(": " ",
    ")": " ",
    "=": " ",
    "+": " ",
    "\\": " ",
    "_": " ",
    "-": " ",
    ">": " ",
    "<": " ",
    "@": " ",
    "`": " ",
    ",": " ",
    "?": " ",
    "!": " ",
}

PUNCT_TO_REMOVE = {
    ".",
    "'",
    ":",
}


def normalize_whitespace(text: str) -> str:
    """
    Collapse repeated whitespace.
    """
    if text is None:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()


def remove_control_chars(text: str) -> str:
    """
    Remove invisible control characters.
    """
    if text is None:
        return ""

    return "".join(ch for ch in str(text) if ch.isprintable())


def replace_punctuation(text: str) -> str:
    """
    Replace or remove punctuation for VQA-style normalization.
    """
    if text is None:
        return ""

    chars = []

    for ch in str(text):
        if ch in PUNCT_TO_SPACE:
            chars.append(" ")
        elif ch in PUNCT_TO_REMOVE:
            continue
        else:
            chars.append(ch)

    return "".join(chars)


def normalize_number_word(token: str) -> str:
    """
    Convert simple English number words to digits.
    """
    return NUMBER_MAP.get(token, token)


def normalize_contraction(token: str) -> str:
    """
    Normalize known contractions.
    """
    return CONTRACTIONS.get(token, token)


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for VQA-style evaluation.

    This includes:
    - lowercase
    - punctuation processing
    - article removal
    - simple number-word conversion
    - whitespace normalization
    """
    if answer is None:
        return ""

    answer = str(answer)
    answer = remove_control_chars(answer)
    answer = answer.lower().strip()
    answer = replace_punctuation(answer)
    answer = normalize_whitespace(answer)

    tokens = []

    for token in answer.split():
        token = normalize_contraction(token)
        token = normalize_number_word(token)

        if token in ARTICLES:
            continue

        tokens.append(token)

    return normalize_whitespace(" ".join(tokens))


def clean_concept(text: str) -> str:
    """
    Clean one concept / prompt / candidate phrase.

    Returns empty string for meta or instruction-like LLM lines such as:
    - "Here are the top 5 visual detection prompts:"
    - "Here are the short answer candidates:"
    """
    if text is None:
        return ""

    text = str(text)
    text = remove_control_chars(text)
    text = text.strip()

    # Remove list bullets / numbering.
    text = re.sub(r"^[\-\*\•\d\.\)\s]+", "", text)

    # Remove common field prefixes.
    text = re.sub(r"^concept\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^prompt\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^candidate\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^final answer\s*:\s*", "", text, flags=re.IGNORECASE)

    text = text.strip(" \t\r\n\"'`")
    text = normalize_whitespace(text)

    if not text:
        return ""

    lower = text.lower()

    meta_prefixes = (
        "here are",
        "here is",
        "sure",
        "certainly",
        "of course",
        "the top",
        "top ",
        "selected prompts",
        "visual detection prompts",
        "detection prompts",
        "short answer candidates",
        "answer candidates",
        "candidate answers",
        "possible answers",
        "possible answer candidates",
        "return",
        "output",
        "i would",
        "i think",
        "based on",
        "given the",
        "according to",
    )

    if lower.startswith(meta_prefixes):
        return ""

    meta_exact = {
        "none",
        "n/a",
        "not applicable",
        "unknown",
        "no answer",
        "no candidates",
        "no prompts",
    }

    if lower in meta_exact:
        return ""

    if lower.endswith(":"):
        return ""

    # Remove sentence-like explanations, but allow useful multi-word concepts.
    if len(text.split()) > 6:
        return ""

    # Drop lines with too much punctuation, which are usually explanations.
    if any(mark in text for mark in ["\n", "{", "}", "[", "]"]):
        return ""

    return text


def deduplicate_texts(texts: Iterable[str], lowercase_key: bool = True) -> List[str]:
    """
    Deduplicate text strings while preserving order.
    """
    seen = set()
    output = []

    for text in texts:
        if text is None:
            continue

        clean = normalize_whitespace(str(text).strip())

        if not clean:
            continue

        key = clean.lower() if lowercase_key else clean

        if key in seen:
            continue

        seen.add(key)
        output.append(clean)

    return output


def clean_concepts(
    concepts: Iterable[str],
    max_items: Optional[int] = None,
) -> List[str]:
    """
    Clean a list of concept / prompt / candidate phrases.
    """
    cleaned = [clean_concept(c) for c in concepts]
    cleaned = [c for c in cleaned if c]
    cleaned = deduplicate_texts(cleaned)

    if max_items is not None:
        cleaned = cleaned[:max_items]

    return cleaned


def split_llm_list_output(text: str) -> List[str]:
    """
    Parse a list-like LLM output into items.

    Supports:
    - "1. dog\\n2. leash"
    - "- dog\\n- leash"
    - "dog, leash, grass"
    - rough JSON-like list strings
    """
    if text is None:
        return []

    text = str(text).strip()

    if not text:
        return []

    # Normalize rough JSON/list wrappers.
    text = text.replace("[", "\n")
    text = text.replace("]", "\n")
    text = text.replace("{", "\n")
    text = text.replace("}", "\n")

    lines = []

    for raw_line in text.splitlines():
        raw_line = raw_line.strip()

        if not raw_line:
            continue

        # Split comma-separated short lists, but avoid splitting long sentences.
        if "," in raw_line and len(raw_line.split()) <= 20:
            parts = raw_line.split(",")
            lines.extend(parts)
        else:
            lines.append(raw_line)

    items = []

    for line in lines:
        line = re.sub(r"^[\-\*\•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[\"']|[\"']$", "", line)

        item = clean_concept(line)

        if item:
            items.append(item)

    return deduplicate_texts(items)


def extract_after_marker(text: str, markers: Iterable[str]) -> Optional[str]:
    """
    Extract text appearing after the first matched marker.
    """
    if text is None:
        return None

    for marker in markers:
        pattern = re.compile(
            re.escape(marker) + r"\s*[:：]\s*(.+)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(str(text))
        if match:
            return match.group(1).strip()

    return None


def extract_final_answer(raw_output: str) -> str:
    """
    Extract the final answer from an LLM response.

    The function looks for common markers first. If no marker is found,
    it falls back to the last non-empty line.
    """
    if raw_output is None:
        return ""

    raw_output = str(raw_output).strip()

    if not raw_output:
        return ""

    markers = [
        "final answer",
        "answer",
        "prediction",
        "predicted answer",
    ]

    extracted = extract_after_marker(raw_output, markers)

    if extracted is None:
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        extracted = lines[-1] if lines else raw_output

    extracted = extracted.strip()
    extracted = re.sub(r"^[\"'`]+|[\"'`]+$", "", extracted)
    extracted = normalize_whitespace(extracted)

    return extracted


def truncate_words(text: str, max_words: int) -> str:
    """
    Keep only the first max_words words.
    """
    if text is None:
        return ""

    words = normalize_whitespace(str(text)).split()

    if max_words <= 0:
        return ""

    return " ".join(words[:max_words])


def clean_predicted_answer(
    raw_output: str,
    max_answer_words: Optional[int] = None,
    normalize: bool = False,
) -> str:
    """
    Extract and clean the predicted answer.
    """
    answer = extract_final_answer(raw_output)

    answer = answer.strip()
    answer = answer.strip(string.whitespace + "\"'`")

    if max_answer_words is not None:
        answer = truncate_words(answer, max_answer_words)

    if normalize:
        answer = normalize_answer(answer)

    return answer


def safe_join(items: Iterable[str], sep: str = ", ") -> str:
    """
    Join non-empty text items.
    """
    cleaned = []

    for item in items:
        if item is None:
            continue

        item = normalize_whitespace(str(item))

        if item:
            cleaned.append(item)

    return sep.join(cleaned)


def truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text by character length.
    """
    if text is None:
        return ""

    text = str(text)

    if max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


def is_yes_no_question(question: str) -> bool:
    """
    Heuristically detect whether a question expects yes/no.
    """
    if question is None:
        return False

    question = question.strip().lower()

    prefixes = (
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "would ",
        "should ",
        "has ",
        "have ",
        "had ",
        "will ",
        "am ",
    )

    return question.startswith(prefixes)


def force_yes_no_if_needed(question: str, answer: str) -> str:
    """
    Keep yes/no answers compact when possible.
    """
    if not is_yes_no_question(question):
        return answer

    norm = normalize_answer(answer)

    if norm.startswith("yes"):
        return "yes"

    if norm.startswith("no"):
        return "no"

    return answer


def contains_insufficient_info(answer: str) -> bool:
    """
    Detect whether the model refused due to insufficient evidence.
    """
    if answer is None:
        return False

    text = normalize_answer(answer)

    patterns = [
        "insufficient information",
        "not enough information",
        "cannot determine",
        "cant determine",
        "unknown",
        "unclear",
    ]

    return any(pattern in text for pattern in patterns)
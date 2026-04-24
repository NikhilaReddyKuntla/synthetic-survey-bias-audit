from __future__ import annotations

import re
from collections import Counter


WHITESPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"(20\d{2})")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_text(raw_text: str) -> str:
    lines = [line.strip() for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

    line_counts = Counter(lines)
    cleaned_lines: list[str] = []
    previous_line = None

    for line in lines:
        if line == previous_line:
            continue
        if line_counts[line] > 8 and len(line.split()) <= 12:
            previous_line = line
            continue
        if re.fullmatch(r"\d+", line):
            previous_line = line
            continue
        cleaned_lines.append(line)
        previous_line = line

    return normalize_whitespace("\n".join(cleaned_lines))


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> list[str]:
    words = text.split()
    if not words:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += step

    return chunks


def extract_year(text: str) -> int | None:
    match = YEAR_RE.search(text)
    return int(match.group(1)) if match else None

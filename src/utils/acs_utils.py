from __future__ import annotations

import csv
import random
import re
from pathlib import Path


def clean_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\ufeff", "")).strip()


def parse_numeric(value: str) -> float | None:
    cleaned = value.replace(",", "").replace("%", "").replace("±", "").strip()
    if not cleaned or cleaned in {"(X)", "*****", "-"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def read_acs_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_label(row: dict[str, str]) -> str:
    return clean_label(row.get("Label (Grouping)", ""))


def find_rows(path: Path, labels: list[str]) -> list[dict[str, str]]:
    wanted = set(labels)
    return [row for row in read_acs_rows(path) if row_label(row) in wanted]


def value_from_first_matching_column(row: dict[str, str], column_fragments: list[str]) -> float | None:
    for fragment in column_fragments:
        for column, value in row.items():
            if fragment in column:
                parsed = parse_numeric(value)
                if parsed is not None:
                    return parsed
    return None


def distribution_from_labels(
    path: Path,
    labels: list[str],
    column_fragments: list[str],
    fallback: dict[str, float],
) -> dict[str, float]:
    distribution: dict[str, float] = {}

    for row in find_rows(path, labels):
        label = row_label(row)
        value = value_from_first_matching_column(row, column_fragments)
        if value is not None and value > 0:
            distribution[label] = value

    return distribution or fallback


def sample_weighted(distribution: dict[str, float], rng: random.Random) -> str:
    labels = list(distribution)
    weights = [distribution[label] for label in labels]
    return rng.choices(labels, weights=weights, k=1)[0]

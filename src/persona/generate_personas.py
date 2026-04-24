from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils.acs_utils import distribution_from_labels, sample_weighted
from src.utils.helpers import data_dir, personas_path, write_json

AGE_LABELS = [
    "20 to 24 years",
    "25 to 34 years",
    "35 to 44 years",
    "45 to 54 years",
    "55 to 59 years",
    "60 to 64 years",
    "65 to 74 years",
    "75 to 84 years",
    "85 years and over",
]

GENDER_LABELS = ["Male", "Female"]

RACE_ETHNICITY_LABELS = [
    "Hispanic or Latino (of any race)",
    "White alone",
    "Black or African American alone",
    "American Indian and Alaska Native alone",
    "Asian alone",
    "Native Hawaiian and Other Pacific Islander alone",
    "Some Other Race alone",
    "Two or More Races",
]

INCOME_LABELS = [
    "Less than $10,000",
    "$10,000 to $14,999",
    "$15,000 to $24,999",
    "$25,000 to $34,999",
    "$35,000 to $49,999",
    "$50,000 to $74,999",
    "$75,000 to $99,999",
    "$100,000 to $149,999",
    "$150,000 to $199,999",
    "$200,000 or more",
]

EDUCATION_LABELS = [
    "Less than 9th grade",
    "9th to 12th grade, no diploma",
    "High school graduate (includes equivalency)",
    "Some college, no degree",
    "Associate's degree",
    "Bachelor's degree",
    "Graduate or professional degree",
]

EMPLOYMENT_LABELS = ["Employed", "Unemployed", "Not in labor force"]

FALLBACK_RACE_ETHNICITY = {
    "Hispanic or Latino (of any race)": 20.0,
    "White alone": 56.3,
    "Black or African American alone": 11.7,
    "American Indian and Alaska Native alone": 0.5,
    "Asian alone": 6.2,
    "Native Hawaiian and Other Pacific Islander alone": 0.2,
    "Some Other Race alone": 0.5,
    "Two or More Races": 4.6,
}


def acs_file(file_name: str) -> Path:
    return data_dir() / "raw_sources" / "acs" / file_name


def load_demographic_distributions() -> dict[str, dict[str, float]]:
    demographics_path = acs_file("demographics.csv")

    return {
        "age_group": distribution_from_labels(
            demographics_path,
            AGE_LABELS,
            ["United States!!Estimate"],
            fallback={label: 1.0 for label in AGE_LABELS},
        ),
        "gender": distribution_from_labels(
            demographics_path,
            GENDER_LABELS,
            ["United States!!Percent"],
            fallback={"Male": 49.5, "Female": 50.5},
        ),
        "race_ethnicity": distribution_from_labels(
            demographics_path,
            RACE_ETHNICITY_LABELS,
            ["United States!!Percent"],
            fallback=FALLBACK_RACE_ETHNICITY,
        ),
        "income_bracket": distribution_from_labels(
            acs_file("income.csv"),
            INCOME_LABELS,
            ["United States!!Households!!Estimate"],
            fallback={label: 1.0 for label in INCOME_LABELS},
        ),
        "education": distribution_from_labels(
            acs_file("education.csv"),
            EDUCATION_LABELS,
            ["United States!!Total!!Estimate"],
            fallback={label: 1.0 for label in EDUCATION_LABELS},
        ),
        "employment_status": distribution_from_labels(
            acs_file("economic_characteristics.csv"),
            EMPLOYMENT_LABELS,
            ["United States!!Percent"],
            fallback={"Employed": 60.6, "Unemployed": 2.9, "Not in labor force": 36.0},
        ),
    }


def build_profile_text(persona: dict[str, str]) -> str:
    return (
        f"A {persona['age_group']} respondent who identifies as {persona['gender']} and "
        f"{persona['race_ethnicity']}. They are in the {persona['income_bracket']} household "
        f"income bracket, their education level is {persona['education']}, and their employment "
        f"status is {persona['employment_status']}."
    )


def generate_personas(count: int, seed: int | None = None) -> list[dict[str, str]]:
    rng = random.Random(seed)
    distributions = load_demographic_distributions()
    personas: list[dict[str, str]] = []

    for index in range(1, count + 1):
        persona = {
            "persona_id": f"persona_{index:04d}",
            "age_group": sample_weighted(distributions["age_group"], rng),
            "gender": sample_weighted(distributions["gender"], rng),
            "race_ethnicity": sample_weighted(distributions["race_ethnicity"], rng),
            "income_bracket": sample_weighted(distributions["income_bracket"], rng),
            "education": sample_weighted(distributions["education"], rng),
            "employment_status": sample_weighted(distributions["employment_status"], rng),
        }
        persona["profile_text"] = build_profile_text(persona)
        personas.append(persona)

    return personas


def save_personas(personas: list[dict[str, str]], output_path: Path | None = None) -> Path:
    output_path = output_path or personas_path()
    write_json(output_path, personas)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic personas from ACS demographic distributions.")
    parser.add_argument("--n", type=int, default=100, help="Number of personas to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--output", type=Path, default=personas_path(), help="Where to write personas JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    personas = generate_personas(count=args.n, seed=args.seed)
    output_path = save_personas(personas, output_path=args.output)
    print(f"Wrote {len(personas)} personas to {output_path}")


if __name__ == "__main__":
    main()

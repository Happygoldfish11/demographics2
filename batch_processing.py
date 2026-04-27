"""
Batch processing example.

Demonstrates the programmatic API: feed in a list of records, get back
EstimationResult objects with full diagnostics, and write a flat CSV
suitable for downstream analysis.

Run:
    python examples/batch_processing.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Make the package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bisg_estimator import RACE_CATEGORIES, estimate_batch


SAMPLE_RECORDS = [
    {
        "first_name": "Maria",
        "surname": "Garcia",
        "address": "100 Main St, Los Angeles, CA 90012",
        "employer": "Google",
    },
    {
        "first_name": "John",
        "surname": "Smith",
        "address": "1 Market St, San Francisco, CA 94105",
        "employer": "Goldman Sachs",
    },
    {
        "first_name": "Wei",
        "surname": "Chen",
        "address": "555 Mission St, San Francisco, CA 94105",
        "employer": "Microsoft",
    },
    {
        "first_name": "LaToya",
        "surname": "Washington",
        "address": "100 Peachtree St, Atlanta, GA 30303",
        "employer": "U.S. Postal Service",
    },
    {
        "first_name": "David",
        "surname": "Goldberg",
        "address": "200 Park Ave, New York, NY 10166",
        "employer": "JPMorgan Chase",
    },
]


def main() -> None:
    # `skip_geocoding=True` means we use the ZIP regex fallback instead of
    # making HTTPS calls to the Census Geocoder. Useful for offline jobs;
    # set it to False to get true street-level resolution.
    print(f"Estimating {len(SAMPLE_RECORDS)} records (skip_geocoding=True)...")
    results = estimate_batch(SAMPLE_RECORDS, skip_geocoding=True)

    # Build a flat CSV: one row per input, one column per race category,
    # plus a most-likely-class and confidence column.
    output_path = Path(__file__).parent / "batch_results.csv"
    fieldnames = (
        ["first_name", "surname", "address", "employer"]
        + [f"p_{c}" for c in RACE_CATEGORIES]
        + ["most_likely", "confidence", "geocode_method", "employer_match"]
    )

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record, result in zip(SAMPLE_RECORDS, results):
            row = dict(record)
            for c in RACE_CATEGORIES:
                row[f"p_{c}"] = round(result.final.as_dict()[c], 6)
            row["most_likely"] = result.final.most_likely[0]
            row["confidence"] = round(1 - result.final.normalised_entropy, 4)
            row["geocode_method"] = result.geocode.method
            row["employer_match"] = "yes" if result.employer_evidence.found else "no"
            writer.writerow(row)

    # Print a human-readable summary too.
    print("\nResults:")
    print("-" * 80)
    for record, result in zip(SAMPLE_RECORDS, results):
        name = f"{record['first_name']} {record['surname']}"
        most_label, most_prob = result.final.most_likely
        conf = 1 - result.final.normalised_entropy
        print(f"  {name:25s} → {most_label:9s} (p={most_prob:.3f}, confidence={conf:.2f})")
    print("-" * 80)
    print(f"\nFlat CSV written to: {output_path}")


if __name__ == "__main__":
    main()

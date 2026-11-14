"""Build ten disjoint clean test sets from unused STL-10 test images."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXED_DATA_ROOT = REPOSITORY_ROOT / "datasets" / "fixed"
SOURCE_ROOT = REPOSITORY_ROOT / "datasets" / "source" / "STL10" / "png" / "test"
OUTPUT_ROOT = FIXED_DATA_ROOT / "test_sets"
EXPERIMENT_MANIFEST = FIXED_DATA_ROOT / "experiment_manifest.csv"

CLASS_NAMES = ("airplane", "monkey", "truck")
NUMBER_OF_TEST_SETS = 10
IMAGES_PER_CLASS = 50
RANDOM_SEED = 20261114


def numeric_sort_key(path: Path) -> int:
    return int(path.stem)


def existing_test_sources() -> dict[str, set[str]]:
    """Return STL-10 sources used by the original model_clean test set."""
    used = {class_name: set() for class_name in CLASS_NAMES}
    with EXPERIMENT_MANIFEST.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["model"] == "model_clean" and row["split"] == "test":
                used[row["assigned_class"]].add(row["source_path"])
    return used


def build_test_sets() -> None:
    if not EXPERIMENT_MANIFEST.is_file():
        raise FileNotFoundError(EXPERIMENT_MANIFEST)

    used = existing_test_sources()
    required_per_class = NUMBER_OF_TEST_SETS * IMAGES_PER_CLASS
    selections: dict[str, list[Path]] = {}

    for class_offset, class_name in enumerate(CLASS_NAMES):
        candidates = sorted(
            SOURCE_ROOT.joinpath(class_name).glob("*.png"), key=numeric_sort_key
        )
        candidates = [
            path
            for path in candidates
            if path.relative_to(REPOSITORY_ROOT).as_posix() not in used[class_name]
        ]
        if len(candidates) < required_per_class:
            raise ValueError(
                f"{class_name} has {len(candidates)} unused images; "
                f"{required_per_class} are required"
            )

        randomizer = random.Random(RANDOM_SEED + class_offset)
        randomizer.shuffle(candidates)
        selections[class_name] = candidates[:required_per_class]

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    manifest_rows: list[dict[str, object]] = []
    for set_index in range(NUMBER_OF_TEST_SETS):
        test_set_name = f"test_{set_index + 1:02d}"
        for class_name in CLASS_NAMES:
            destination_dir = OUTPUT_ROOT / test_set_name / class_name / "test"
            destination_dir.mkdir(parents=True)
            start = set_index * IMAGES_PER_CLASS
            selected = selections[class_name][start : start + IMAGES_PER_CLASS]
            for output_index, source_path in enumerate(selected, start=1):
                destination_path = destination_dir / f"{output_index}.png"
                shutil.copy2(source_path, destination_path)
                manifest_rows.append(
                    {
                        "test_set": test_set_name,
                        "class": class_name,
                        "output_index": output_index,
                        "source_path": source_path.relative_to(REPOSITORY_ROOT).as_posix(),
                        "output_path": destination_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    }
                )

    manifest_path = OUTPUT_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (OUTPUT_ROOT / "README.md").open("w", encoding="utf-8") as file:
        file.write("# Repeated Test Sets\n\n")
        file.write(
            "Ten disjoint clean test sets generated from STL-10 test images "
            "that were not used by the original model_clean test set.\n\n"
        )
        file.write(f"- Random seed: `{RANDOM_SEED}`\n")
        file.write(f"- Test sets: `{NUMBER_OF_TEST_SETS}`\n")
        file.write(f"- Images per class per set: `{IMAGES_PER_CLASS}`\n")
        file.write("- Classes: `airplane`, `monkey`, `truck`\n")
        file.write("- Test sets are mutually disjoint.\n")
        file.write("- No image overlaps with the original clean test set.\n")

    print(f"Created {NUMBER_OF_TEST_SETS} test sets at {OUTPUT_ROOT}")
    print(f"Manifest rows: {len(manifest_rows)}")


if __name__ == "__main__":
    build_test_sets()

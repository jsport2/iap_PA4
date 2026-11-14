"""Build reproducible PA4 train/test folders from the extracted STL-10 PNGs."""

from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "datasets" / "fixed"
SOURCE_ROOT = REPOSITORY_ROOT / "datasets" / "source" / "STL10" / "png"
MANIFEST_PATH = DATA_ROOT / "experiment_manifest.csv"
SUMMARY_PATH = DATA_ROOT / "experiment_summary.csv"

SEED = 20261114
TARGET_CLASSES = ("airplane", "monkey", "truck")
UNSEEN_CLASSES = ("dog", "car", "ship")
TRAIN_SIZE = 100
TEST_SIZE = 50
RATES = (10, 20, 30)

MODEL_NAMES = (
    "model_clean",
    "model_mixed_p10",
    "model_mixed_p20",
    "model_mixed_p30",
    "model_systematic_p10",
    "model_systematic_p20",
    "model_systematic_p30",
    "model_unseen_p10",
    "model_unseen_p20",
    "model_unseen_p30",
)

# Implements B -> A, C -> B, and A -> C for
# A=airplane, B=monkey, and C=truck.
SYSTEMATIC_DESTINATION = {
    "airplane": "truck",
    "monkey": "airplane",
    "truck": "monkey",
}


@dataclass(frozen=True)
class Assignment:
    source: Path
    true_class: str
    assigned_class: str
    contamination_type: str


def sorted_pngs(path: Path) -> list[Path]:
    files = list(path.glob("*.png"))
    return sorted(files, key=lambda item: int(item.stem))


def select_source_pools() -> tuple[
    dict[str, list[Path]],
    dict[str, list[Path]],
    dict[str, list[Path]],
]:
    rng = random.Random(SEED)
    train_pools: dict[str, list[Path]] = {}
    test_pools: dict[str, list[Path]] = {}
    unseen_pools: dict[str, list[Path]] = {}

    for class_name in TARGET_CLASSES:
        train_files = sorted_pngs(SOURCE_ROOT / "train" / class_name)
        test_files = sorted_pngs(SOURCE_ROOT / "test" / class_name)
        train_pools[class_name] = rng.sample(train_files, TRAIN_SIZE)
        test_pools[class_name] = rng.sample(test_files, TEST_SIZE)

    for class_name in UNSEEN_CLASSES:
        train_files = sorted_pngs(SOURCE_ROOT / "train" / class_name)
        unseen_pools[class_name] = rng.sample(train_files, max(RATES))

    return train_pools, test_pools, unseen_pools


def build_unseen_sequences(
    unseen_pools: dict[str, list[Path]],
) -> dict[str, list[tuple[Path, str]]]:
    """Build unique, balanced, nested unseen contamination sequences."""
    rng = random.Random(f"{SEED}:unseen-sequences")
    cursor = {class_name: 0 for class_name in UNSEEN_CLASSES}

    # Each ten-image block is balanced globally across the three target classes.
    block_counts = {
        "airplane": (
            {"dog": 4, "car": 3, "ship": 3},
            {"dog": 3, "car": 4, "ship": 3},
            {"dog": 3, "car": 3, "ship": 4},
        ),
        "monkey": (
            {"dog": 3, "car": 4, "ship": 3},
            {"dog": 3, "car": 3, "ship": 4},
            {"dog": 4, "car": 3, "ship": 3},
        ),
        "truck": (
            {"dog": 3, "car": 3, "ship": 4},
            {"dog": 4, "car": 3, "ship": 3},
            {"dog": 3, "car": 4, "ship": 3},
        ),
    }

    sequences: dict[str, list[tuple[Path, str]]] = {
        class_name: [] for class_name in TARGET_CLASSES
    }
    for target_class in TARGET_CLASSES:
        for block in block_counts[target_class]:
            block_entries: list[tuple[Path, str]] = []
            for true_class, count in block.items():
                start = cursor[true_class]
                end = start + count
                selected = unseen_pools[true_class][start:end]
                cursor[true_class] = end
                block_entries.extend((path, true_class) for path in selected)
            rng.shuffle(block_entries)
            sequences[target_class].extend(block_entries)

    for class_name in UNSEEN_CLASSES:
        if cursor[class_name] != max(RATES):
            raise ValueError(f"Unused unseen images for {class_name}: {cursor[class_name]}")
    return sequences


def clean_assignments(train_pools: dict[str, list[Path]]) -> dict[str, list[Assignment]]:
    return {
        class_name: [
            Assignment(path, class_name, class_name, "clean")
            for path in train_pools[class_name]
        ]
        for class_name in TARGET_CLASSES
    }


def mixed_assignments(
    train_pools: dict[str, list[Path]], rate: int
) -> dict[str, list[Assignment]]:
    assignments = {class_name: [] for class_name in TARGET_CLASSES}
    for source_class in TARGET_CLASSES:
        corruption_candidates = train_pools[source_class][:rate]
        correct_images = train_pools[source_class][rate:]
        assignments[source_class].extend(
            Assignment(path, source_class, source_class, "clean")
            for path in correct_images
        )

        destinations = [name for name in TARGET_CLASSES if name != source_class]
        for index, path in enumerate(corruption_candidates):
            destination = destinations[index % 2]
            assignments[destination].append(
                Assignment(path, source_class, destination, "mixed")
            )
    return assignments


def systematic_assignments(
    train_pools: dict[str, list[Path]], rate: int
) -> dict[str, list[Assignment]]:
    assignments = {class_name: [] for class_name in TARGET_CLASSES}
    for source_class in TARGET_CLASSES:
        corruption_candidates = train_pools[source_class][:rate]
        correct_images = train_pools[source_class][rate:]
        assignments[source_class].extend(
            Assignment(path, source_class, source_class, "clean")
            for path in correct_images
        )

        destination = SYSTEMATIC_DESTINATION[source_class]
        assignments[destination].extend(
            Assignment(path, source_class, destination, "systematic")
            for path in corruption_candidates
        )
    return assignments


def unseen_assignments(
    train_pools: dict[str, list[Path]],
    unseen_sequences: dict[str, list[tuple[Path, str]]],
    rate: int,
) -> dict[str, list[Assignment]]:
    assignments = {class_name: [] for class_name in TARGET_CLASSES}
    for target_class in TARGET_CLASSES:
        assignments[target_class].extend(
            Assignment(path, target_class, target_class, "clean")
            for path in train_pools[target_class][rate:]
        )
        assignments[target_class].extend(
            Assignment(path, true_class, target_class, "unseen")
            for path, true_class in unseen_sequences[target_class][:rate]
        )
    return assignments


def validate_assignments(
    model_name: str,
    assignments: dict[str, list[Assignment]],
    expected_rate: int,
) -> None:
    all_sources: list[Path] = []
    for assigned_class in TARGET_CLASSES:
        entries = assignments[assigned_class]
        if len(entries) != TRAIN_SIZE:
            raise ValueError(
                f"{model_name}/{assigned_class}: expected {TRAIN_SIZE}, got {len(entries)}"
            )
        contaminated = sum(entry.contamination_type != "clean" for entry in entries)
        if contaminated != expected_rate:
            raise ValueError(
                f"{model_name}/{assigned_class}: expected {expected_rate} contaminated, "
                f"got {contaminated}"
            )
        all_sources.extend(entry.source for entry in entries)

    if len(all_sources) != len(set(all_sources)):
        raise ValueError(f"{model_name}: duplicate source image in training data")


def copy_model_dataset(
    model_name: str,
    contamination_type: str,
    rate: int,
    train_assignments: dict[str, list[Assignment]],
    test_pools: dict[str, list[Path]],
    manifest_writer: csv.writer,
    summary_writer: csv.writer,
) -> None:
    model_dir = DATA_ROOT / model_name
    if model_dir.exists():
        shutil.rmtree(model_dir)

    validate_assignments(model_name, train_assignments, rate)

    for assigned_class in TARGET_CLASSES:
        train_dir = model_dir / assigned_class / "train"
        test_dir = model_dir / assigned_class / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        train_entries = list(train_assignments[assigned_class])
        random.Random(f"{SEED}:{model_name}:{assigned_class}:train").shuffle(train_entries)
        for output_index, entry in enumerate(train_entries, start=1):
            output_path = train_dir / f"{output_index}.png"
            shutil.copy2(entry.source, output_path)
            manifest_writer.writerow(
                (
                    model_name,
                    "train",
                    assigned_class,
                    output_index,
                    entry.true_class,
                    entry.contamination_type,
                    rate,
                    entry.source.relative_to(REPOSITORY_ROOT).as_posix(),
                    output_path.relative_to(REPOSITORY_ROOT).as_posix(),
                )
            )

        test_entries = list(test_pools[assigned_class])
        random.Random(f"{SEED}:{assigned_class}:test").shuffle(test_entries)
        for output_index, source in enumerate(test_entries, start=1):
            output_path = test_dir / f"{output_index}.png"
            shutil.copy2(source, output_path)
            manifest_writer.writerow(
                (
                    model_name,
                    "test",
                    assigned_class,
                    output_index,
                    assigned_class,
                    "clean",
                    0,
                    source.relative_to(REPOSITORY_ROOT).as_posix(),
                    output_path.relative_to(REPOSITORY_ROOT).as_posix(),
                )
            )

        clean_count = sum(
            entry.contamination_type == "clean" for entry in train_entries
        )
        summary_writer.writerow(
            (
                model_name,
                contamination_type,
                rate,
                assigned_class,
                len(train_entries),
                clean_count,
                len(train_entries) - clean_count,
                len(test_entries),
            )
        )


def main() -> None:
    train_pools, test_pools, unseen_pools = select_source_pools()
    unseen_sequences = build_unseen_sequences(unseen_pools)

    model_specs: list[tuple[str, str, int, dict[str, list[Assignment]]]] = [
        ("model_clean", "clean", 0, clean_assignments(train_pools))
    ]
    for rate in RATES:
        model_specs.append(
            (
                f"model_mixed_p{rate}",
                "mixed",
                rate,
                mixed_assignments(train_pools, rate),
            )
        )
    for rate in RATES:
        model_specs.append(
            (
                f"model_systematic_p{rate}",
                "systematic",
                rate,
                systematic_assignments(train_pools, rate),
            )
        )
    for rate in RATES:
        model_specs.append(
            (
                f"model_unseen_p{rate}",
                "unseen",
                rate,
                unseen_assignments(train_pools, unseen_sequences, rate),
            )
        )

    if tuple(spec[0] for spec in model_specs) != MODEL_NAMES:
        raise ValueError("Generated model order does not match MODEL_NAMES")

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as manifest_file, \
        SUMMARY_PATH.open("w", newline="", encoding="utf-8") as summary_file:
        manifest_writer = csv.writer(manifest_file)
        manifest_writer.writerow(
            (
                "model",
                "split",
                "assigned_class",
                "output_index",
                "true_class",
                "contamination_type",
                "contamination_rate",
                "source_path",
                "output_path",
            )
        )

        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(
            (
                "model",
                "contamination_type",
                "contamination_rate",
                "assigned_class",
                "train_total",
                "train_clean",
                "train_contaminated",
                "test_total",
            )
        )

        for model_name, contamination_type, rate, assignments in model_specs:
            copy_model_dataset(
                model_name,
                contamination_type,
                rate,
                assignments,
                test_pools,
                manifest_writer,
                summary_writer,
            )

    print(f"Random seed: {SEED}")
    print(f"Created {len(model_specs)} model datasets under {DATA_ROOT}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

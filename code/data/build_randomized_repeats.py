"""Build randomized train/test datasets for training repetitions 4 through 10.

Within one repetition, all ten experimental conditions share the same target
image pools and the same ten test sets. Different repetitions use deterministic
random samples from the STL-10 source pools; overlap is allowed by design.
"""

from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "datasets" / "source" / "STL10" / "png"
OUTPUT_ROOT = REPOSITORY_ROOT / "datasets" / "repeats"

BASE_SEED = 20261115
REPEATS = range(4, 11)
TARGET_CLASSES = ("airplane", "monkey", "truck")
UNSEEN_CLASSES = ("dog", "car", "ship")
RATES = (10, 20, 30)
TRAIN_SIZE = 100
TEST_SIZE = 50
TEST_SET_COUNT = 10

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
    return sorted(path.glob("*.png"), key=lambda item: int(item.stem))


def sample_repeat_pools(
    repeat: int,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]], dict[str, list[Path]]]:
    rng = random.Random(f"{BASE_SEED}:repeat:{repeat}:pools")
    train_pools: dict[str, list[Path]] = {}
    test_pools: dict[str, list[Path]] = {}
    unseen_pools: dict[str, list[Path]] = {}

    for class_name in TARGET_CLASSES:
        train_pools[class_name] = rng.sample(
            sorted_pngs(SOURCE_ROOT / "train" / class_name), TRAIN_SIZE
        )
        test_pools[class_name] = rng.sample(
            sorted_pngs(SOURCE_ROOT / "test" / class_name),
            TEST_SIZE * TEST_SET_COUNT,
        )

    for class_name in UNSEEN_CLASSES:
        unseen_pools[class_name] = rng.sample(
            sorted_pngs(SOURCE_ROOT / "train" / class_name), max(RATES)
        )

    return train_pools, test_pools, unseen_pools


def build_unseen_sequences(
    repeat: int,
    unseen_pools: dict[str, list[Path]],
) -> dict[str, list[tuple[Path, str]]]:
    rng = random.Random(f"{BASE_SEED}:repeat:{repeat}:unseen")
    cursor = {class_name: 0 for class_name in UNSEEN_CLASSES}
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
            entries: list[tuple[Path, str]] = []
            for true_class, count in block.items():
                start = cursor[true_class]
                end = start + count
                entries.extend(
                    (path, true_class) for path in unseen_pools[true_class][start:end]
                )
                cursor[true_class] = end
            rng.shuffle(entries)
            sequences[target_class].extend(entries)
    return sequences


def clean_assignments(train_pools: dict[str, list[Path]]) -> dict[str, list[Assignment]]:
    return {
        name: [Assignment(path, name, name, "clean") for path in train_pools[name]]
        for name in TARGET_CLASSES
    }


def target_noise_assignments(
    train_pools: dict[str, list[Path]], rate: int, systematic: bool
) -> dict[str, list[Assignment]]:
    assignments = {name: [] for name in TARGET_CLASSES}
    noise_type = "systematic" if systematic else "mixed"
    for source_class in TARGET_CLASSES:
        contaminated = train_pools[source_class][:rate]
        assignments[source_class].extend(
            Assignment(path, source_class, source_class, "clean")
            for path in train_pools[source_class][rate:]
        )
        if systematic:
            destinations = [SYSTEMATIC_DESTINATION[source_class]]
        else:
            destinations = [name for name in TARGET_CLASSES if name != source_class]
        for index, path in enumerate(contaminated):
            destination = destinations[index % len(destinations)]
            assignments[destination].append(
                Assignment(path, source_class, destination, noise_type)
            )
    return assignments


def unseen_assignments(
    train_pools: dict[str, list[Path]],
    unseen_sequences: dict[str, list[tuple[Path, str]]],
    rate: int,
) -> dict[str, list[Assignment]]:
    assignments = {name: [] for name in TARGET_CLASSES}
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
    sources: list[Path] = []
    for assigned_class in TARGET_CLASSES:
        entries = assignments[assigned_class]
        if len(entries) != TRAIN_SIZE:
            raise ValueError(f"{model_name}/{assigned_class}: {len(entries)} images")
        contaminated = sum(item.contamination_type != "clean" for item in entries)
        if contaminated != expected_rate:
            raise ValueError(
                f"{model_name}/{assigned_class}: {contaminated} contaminated images"
            )
        sources.extend(item.source for item in entries)
    if len(sources) != len(set(sources)):
        raise ValueError(f"{model_name}: duplicate source image")


def write_training_condition(
    repeat: int,
    model_name: str,
    rate: int,
    assignments: dict[str, list[Assignment]],
    manifest: csv.writer,
) -> None:
    validate_assignments(model_name, assignments, rate)
    condition_root = OUTPUT_ROOT / f"repeat_{repeat:02d}" / "train" / model_name
    if condition_root.exists():
        shutil.rmtree(condition_root)

    for assigned_class in TARGET_CLASSES:
        output_dir = condition_root / assigned_class
        output_dir.mkdir(parents=True, exist_ok=True)
        entries = list(assignments[assigned_class])
        random.Random(
            f"{BASE_SEED}:repeat:{repeat}:{model_name}:{assigned_class}"
        ).shuffle(entries)
        for output_index, entry in enumerate(entries, start=1):
            output_path = output_dir / f"{output_index}.png"
            shutil.copy2(entry.source, output_path)
            manifest.writerow(
                (
                    repeat,
                    model_name,
                    assigned_class,
                    output_index,
                    entry.true_class,
                    entry.contamination_type,
                    rate,
                    entry.source.relative_to(REPOSITORY_ROOT).as_posix(),
                    output_path.relative_to(REPOSITORY_ROOT).as_posix(),
                )
            )


def write_test_sets(
    repeat: int,
    test_pools: dict[str, list[Path]],
    manifest: csv.writer,
) -> None:
    test_root = OUTPUT_ROOT / f"repeat_{repeat:02d}" / "test_sets"
    if test_root.exists():
        shutil.rmtree(test_root)

    for test_index in range(TEST_SET_COUNT):
        test_name = f"test_{test_index + 1:02d}"
        start = test_index * TEST_SIZE
        end = start + TEST_SIZE
        for class_name in TARGET_CLASSES:
            output_dir = test_root / test_name / class_name / "test"
            output_dir.mkdir(parents=True, exist_ok=True)
            selected = list(test_pools[class_name][start:end])
            random.Random(
                f"{BASE_SEED}:repeat:{repeat}:{test_name}:{class_name}"
            ).shuffle(selected)
            for output_index, source in enumerate(selected, start=1):
                output_path = output_dir / f"{output_index}.png"
                shutil.copy2(source, output_path)
                manifest.writerow(
                    (
                        repeat,
                        test_name,
                        class_name,
                        output_index,
                        source.relative_to(REPOSITORY_ROOT).as_posix(),
                        output_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    )
                )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_manifest_path = OUTPUT_ROOT / "train_manifest.csv"
    test_manifest_path = OUTPUT_ROOT / "test_manifest.csv"
    with train_manifest_path.open("w", newline="", encoding="utf-8") as train_file, \
        test_manifest_path.open("w", newline="", encoding="utf-8") as test_file:
        train_writer = csv.writer(train_file)
        train_writer.writerow(
            (
                "training_repeat",
                "model",
                "assigned_class",
                "output_index",
                "true_class",
                "contamination_type",
                "contamination_rate",
                "source_path",
                "output_path",
            )
        )
        test_writer = csv.writer(test_file)
        test_writer.writerow(
            (
                "training_repeat",
                "test_set",
                "class",
                "output_index",
                "source_path",
                "output_path",
            )
        )

        for repeat in REPEATS:
            train_pools, test_pools, unseen_pools = sample_repeat_pools(repeat)
            unseen_sequences = build_unseen_sequences(repeat, unseen_pools)
            specs = [("model_clean", 0, clean_assignments(train_pools))]
            for rate in RATES:
                specs.append(
                    (
                        f"model_mixed_p{rate}",
                        rate,
                        target_noise_assignments(train_pools, rate, systematic=False),
                    )
                )
            for rate in RATES:
                specs.append(
                    (
                        f"model_systematic_p{rate}",
                        rate,
                        target_noise_assignments(train_pools, rate, systematic=True),
                    )
                )
            for rate in RATES:
                specs.append(
                    (
                        f"model_unseen_p{rate}",
                        rate,
                        unseen_assignments(train_pools, unseen_sequences, rate),
                    )
                )

            for model_name, rate, assignments in specs:
                write_training_condition(
                    repeat, model_name, rate, assignments, train_writer
                )
            write_test_sets(repeat, test_pools, test_writer)

    print(f"Seed base: {BASE_SEED}")
    print(f"Created repetitions {min(REPEATS)}-{max(REPEATS)} under {OUTPUT_ROOT}")
    print(f"Training manifest: {train_manifest_path}")
    print(f"Test manifest: {test_manifest_path}")


if __name__ == "__main__":
    main()

"""Shared paths, labels, image loading, and metrics for PA4 evaluation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPOSITORY_ROOT / "models"
DATASETS_ROOT = REPOSITORY_ROOT / "datasets"
FIXED_DATA_ROOT = DATASETS_ROOT / "fixed"
REPEATED_DATA_ROOT = DATASETS_ROOT / "repeats"

CLASS_NAMES = ("airplane", "monkey", "truck")
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
TRAINING_REPEATS = tuple(range(1, 11))
TEST_SET_NAMES = tuple(f"test_{index:02d}" for index in range(1, 11))


def run_id(training_repeat: int) -> str:
    return f"run_{training_repeat:02d}"


def model_directory(model_name: str, training_repeat: int) -> Path:
    return MODELS_ROOT / model_name / run_id(training_repeat)


def test_sets_root(training_repeat: int) -> Path:
    if training_repeat <= 3:
        return FIXED_DATA_ROOT / "test_sets"
    return REPEATED_DATA_ROOT / f"repeat_{training_repeat:02d}" / "test_sets"


def read_labels(labels_path: Path) -> list[str]:
    labels: list[str] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid label line in {labels_path}: {line!r}")
        labels.append(parts[1])
    if tuple(labels) != CLASS_NAMES:
        raise ValueError(f"Unexpected label order in {labels_path}: {labels}")
    return labels


def read_image(image_path: Path) -> np.ndarray:
    """Read an image even when its Windows path contains non-ASCII text."""
    encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return image


def load_test_batch(
    test_root: Path,
    test_set_name: str,
    class_names: list[str],
    input_height: int,
    input_width: int,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    images: list[np.ndarray] = []
    true_indices: list[int] = []
    image_paths: list[Path] = []

    for true_index, class_name in enumerate(class_names):
        test_dir = test_root / test_set_name / class_name / "test"
        paths = sorted(test_dir.glob("*.png"), key=lambda path: int(path.stem))
        if len(paths) != 50:
            raise ValueError(f"Expected 50 images in {test_dir}, found {len(paths)}")
        for image_path in paths:
            image_bgr = read_image(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                image_rgb,
                (input_width, input_height),
                interpolation=cv2.INTER_AREA,
            )
            images.append((resized.astype(np.float32) / 127.5) - 1.0)
            true_indices.append(true_index)
            image_paths.append(image_path)

    return np.stack(images), np.asarray(true_indices), image_paths


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def calculate_metrics(
    confusion: np.ndarray, class_names: list[str]
) -> tuple[list[dict[str, object]], float, float]:
    rows: list[dict[str, object]] = []
    f1_values: list[float] = []
    for index, class_name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1_score = safe_divide(2 * precision * recall, precision + recall)
        f1_values.append(f1_score)
        rows.append(
            {
                "class": class_name,
                "precision": f"{precision:.8f}",
                "recall": f"{recall:.8f}",
                "f1_score": f"{f1_score:.8f}",
                "support": support,
                "correct": true_positive,
            }
        )
    accuracy = safe_divide(int(np.trace(confusion)), int(confusion.sum()))
    return rows, accuracy, float(np.mean(f1_values))


def parse_condition(model_name: str) -> tuple[str, int]:
    if model_name == "model_clean":
        return "clean", 0
    for contamination_type in ("mixed", "systematic", "unseen"):
        prefix = f"model_{contamination_type}_p"
        if model_name.startswith(prefix):
            return contamination_type, int(model_name.removeprefix(prefix))
    raise ValueError(f"Unknown model condition: {model_name}")

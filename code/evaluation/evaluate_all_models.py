"""Evaluate all final PA4 models and write one CSV per result type.

The default configuration evaluates ten conditions, ten trained models per
condition, and ten clean test sets per model. Edit the three selection
constants below for a smaller run, or use the optional CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import load_model

from evaluation_utils import (
    CLASS_NAMES,
    MODEL_NAMES,
    REPOSITORY_ROOT,
    TEST_SET_NAMES,
    TRAINING_REPEATS,
    calculate_metrics,
    load_test_batch,
    model_directory,
    parse_condition,
    read_labels,
    run_id,
    safe_divide,
    test_sets_root,
)


# Comment out entries here only when intentionally running a subset.
SELECTED_MODELS = list(MODEL_NAMES)
SELECTED_REPEATS = list(TRAINING_REPEATS)
SELECTED_TEST_SETS = list(TEST_SET_NAMES)

BATCH_SIZE = 32
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results"

EVALUATION_FIELDS = [
    "model",
    "run_id",
    "training_repeat",
    "contamination_type",
    "contamination_rate",
    "test_set",
    "total_images",
    "correct_images",
    "overall_accuracy",
    "macro_f1",
    "recall_airplane",
    "recall_monkey",
    "recall_truck",
]
PREDICTION_FIELDS = [
    "model",
    "run_id",
    "training_repeat",
    "test_set",
    "image",
    "true_label",
    "predicted_label",
    "correct",
    "confidence",
    "score_airplane",
    "score_monkey",
    "score_truck",
]
METRIC_FIELDS = [
    "model",
    "run_id",
    "training_repeat",
    "test_set",
    "class",
    "precision",
    "recall",
    "f1_score",
    "support",
    "correct",
]
CONFUSION_FIELDS = [
    "model",
    "run_id",
    "training_repeat",
    "test_set",
    "true_class",
    "predicted_airplane",
    "predicted_monkey",
    "predicted_truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--repeats", nargs="+", type=int, choices=TRAINING_REPEATS)
    parser.add_argument(
        "--test-sets",
        nargs="+",
        type=int,
        choices=range(1, 11),
        metavar="N",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; relative paths start at the repository root.",
    )
    return parser.parse_args()


def resolve_output_dir(path: Path | None) -> Path:
    if path is None:
        return DEFAULT_RESULTS_ROOT
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def safe_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def write_rows(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def confusion_rows(
    model_name: str,
    training_repeat: int,
    test_set_name: str,
    class_names: list[str],
    confusion: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, true_class in enumerate(class_names):
        rows.append(
            {
                "model": model_name,
                "run_id": run_id(training_repeat),
                "training_repeat": training_repeat,
                "test_set": test_set_name,
                "true_class": true_class,
                "predicted_airplane": int(confusion[index, 0]),
                "predicted_monkey": int(confusion[index, 1]),
                "predicted_truck": int(confusion[index, 2]),
            }
        )
    return rows


def evaluate_model_run(
    model_name: str,
    training_repeat: int,
    test_set_names: list[str],
    prediction_writer: csv.DictWriter,
    metric_writer: csv.DictWriter,
    confusion_writer: csv.DictWriter,
) -> list[dict[str, object]]:
    model_dir = model_directory(model_name, training_repeat)
    model_file = model_dir / "keras_model.h5"
    labels_file = model_dir / "labels.txt"
    if not model_file.is_file() or not labels_file.is_file():
        raise FileNotFoundError(f"Incomplete model folder: {model_dir}")

    keras_backend.clear_session()
    model = load_model(model_file, compile=False)
    class_names = read_labels(labels_file)
    input_height = int(model.input_shape[1])
    input_width = int(model.input_shape[2])
    contamination_type, contamination_rate = parse_condition(model_name)
    active_test_root = test_sets_root(training_repeat)
    if not active_test_root.is_dir():
        raise FileNotFoundError(f"Missing test-set root: {active_test_root}")

    summaries: list[dict[str, object]] = []
    aggregate_confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    for test_set_name in test_set_names:
        images, true_indices, image_paths = load_test_batch(
            active_test_root,
            test_set_name,
            class_names,
            input_height,
            input_width,
        )
        scores = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
        predicted_indices = np.argmax(scores, axis=1)
        confusion = np.zeros_like(aggregate_confusion)
        for true_index, predicted_index in zip(
            true_indices, predicted_indices, strict=True
        ):
            confusion[int(true_index), int(predicted_index)] += 1
        aggregate_confusion += confusion

        metric_rows, accuracy, macro_f1 = calculate_metrics(confusion, class_names)
        for metric in metric_rows:
            metric_writer.writerow(
                {
                    "model": model_name,
                    "run_id": run_id(training_repeat),
                    "training_repeat": training_repeat,
                    "test_set": test_set_name,
                    **metric,
                }
            )
        confusion_writer.writerows(
            confusion_rows(
                model_name,
                training_repeat,
                test_set_name,
                class_names,
                confusion,
            )
        )

        for row_index, image_path in enumerate(image_paths):
            true_index = int(true_indices[row_index])
            predicted_index = int(predicted_indices[row_index])
            prediction_writer.writerow(
                {
                    "model": model_name,
                    "run_id": run_id(training_repeat),
                    "training_repeat": training_repeat,
                    "test_set": test_set_name,
                    "image": image_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "true_label": class_names[true_index],
                    "predicted_label": class_names[predicted_index],
                    "correct": true_index == predicted_index,
                    "confidence": f"{float(scores[row_index, predicted_index]):.8f}",
                    "score_airplane": f"{float(scores[row_index, 0]):.8f}",
                    "score_monkey": f"{float(scores[row_index, 1]):.8f}",
                    "score_truck": f"{float(scores[row_index, 2]):.8f}",
                }
            )

        summary: dict[str, object] = {
            "model": model_name,
            "run_id": run_id(training_repeat),
            "training_repeat": training_repeat,
            "contamination_type": contamination_type,
            "contamination_rate": contamination_rate,
            "test_set": test_set_name,
            "total_images": int(confusion.sum()),
            "correct_images": int(np.trace(confusion)),
            "overall_accuracy": f"{accuracy:.8f}",
            "macro_f1": f"{macro_f1:.8f}",
        }
        for index, class_name in enumerate(class_names):
            correct = int(confusion[index, index])
            support = int(confusion[index, :].sum())
            summary[f"recall_{class_name}"] = f"{safe_divide(correct, support):.8f}"
        summaries.append(summary)
        print(
            f"{model_name:24s} {run_id(training_repeat)} {test_set_name}: "
            f"accuracy={accuracy * 100:6.2f}%",
            flush=True,
        )

    confusion_writer.writerows(
        confusion_rows(
            model_name,
            training_repeat,
            "ALL",
            class_names,
            aggregate_confusion,
        )
    )
    return summaries


def summarize_model_runs(
    all_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    keys = sorted(
        {(str(row["model"]), int(row["training_repeat"])) for row in all_rows}
    )
    for model_name, training_repeat in keys:
        rows = [
            row
            for row in all_rows
            if row["model"] == model_name
            and int(row["training_repeat"]) == training_repeat
        ]
        accuracies = [float(row["overall_accuracy"]) for row in rows]
        macro_f1_values = [float(row["macro_f1"]) for row in rows]
        summary: dict[str, object] = {
            "model": model_name,
            "run_id": run_id(training_repeat),
            "training_repeat": training_repeat,
            "test_sets": len(rows),
            "images_per_test_set": int(rows[0]["total_images"]),
            "accuracy_mean": f"{statistics.mean(accuracies):.8f}",
            "accuracy_sample_std": f"{safe_stdev(accuracies):.8f}",
            "accuracy_min": f"{min(accuracies):.8f}",
            "accuracy_max": f"{max(accuracies):.8f}",
            "macro_f1_mean": f"{statistics.mean(macro_f1_values):.8f}",
            "macro_f1_sample_std": f"{safe_stdev(macro_f1_values):.8f}",
        }
        for class_name in CLASS_NAMES:
            recalls = [float(row[f"recall_{class_name}"]) for row in rows]
            summary[f"recall_{class_name}_mean"] = f"{statistics.mean(recalls):.8f}"
            summary[f"recall_{class_name}_sample_std"] = f"{safe_stdev(recalls):.8f}"
        output.append(summary)
    return output


def summarize_conditions(
    model_run_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for model_name in sorted({str(row["model"]) for row in model_run_rows}):
        rows = [row for row in model_run_rows if row["model"] == model_name]
        accuracies = [float(row["accuracy_mean"]) for row in rows]
        macro_f1_values = [float(row["macro_f1_mean"]) for row in rows]
        contamination_type, contamination_rate = parse_condition(model_name)
        summary: dict[str, object] = {
            "model": model_name,
            "contamination_type": contamination_type,
            "contamination_rate": contamination_rate,
            "training_repeats": len(rows),
            "test_sets_per_repeat": int(rows[0]["test_sets"]),
            "accuracy_mean_across_training_repeats": f"{statistics.mean(accuracies):.8f}",
            "accuracy_training_repeat_sample_std": f"{safe_stdev(accuracies):.8f}",
            "accuracy_min_training_repeat_mean": f"{min(accuracies):.8f}",
            "accuracy_max_training_repeat_mean": f"{max(accuracies):.8f}",
            "macro_f1_mean_across_training_repeats": f"{statistics.mean(macro_f1_values):.8f}",
            "macro_f1_training_repeat_sample_std": f"{safe_stdev(macro_f1_values):.8f}",
        }
        for class_name in CLASS_NAMES:
            recalls = [float(row[f"recall_{class_name}_mean"]) for row in rows]
            summary[f"recall_{class_name}_mean_across_training_repeats"] = f"{statistics.mean(recalls):.8f}"
            summary[f"recall_{class_name}_training_repeat_sample_std"] = f"{safe_stdev(recalls):.8f}"
        output.append(summary)
    return output


def paired_differences(
    all_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    clean = {
        (int(row["training_repeat"]), str(row["test_set"])): float(
            row["overall_accuracy"]
        )
        for row in all_rows
        if row["model"] == "model_clean"
    }
    output: list[dict[str, object]] = []
    for row in all_rows:
        if row["model"] == "model_clean":
            continue
        key = (int(row["training_repeat"]), str(row["test_set"]))
        if key not in clean:
            continue
        model_accuracy = float(row["overall_accuracy"])
        output.append(
            {
                "model": row["model"],
                "run_id": row["run_id"],
                "training_repeat": row["training_repeat"],
                "test_set": row["test_set"],
                "model_accuracy": f"{model_accuracy:.8f}",
                "clean_accuracy_same_repeat": f"{clean[key]:.8f}",
                "difference_vs_clean": f"{model_accuracy - clean[key]:.8f}",
            }
        )
    return output


def write_results_readme(results_root: Path) -> None:
    (results_root / "README.md").write_text(
        "# Evaluation results\n\n"
        "Results are grouped by measurement rather than by model folder. Rows "
        "carry the model condition, training run, and test-set identifiers "
        "needed to trace an observation back to its source.\n\n"
        "- `all_evaluations.csv` has one summary per model, run, and test set.\n"
        "- `all_predictions.csv` has one row per image prediction.\n"
        "- `class_metrics.csv` records class-level precision, recall, and F1.\n"
        "- `confusion_matrices.csv` records test-level and aggregate confusion "
        "matrices.\n"
        "- `model_run_statistics.csv` summarizes each trained model.\n"
        "- `condition_statistics.csv` summarizes each data condition.\n"
        "- `paired_differences_vs_clean.csv` pairs each accuracy with the clean "
        "condition from the same run and test set.\n",
        encoding="utf-8",
    )


def run_evaluation(
    model_names: list[str],
    training_repeats: list[int],
    test_set_names: list[str],
    results_root: Path,
) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    generated_names = (
        "all_evaluations.csv",
        "all_predictions.csv",
        "class_metrics.csv",
        "confusion_matrices.csv",
        "model_run_statistics.csv",
        "condition_statistics.csv",
        "paired_differences_vs_clean.csv",
    )
    for name in generated_names:
        path = results_root / name
        if path.exists():
            path.unlink()

    all_rows: list[dict[str, object]] = []
    with ExitStack() as stack:
        prediction_file = stack.enter_context(
            (results_root / "all_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            )
        )
        metric_file = stack.enter_context(
            (results_root / "class_metrics.csv").open(
                "w", newline="", encoding="utf-8"
            )
        )
        confusion_file = stack.enter_context(
            (results_root / "confusion_matrices.csv").open(
                "w", newline="", encoding="utf-8"
            )
        )
        prediction_writer = csv.DictWriter(
            prediction_file, fieldnames=PREDICTION_FIELDS
        )
        metric_writer = csv.DictWriter(metric_file, fieldnames=METRIC_FIELDS)
        confusion_writer = csv.DictWriter(
            confusion_file, fieldnames=CONFUSION_FIELDS
        )
        prediction_writer.writeheader()
        metric_writer.writeheader()
        confusion_writer.writeheader()

        for model_name in model_names:
            for training_repeat in training_repeats:
                all_rows.extend(
                    evaluate_model_run(
                        model_name,
                        training_repeat,
                        test_set_names,
                        prediction_writer,
                        metric_writer,
                        confusion_writer,
                    )
                )

    model_run_rows = summarize_model_runs(all_rows)
    condition_rows = summarize_conditions(model_run_rows)
    paired_rows = paired_differences(all_rows)
    paired_fields = [
        "model",
        "run_id",
        "training_repeat",
        "test_set",
        "model_accuracy",
        "clean_accuracy_same_repeat",
        "difference_vs_clean",
    ]
    write_rows(results_root / "all_evaluations.csv", EVALUATION_FIELDS, all_rows)
    write_rows(
        results_root / "model_run_statistics.csv",
        list(model_run_rows[0].keys()),
        model_run_rows,
    )
    write_rows(
        results_root / "condition_statistics.csv",
        list(condition_rows[0].keys()),
        condition_rows,
    )
    write_rows(
        results_root / "paired_differences_vs_clean.csv",
        paired_fields,
        paired_rows,
    )
    write_results_readme(results_root)
    print(f"\nCompleted {len(all_rows)} model/run/test-set evaluations.")
    print(f"Results: {results_root}")


def main() -> None:
    args = parse_args()
    models = list(args.models) if args.models else SELECTED_MODELS
    repeats = list(args.repeats) if args.repeats else SELECTED_REPEATS
    test_sets = (
        [f"test_{index:02d}" for index in args.test_sets]
        if args.test_sets
        else SELECTED_TEST_SETS
    )
    run_evaluation(models, repeats, test_sets, resolve_output_dir(args.output_dir))


if __name__ == "__main__":
    main()

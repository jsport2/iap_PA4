"""Recreate the report's accuracy plot and confusion-matrix panel."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "results"
FIGURE_DIR = REPOSITORY_ROOT / "report" / "figures"


def load_statistics() -> dict[str, dict[str, str]]:
    with (RESULTS_ROOT / "condition_statistics.csv").open(
        "r", newline="", encoding="utf-8"
    ) as file:
        return {row["model"]: row for row in csv.DictReader(file)}


def main() -> None:
    statistics_rows = load_statistics()
    rates = np.array([10, 20, 30])
    conditions = {
        "Mixed": "model_mixed_p",
        "Systematic": "model_systematic_p",
        "Unseen": "model_unseen_p",
    }
    colors = {
        "Mixed": "#2878B5",
        "Systematic": "#C43C39",
        "Unseen": "#54A24B",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
        }
    )
    figure, (accuracy_axis, matrix_axis) = plt.subplots(
        1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.18, 1.0]}
    )

    for label, prefix in conditions.items():
        means = [
            100
            * float(
                statistics_rows[f"{prefix}{rate}"][
                    "accuracy_mean_across_training_repeats"
                ]
            )
            for rate in rates
        ]
        deviations = [
            100
            * float(
                statistics_rows[f"{prefix}{rate}"][
                    "accuracy_training_repeat_sample_std"
                ]
            )
            for rate in rates
        ]
        accuracy_axis.errorbar(
            rates,
            means,
            yerr=deviations,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors[label],
            label=label,
        )
    clean_accuracy = 100 * float(
        statistics_rows["model_clean"]["accuracy_mean_across_training_repeats"]
    )
    accuracy_axis.axhline(
        clean_accuracy,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"Clean ({clean_accuracy:.1f}%)",
    )
    accuracy_axis.set_xticks(rates)
    accuracy_axis.set_ylim(64, 100)
    accuracy_axis.set_xlabel("Contamination rate (%)")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_title(
        "(a) Accuracy across ten model runs", loc="left", fontweight="bold"
    )
    accuracy_axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    accuracy_axis.spines[["top", "right"]].set_visible(False)
    accuracy_axis.legend(frameon=False, fontsize=7.5, loc="lower left")

    confusion = np.zeros((3, 3), dtype=int)
    with (RESULTS_ROOT / "confusion_matrices.csv").open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["model"] == "model_systematic_p30"
            and row["test_set"] == "ALL"
        ]
    class_index = {name: index for index, name in enumerate(("airplane", "monkey", "truck"))}
    for row in rows:
        index = class_index[row["true_class"]]
        confusion[index] += np.array(
            [
                int(row["predicted_airplane"]),
                int(row["predicted_monkey"]),
                int(row["predicted_truck"]),
            ]
        )
    normalized = confusion / confusion.sum(axis=1, keepdims=True)
    matrix_axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    labels = ["Airplane", "Monkey", "Truck"]
    matrix_axis.set_xticks(range(3), labels, rotation=25, ha="right")
    matrix_axis.set_yticks(range(3), labels)
    matrix_axis.set_xlabel("Predicted class")
    matrix_axis.set_ylabel("True class")
    matrix_axis.set_title(
        "(b) Systematic contamination, 30%", loc="left", fontweight="bold"
    )
    for row in range(3):
        for column in range(3):
            color = "white" if normalized[row, column] > 0.55 else "black"
            matrix_axis.text(
                column,
                row,
                f"{100 * normalized[row, column]:.1f}%\n"
                f"({confusion[row, column]})",
                ha="center",
                va="center",
                color=color,
                fontsize=7.5,
            )

    figure.tight_layout(w_pad=2.5)
    figure.savefig(FIGURE_DIR / "results_overview.pdf", bbox_inches="tight")
    figure.savefig(
        FIGURE_DIR / "results_overview.png", dpi=220, bbox_inches="tight"
    )
    print(FIGURE_DIR / "results_overview.pdf")


if __name__ == "__main__":
    main()

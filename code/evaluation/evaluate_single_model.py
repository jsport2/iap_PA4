"""Run one model/test-set evaluation with editable selections."""

from __future__ import annotations

from evaluate_all_models import REPOSITORY_ROOT, run_evaluation


# Edit only these three values for a small, reproducible check.
MODEL_NAME = "model_clean"
TRAINING_REPEAT = 1
TEST_SET_NUMBER = 1


if __name__ == "__main__":
    run_evaluation(
        [MODEL_NAME],
        [TRAINING_REPEAT],
        [f"test_{TEST_SET_NUMBER:02d}"],
        REPOSITORY_ROOT / "results_single",
    )

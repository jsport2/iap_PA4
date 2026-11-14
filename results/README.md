# Evaluation results

The final results are grouped by measurement rather than by model folder. Rows
carry the model condition, training run, and test-set identifiers needed to
trace an observation back to its source.

- `all_evaluations.csv` has 1,000 model/run/test-set accuracy summaries.
- `all_predictions.csv` has all 150,000 image-level predictions.
- `class_metrics.csv` has 3,000 rows of class-level precision, recall, and F1.
- `confusion_matrices.csv` records both test-level and aggregated confusion
  matrices.
- `model_run_statistics.csv` summarizes the 100 trained models.
- `condition_statistics.csv` summarizes the ten data conditions.
- `paired_differences_vs_clean.csv` contains 900 accuracy differences paired
  with the clean condition from the same run and test set.

# Code

The scripts use repository-relative paths and can be run from the repository
root without editing machine-specific directories.

## Prepare the data

- `data/convert_stl10_parquet_to_png.py` extracts the labeled STL-10 Parquet
  files into class-organized PNG directories.
- `data/build_fixed_datasets.py` builds the ten fixed training conditions.
- `data/build_fixed_test_sets.py` selects ten disjoint fixed test sets.
- `data/build_randomized_repeats.py` builds the training and test data for runs
  4--10.

## Evaluate the models

- `evaluation/evaluate_all_models.py` evaluates the selected conditions, runs,
  and test sets, then writes consolidated CSV files.
- `evaluation/evaluate_single_model.py` provides a short, editable one-model
  check.
- `evaluation/evaluation_utils.py` contains the shared path, image-loading, and
  metric functions.

## Check models and rebuild figures

- `models/verify_model.py` loads one Keras export and checks it against a test
  dataset.
- `visualization/make_result_figure.py` recreates the figure used in the final
  report from the consolidated CSV results.

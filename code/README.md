# Code

## Data preparation

- `data/convert_stl10_parquet_to_png.py`: extracts the labeled STL-10 Parquet
  mirror into class-organized PNG files.
- `data/build_fixed_datasets.py`: creates the ten fixed data conditions.
- `data/build_fixed_test_sets.py`: creates ten disjoint fixed test sets.
- `data/build_randomized_repeats.py`: creates training and test data for runs
  4--10.

## Evaluation

- `evaluation/evaluate_all_models.py`: final evaluator; writes consolidated
  result CSVs.
- `evaluation/evaluate_single_model.py`: editable one-model smoke test.
- `evaluation/evaluation_utils.py`: shared image, path, and metric functions.

## Other utilities

- `models/verify_model.py`: checks one exported Keras model and test dataset.
- `visualization/make_result_figure.py`: recreates the report figure from the
  consolidated final results.

All scripts resolve paths from their own location, so they can be launched from
the repository root without editing absolute paths.

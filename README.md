# Training-Set Contamination in a Three-Class Image Classifier

EF2039 Programming Assignment 4 tests how contaminated training data changes a
three-class image classifier. The repository preserves the processed datasets,
data-preparation and evaluation code, 100 exported Keras models, consolidated
results, and the final two-page report.

## Research questions

1. How does model performance change as the contamination rate rises while
   each assigned class remains fixed at 100 training images?
2. At the same contamination rate, do mixed target-class errors, systematic
   label errors, and unseen-class images produce different losses and error
   patterns?

The target classes are `airplane`, `monkey`, and `truck`. Images of `dog`,
`car`, and `ship` provide the unseen-class contamination. Every condition has
100 assigned training images for each target label, so contamination rate and
type change without changing class size.

## Results

Each value below is the mean accuracy of ten independently trained models.
Every model was tested on ten balanced test sets. The value after `±` is the
sample standard deviation across the ten model means.

| Training condition | 10% | 20% | 30% |
|---|---:|---:|---:|
| Mixed target-class noise | 87.77% ± 2.97 | 79.65% ± 6.08 | 73.65% ± 5.72 |
| Systematic label noise | 86.67% ± 5.08 | 79.95% ± 5.02 | 70.03% ± 6.17 |
| Unseen-class contamination | 93.61% ± 1.56 | 91.95% ± 2.51 | 90.17% ± 2.21 |

The clean condition reached 96.22% ± 0.63. The contaminated conditions became
less accurate as the contamination rate increased, but unseen-class images
caused a smaller loss than either form of wrong target-class labeling. The
complete 1,000 evaluation summaries and 150,000 image-level predictions are in
`results/`.

## Repository contents

```text
.
|-- README.md
|-- code/
|   |-- data/                 # Dataset conversion and sampling
|   |-- evaluation/           # Full and single-model evaluation
|   |-- models/               # Exported-model verification
|   `-- visualization/        # Report figure generation
|-- config/                   # Conda and pip environment files
|-- datasets/
|   |-- fixed/                # Shared data for runs 1--3
|   `-- repeats/              # Run-specific data for runs 4--10
|-- docs/                     # Dataset and model documentation
|-- models/                   # Ten conditions, run_01 through run_10
|-- report/                   # Final PDF and LaTeX source
`-- results/                  # Consolidated evaluation CSV files
```

Only the final ten-run experiment is included. Pilot outputs from earlier
stages were left out of the submission package.

## Environment

The evaluation was verified with Python 3.10, TensorFlow 2.15.0, NumPy 1.26.4,
and OpenCV 4.11.0. The Conda file also installs the packages needed to rebuild
the datasets and figures.

To create the Conda environment:

```powershell
conda env create -f .\config\environment.yml
conda activate pa4
```

To use a standard virtual environment instead:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\config\requirements.txt
```

## Run the evaluation

From the repository root, this command evaluates all ten conditions, ten
training runs per condition, and ten test sets per model:

```powershell
python .\code\evaluation\evaluate_all_models.py
```

It rewrites the seven consolidated CSV files in `results/`, producing 1,000
evaluation summaries and 150,000 image predictions.

For a quick check, edit the three selection values at the top of
`code/evaluation/evaluate_single_model.py` and run:

```powershell
python .\code\evaluation\evaluate_single_model.py
```

The full evaluator also accepts command-line selections. This example tests
the first clean model on the first test set and writes to `results_smoke/`:

```powershell
python .\code\evaluation\evaluate_all_models.py `
  --models model_clean --repeats 1 --test-sets 1 `
  --output-dir results_smoke
```

## Rebuild the report figure

```powershell
python .\code\visualization\make_result_figure.py
```

The script reads the consolidated results and updates the PDF and PNG files in
`report/figures/`.

## Rebuild the datasets

The processed images used by the experiment are already included. To recreate
them from the labeled STL-10 Parquet files, place the raw files at the paths in
`docs/DATASET.md` and run:

```powershell
python .\code\data\convert_stl10_parquet_to_png.py
python .\code\data\build_fixed_datasets.py
python .\code\data\build_fixed_test_sets.py
python .\code\data\build_randomized_repeats.py
```

Runs 1--3 share the fixed training pools and test suite. Runs 4--10 use
deterministically sampled training pools and run-specific test suites. Within
the same run, every condition uses the same target pools and test images.

## Additional documentation

- `docs/DATASET.md` records the source revision, checksums, classes, manifests,
  and reconstruction procedure.
- `docs/MODEL_SETTINGS.md` records the Teachable Machine settings and model
  naming convention.
- `docs/UPLOAD.md` explains how to clone, update, and verify the GitHub copy.
- The README files under `code/`, `datasets/`, `models/`, `report/`, and
  `results/` describe the contents of those folders.

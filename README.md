# Training-Set Contamination in a Three-Class Image Classifier

This repository contains the final code, processed datasets, 100 exported Keras
models, consolidated evaluation results, and two-page report for EF2039
Programming Assignment 4.

## Research questions

1. How does the contamination rate affect performance when each assigned class
   remains fixed at 100 training images?
2. At the same contamination rate, do mixed target-class errors, systematic
   errors, and unseen-class images produce different losses and error patterns?

The target classes are `airplane`, `monkey`, and `truck`. The unseen
contamination classes are `dog`, `car`, and `ship`. Every condition contains
100 assigned training images per target label.

## Final results

The table reports mean accuracy over ten trained models per condition. Each
model was evaluated on ten balanced test sets. Values after the plus-minus sign
are sample standard deviations across the ten model means.

| Training condition | 10% | 20% | 30% |
|---|---:|---:|---:|
| Mixed target-class noise | 87.77% +/- 2.97 | 79.65% +/- 6.08 | 73.65% +/- 5.72 |
| Systematic label noise | 86.67% +/- 5.08 | 79.95% +/- 5.02 | 70.03% +/- 6.17 |
| Unseen-class contamination | 93.61% +/- 1.56 | 91.95% +/- 2.51 | 90.17% +/- 2.21 |

The clean models achieved 96.22% +/- 0.63. The complete 1,000 evaluation
summaries and 150,000 image predictions are stored in consolidated CSV files
under `results/`.

## Repository layout

```text
.
|-- README.md
|-- code/
|   |-- data/                 # Dataset conversion and deterministic sampling
|   |-- evaluation/           # Final all-model and single-model evaluation
|   |-- models/               # Exported-model verification
|   `-- visualization/        # Report figure generation
|-- config/                   # Conda and pip environments
|-- datasets/
|   |-- fixed/                # Data shared by training runs 1--3
|   `-- repeats/              # Repeat-specific data for runs 4--10
|-- docs/                     # Dataset, model, and upload documentation
|-- models/                   # Ten conditions, run_01 through run_10
|-- report/                   # Final PDF and LaTeX source
`-- results/                  # Consolidated final CSV results
```

Only the final ten-repeat analysis is retained. Earlier pilot result folders
are intentionally excluded from this upload package.

## Environment

The evaluation was verified with Python 3.10, TensorFlow 2.15.0, NumPy 1.26.4,
and OpenCV 4.11.0.

Create the Conda environment:

```powershell
conda env create -f .\config\environment.yml
conda activate pa4
```

Alternatively, create a virtual environment and install the pinned packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\config\requirements.txt
```

## Reproduce the final evaluation

Run the complete 100-model evaluation from the repository root:

```powershell
python .\code\evaluation\evaluate_all_models.py
```

This recreates the seven result CSV files under `results/`. It evaluates ten
conditions, ten model-training runs per condition, and ten test sets per model,
for 1,000 evaluation summaries and 150,000 image predictions.

For a quick one-model check, edit the three selection values at the top of
`code/evaluation/evaluate_single_model.py`, then run:

```powershell
python .\code\evaluation\evaluate_single_model.py
```

The all-model evaluator also accepts optional selections. For example:

```powershell
python .\code\evaluation\evaluate_all_models.py `
  --models model_clean --repeats 1 --test-sets 1 `
  --output-dir results_smoke
```

## Recreate the report figure

```powershell
python .\code\visualization\make_result_figure.py
```

The script reads the consolidated final results and updates the PDF and PNG
figures under `report/figures/`.

## Dataset reconstruction

The processed images needed for evaluation are included. To rebuild every
processed dataset from the original labeled STL-10 Parquet files, place the raw
files at the paths documented in `docs/DATASET.md`, then run:

```powershell
python .\code\data\convert_stl10_parquet_to_png.py
python .\code\data\build_fixed_datasets.py
python .\code\data\build_fixed_test_sets.py
python .\code\data\build_randomized_repeats.py
```

Runs 1--3 share the fixed training pools and fixed test suite. Runs 4--10 use
deterministically sampled training pools and repeat-specific test suites.
Conditions within the same run share the same target pools and test images.

## Documentation

- `docs/DATASET.md`: exact source revision, checksums, classes, manifests, and
  reconstruction procedure.
- `docs/MODEL_SETTINGS.md`: Teachable Machine settings and model naming.
- `docs/UPLOAD.md`: GitHub upload and clone-verification steps.
- `code/README.md`, `datasets/README.md`, `models/README.md`, and
  `results/README.md`: folder-level guides.

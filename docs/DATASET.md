# Dataset documentation

## Exact source

The images are from STL-10, introduced by Adam Coates, Andrew Y. Ng, and
Honglak Lee.

- Official dataset page: <https://cs.stanford.edu/~acoates/stl10/>
- Download mirror: <https://huggingface.co/datasets/summerhot12138/stl10>
- Mirror revision: `4a63e09954d508560cdec52448db128cffeb4d76`
- Download date: 2026-11-14 (Asia/Seoul)

Only the labeled train and test splits were used. The unlabeled split was not
needed for this experiment.

| Raw file | Rows | Size (bytes) | SHA-256 |
|---|---:|---:|---|
| `train.parquet` | 5,000 | 88,820,161 | `DABC9160E9FB45953DAF9033A8D91F195F4F927750E6987EF0902A758889F483` |
| `test.parquet` | 8,000 | 141,816,319 | `118EA1BACEB999709915D730562F71D91AC2CC621C14D9106300F24D5A9C5AA7` |

## Classes and sizes

- Target classes: `airplane`, `monkey`, `truck`
- Unseen contamination classes: `dog`, `car`, `ship`
- Training images per assigned target label: 100
- Test images per class per test set: 50
- Test sets per model: 10
- Training runs per condition: 10

The tested contamination rates are 10%, 20%, and 30%. Contamination replaces
clean images and never changes the assigned class size.

## Conditions

- `model_clean`: all labels are correct.
- `model_mixed_pXX`: contaminated target images are split between the other two
  target labels.
- `model_systematic_pXX`: monkey is assigned to airplane, truck to monkey, and
  airplane to truck.
- `model_unseen_pXX`: dog, car, and ship images replace target-class images and
  are balanced across assigned labels.

`XX` is 10, 20, or 30.

## Final folder layout

```text
datasets/
|-- fixed/
|   |-- model_clean/ ... model_unseen_p30/
|   |-- test_sets/test_01/ ... test_10/
|   |-- experiment_manifest.csv
|   `-- experiment_summary.csv
`-- repeats/
    |-- repeat_04/ ... repeat_10/
    |-- train_manifest.csv
    `-- test_manifest.csv
```

Runs 1--3 use the same fixed training images and fixed test suite. Runs 4--10
use separate deterministic samples. Test sets are mutually disjoint within a
run; overlap between different runs is allowed. Every condition within the same
run uses the same target pools and test images.

The fixed selection seed is `20261114`. The repeat-specific seed base is
`20261115`.

## Manifests and leakage control

- `datasets/fixed/experiment_manifest.csv` records each fixed condition image.
- `datasets/fixed/test_sets/manifest.csv` records the fixed evaluation suite.
- `datasets/repeats/train_manifest.csv` records training selections for runs
  4--10.
- `datasets/repeats/test_manifest.csv` records test selections for runs 4--10.

Training images come from the STL-10 train split. Evaluation images come from
the STL-10 test split. The included manifests identify the source class,
assigned class, contamination type, and final output path.

## Rebuild from raw files

Place the checksum-verified files at:

```text
datasets/source/STL10/raw/huggingface_parquet/train.parquet
datasets/source/STL10/raw/huggingface_parquet/test.parquet
```

Then run these commands from the repository root:

```powershell
python .\code\data\convert_stl10_parquet_to_png.py
python .\code\data\build_fixed_datasets.py
python .\code\data\build_fixed_test_sets.py
python .\code\data\build_randomized_repeats.py
```

The first command extracts the embedded lossless 96 x 96 RGB PNG images without
re-encoding. The remaining commands recreate all fixed and repeat-specific
experimental folders using repository-relative paths.

# Dataset

## Source files

The images come from STL-10, introduced by Adam Coates, Andrew Y. Ng, and
Honglak Lee. This experiment uses only the labeled train and test splits; the
unlabeled split is not needed.

- Official page: <https://cs.stanford.edu/~acoates/stl10/>
- Parquet mirror: <https://huggingface.co/datasets/summerhot12138/stl10>
- Mirror revision: 4a63e09954d508560cdec52448db128cffeb4d76
- Downloaded: 2026-11-14 (Asia/Seoul)

| Raw file | Rows | Size (bytes) | SHA-256 |
|---|---:|---:|---|
| train.parquet | 5,000 | 88,820,161 | DABC9160E9FB45953DAF9033A8D91F195F4F927750E6987EF0902A758889F483 |
| test.parquet | 8,000 | 141,816,319 | 118EA1BACEB999709915D730562F71D91AC2CC621C14D9106300F24D5A9C5AA7 |

## Experimental classes and sample sizes

- Target classes: airplane, monkey, truck
- Unseen contamination classes: dog, car, ship
- Assigned training images per target label: 100
- Test images per class and test set: 50
- Test sets per model: 10
- Training runs per condition: 10

Contamination rates are 10%, 20%, and 30%. A contaminated image replaces a
clean image, so every assigned class remains at 100 images.

## Training conditions

- **model_clean:** all training images have the correct label.
- **model_mixed_pXX:** contaminated target-class images are divided between the
  other two target labels.
- **model_systematic_pXX:** monkey images are assigned to airplane, truck images
  to monkey, and airplane images to truck.
- **model_unseen_pXX:** balanced dog, car, and ship images replace target-class
  images.

In these names, XX is 10, 20, or 30.

## Processed data layout

~~~text
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
~~~

Runs 1--3 use the fixed training data and fixed test suite. Runs 4--10 use
separate deterministic samples. Test sets do not overlap within a run, although
images may reappear in different runs. Every condition in the same run uses the
same target-class pools and test images.

The fixed-data seed is 20261114; the base seed for runs 4--10 is 20261115.

## Manifests and train-test separation

- **datasets/fixed/experiment_manifest.csv** lists every image in the fixed
  training conditions.
- **datasets/fixed/test_sets/manifest.csv** lists the fixed evaluation images.
- **datasets/repeats/train_manifest.csv** lists the training selections for runs
  4--10.
- **datasets/repeats/test_manifest.csv** lists the test selections for runs
  4--10.

Training images always come from the STL-10 train split, while evaluation
images come from its test split. The manifests record the source class,
assigned class, contamination type, and output path for each copied image.

## Rebuild the processed data

Place the checksum-verified Parquet files at:

~~~text
datasets/source/STL10/raw/huggingface_parquet/train.parquet
datasets/source/STL10/raw/huggingface_parquet/test.parquet
~~~

Then run the following commands from the repository root:

~~~powershell
python .\code\data\convert_stl10_parquet_to_png.py
python .\code\data\build_fixed_datasets.py
python .\code\data\build_fixed_test_sets.py
python .\code\data\build_randomized_repeats.py
~~~

The converter extracts the embedded lossless 96 × 96 RGB PNG data without
re-encoding it. The other scripts rebuild the fixed and run-specific
experimental folders using repository-relative paths.

# PA4 Experiment Datasets

## Folder structure

Each model dataset follows this structure:

```text
data/
  <model_name>/
    airplane/
      train/1.png ... 100.png
      test/1.png ... 50.png
    monkey/
      train/1.png ... 100.png
      test/1.png ... 50.png
    truck/
      train/1.png ... 100.png
      test/1.png ... 50.png
```

All selections were made with the fixed random seed `20261114`. The same 50
clean test images per target class are used for every model. Repeated training
runs should reuse the same model dataset folder so that only training
stochasticity changes.

## Model datasets

| Model | Contamination type | Rate per assigned class |
|---|---|---:|
| `model_clean` | None | 0% |
| `model_mixed_p10` | Mixed cross-class label noise | 10% |
| `model_mixed_p20` | Mixed cross-class label noise | 20% |
| `model_mixed_p30` | Mixed cross-class label noise | 30% |
| `model_systematic_p10` | Systematic single-source label noise | 10% |
| `model_systematic_p20` | Systematic single-source label noise | 20% |
| `model_systematic_p30` | Systematic single-source label noise | 30% |
| `model_unseen_p10` | Unseen-class contamination | 10% |
| `model_unseen_p20` | Unseen-class contamination | 20% |
| `model_unseen_p30` | Unseen-class contamination | 30% |

## Contamination rules

### Mixed cross-class noise

For each source class, the contaminated images are divided equally between the
other two assigned classes. For example, at 20%, ten true monkey images and ten
true truck images are assigned the airplane label.

### Systematic single-source noise

The error directions are fixed:

```text
monkey -> airplane
truck -> monkey
airplane -> truck
```

### Unseen-class contamination

The unseen pool combines `dog`, `car`, and `ship`. Images are unique within each
model dataset. At 30%, every assigned class contains ten dog, ten car, and ten
ship images. The 10% and 20% datasets use balanced nested subsets of the same
30% pool.

## Reproducibility files

- `experiment_manifest.csv`: source and assigned class for every copied image
- `experiment_summary.csv`: clean/contaminated counts for every model and class
- `../scripts/build_experiment_datasets.py`: deterministic generation script
- `STL10/conversion_manifest.csv`: mapping from Parquet rows to extracted PNGs

The original STL-10 Parquet files and the complete extracted train/test PNG
sets remain under `data/STL10/`.

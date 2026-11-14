# Fixed training data and test sets

Runs 1--3 use the data in this directory. The random seed is 20261114, and all
ten training conditions use the same target-class source pools.

## Condition folders

Each condition has 100 assigned training images and 50 companion test images
per target class:

~~~text
datasets/fixed/<condition>/
|-- airplane/
|   |-- train/1.png ... 100.png
|   `-- test/1.png ... 50.png
|-- monkey/
|   |-- train/1.png ... 100.png
|   `-- test/1.png ... 50.png
`-- truck/
    |-- train/1.png ... 100.png
    `-- test/1.png ... 50.png
~~~

The final repeated evaluation uses the ten disjoint suites under test_sets/
rather than the companion test directories inside each condition.

| Condition | Contamination | Rate per assigned class |
|---|---|---:|
| **model_clean** | None | 0% |
| **model_mixed_p10** | Mixed target-class label noise | 10% |
| **model_mixed_p20** | Mixed target-class label noise | 20% |
| **model_mixed_p30** | Mixed target-class label noise | 30% |
| **model_systematic_p10** | Systematic target-class label noise | 10% |
| **model_systematic_p20** | Systematic target-class label noise | 20% |
| **model_systematic_p30** | Systematic target-class label noise | 30% |
| **model_unseen_p10** | Unseen-class contamination | 10% |
| **model_unseen_p20** | Unseen-class contamination | 20% |
| **model_unseen_p30** | Unseen-class contamination | 30% |

## How contamination is assigned

For mixed target-class noise, contaminated images come equally from the other
two target classes. At 20%, for example, the 100 images assigned to airplane
include ten true monkey images and ten true truck images.

Systematic noise follows one fixed cycle:

~~~text
monkey -> airplane
truck -> monkey
airplane -> truck
~~~

Unseen-class contamination uses dog, car, and ship images. At 30%, each
assigned target class contains ten images from each unseen class. The 10% and
20% conditions use balanced, nested subsets of the same 30% pool.

## Records and reconstruction

- **experiment_manifest.csv** records the source class and assigned class of
  every copied condition image.
- **experiment_summary.csv** gives the clean and contaminated counts for each
  condition and assigned class.
- **test_sets/manifest.csv** records every image in the ten fixed evaluation
  sets.

Rebuild this directory from the extracted STL-10 source pool with:

~~~powershell
python .\code\data\build_fixed_datasets.py
python .\code\data\build_fixed_test_sets.py
~~~

The source-file locations and the preceding Parquet conversion step are
documented in docs/DATASET.md.

# GitHub copy and verification

The submitted repository is available at
[github.com/jsport2/iap_PA4](https://github.com/jsport2/iap_PA4). It contains
37,750 tracked files and is about 0.84 GiB, so cloning with Git is more reliable
than downloading or uploading the files individually in a browser.

## Clone the repository

```powershell
git clone https://github.com/jsport2/iap_PA4.git pa4-check
cd pa4-check
```

No individual file exceeds GitHub's 100 MB limit. The largest file is
`results/all_predictions.csv`, which is about 22.9 MiB.

## Verify the cloned copy

Create the environment, then run one clean model against one test set:

```powershell
conda env create -f .\config\environment.yml
conda activate pa4
python .\code\evaluation\evaluate_all_models.py `
  --models model_clean --repeats 1 --test-sets 1 `
  --output-dir results_smoke
```

A successful smoke test reports 150 evaluated images and creates seven CSV
files in `results_smoke/`.

## Push later changes

From an existing local clone:

```powershell
git status
git add --all
git commit -m "Describe the change"
git push origin main
```

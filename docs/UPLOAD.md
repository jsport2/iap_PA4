# GitHub upload and verification

Use Git from the repository root. The final package contains 37,750 files and
is approximately 0.84 GiB, so browser upload is not recommended.

```powershell
git init
git add .
git commit -m "Add PA4 final experiment"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the placeholder repository URL before pushing. The largest included
file is `results/all_predictions.csv` at approximately 22.9 MiB; no file
exceeds GitHub's 100 MB per-file limit.

After upload, verify reproducibility from a separate directory:

```powershell
git clone https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git pa4-check
cd pa4-check
conda env create -f .\config\environment.yml
conda activate pa4
python .\code\evaluation\evaluate_all_models.py `
  --models model_clean --repeats 1 --test-sets 1 `
  --output-dir results_smoke
```

The smoke test should create seven CSV files under `results_smoke/` and report
150 evaluated images.

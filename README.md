# Ketika Machine Learning Gagal: Studi Kasus Dataset Tanpa Signal

This repository demonstrates a reproducible case study showing why machine learning fails when the dataset contains no predictive signal. The dataset contains only categorical features (`Brand`, `Category`, `Color`, `Size`, `Material`) and the target `Price` (numerical).

## Key findings

- Simple baseline (mean) often outperforms trained models when no signal exists.
- Models can produce negative R² indicating worse-than-mean predictions.
- Features in this dataset are not predictive of `Price`.

## Project structure

- `data/`
  - `raw/` (place raw CSVs here)
  - `processed/` (generated processed datasets)
- `notebooks/`
  - `01_eda.ipynb`
  - `02_baseline_model.ipynb`
  - `03_modeling_lasso_ngboost.ipynb`
  - `04_feature_engineering.ipynb`
  - `05_remodeling.ipynb`
  - `06_final_diagnosis.ipynb`
- `src/`
  - `data_preprocessing.py`
  - `feature_engineering.py`
  - `modeling.py`
  - `evaluation.py`
  - `utils.py`
- `results/`
  - `figures/`
  - `metrics/`
- `reports/`
  - `summary.md`

## How to run

1. Create a virtual environment and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place your raw CSV at `data/raw/raw_clothing_prices.csv`.

3. Open and run the notebooks in `notebooks/` in order.

## Key takeaway

"Machine learning cannot learn patterns that do not exist in data"

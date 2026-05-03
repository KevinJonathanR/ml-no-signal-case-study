"""End-to-end runner: load data, split, train baseline/Lasso/NGBoost, evaluate, save metrics and plots."""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data
from feature_engineering import frequency_encode, ordinal_encode_size, add_interaction_features
from modeling import build_baseline_pipeline, build_lasso_pipeline, build_ngboost_pipeline
from evaluation import evaluate_regression
from visualization import save_price_distribution, save_true_vs_pred, save_residuals


def train_and_evaluate(csv_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(str(csv_path))

    # Save distribution
    save_price_distribution(df, out_dir / 'figures' / 'price_distribution.png')

    # Basic split
    X = df.drop(columns=['Price'])
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = [c for c in X_train.columns if X_train[c].dtype.name == 'category' or X_train[c].dtype == object]

    results = {}

    # Baseline
    baseline = build_baseline_pipeline()
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    results['baseline'] = evaluate_regression(y_test, y_pred_baseline)
    save_true_vs_pred(y_test, y_pred_baseline, out_dir / 'figures' / 'baseline_true_vs_pred.png', 'Baseline True vs Pred')
    save_residuals(y_test, y_pred_baseline, out_dir / 'figures' / 'baseline_residuals.png')

    # Lasso
    lasso = build_lasso_pipeline(categorical_features)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    results['lasso'] = evaluate_regression(y_test, y_pred_lasso)
    save_true_vs_pred(y_test, y_pred_lasso, out_dir / 'figures' / 'lasso_true_vs_pred.png', 'Lasso True vs Pred')
    save_residuals(y_test, y_pred_lasso, out_dir / 'figures' / 'lasso_residuals.png')

    # NGBoost
    ngb = build_ngboost_pipeline(categorical_features)
    ngb.fit(X_train, y_train)
    y_pred_ngb = ngb.predict(X_test)
    results['ngboost'] = evaluate_regression(y_test, y_pred_ngb)
    save_true_vs_pred(y_test, y_pred_ngb, out_dir / 'figures' / 'ngboost_true_vs_pred.png', 'NGBoost True vs Pred')
    save_residuals(y_test, y_pred_ngb, out_dir / 'figures' / 'ngboost_residuals.png')

    # Save metrics
    (out_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics' / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('Saved figures to', out_dir / 'figures')
    print('Saved metrics to', out_dir / 'metrics' / 'metrics.json')
    return results


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    csv = base / 'data' / 'raw' / 'raw_clothing_prices.csv'
    out = base / 'results'
    train_and_evaluate(csv, out)

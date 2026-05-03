"""Simple plotting helpers: save distribution, residuals, and true-vs-pred plots."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def save_price_distribution(df: pd.DataFrame, out: Path):
    plt.figure(figsize=(8,4))
    sns.histplot(df['Price'], kde=True, bins=40)
    plt.title('Price distribution')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches='tight')
    plt.close()


def save_true_vs_pred(y_true, y_pred, out: Path, title: str = 'True vs Pred'):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], color='red', linestyle='--')
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title(title)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches='tight')
    plt.close()


def save_residuals(y_true, y_pred, out: Path):
    res = y_true - y_pred
    plt.figure(figsize=(8,4))
    sns.histplot(res, kde=True, bins=40)
    plt.title('Residuals distribution')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches='tight')
    plt.close()

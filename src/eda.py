"""Analyse exploratoire : distribution des classes, protocoles, corrélations.

    python -m src.eda
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import NUM_COLS, RESULTS_DIR, TRAIN_FILE
from .data_loader import class_distribution, load_nsl_kdd


def run(train_path=TRAIN_FILE, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    df = load_nsl_kdd(train_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    print("Dimensions :", df.shape)
    dist = class_distribution(df, "cls")
    print(dist)
    dist.to_csv(results_dir / "eda_distribution_classes.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df["cls"].value_counts().plot.bar(ax=axes[0], title="Distribution des classes")
    df.groupby(["protocol_type", "cls"]).size().unstack(fill_value=0).plot.bar(
        stacked=True, ax=axes[1], title="Protocole par classe")
    fig.tight_layout()
    fig.savefig(results_dir / "eda_classes_protocoles.png", dpi=150)
    plt.close(fig)

    corr = df[NUM_COLS].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title("Corrélation des attributs numériques")
    fig.tight_layout()
    fig.savefig(results_dir / "eda_correlation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()

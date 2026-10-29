"""Métriques d'évaluation adaptées à la détection d'intrusion."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def false_alarm_rate(y_true, y_pred, normal_label):
    """FAR = trafic normal classé comme attaque / trafic normal total."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    normal = y_true == normal_label
    if normal.sum() == 0:
        return float("nan")
    return float((y_pred[normal] != normal_label).mean())


def detection_rate(y_true, y_pred, normal_label):
    """DR = attaques classées comme non normales / attaques totales."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    attack = y_true != normal_label
    if attack.sum() == 0:
        return float("nan")
    return float((y_pred[attack] != normal_label).mean())


def compute_metrics(y_true, y_pred, normal_label):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "detection_rate": detection_rate(y_true, y_pred, normal_label),
        "false_alarm_rate": false_alarm_rate(y_true, y_pred, normal_label),
    }


def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), [str(l) for l in labels], rotation=45)
    ax.set_yticks(range(len(labels)), [str(l) for l in labels])
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

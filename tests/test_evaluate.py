"""Tests unitaires des metriques d'evaluation (src/evaluate.py)."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import (compute_metrics, detection_rate,  # noqa: E402
                          false_alarm_rate, plot_confusion_matrix)

NORMAL = "normal"


def test_classifieur_parfait():
    y = ["normal", "DoS", "normal", "Probe"]
    assert detection_rate(y, y, NORMAL) == 1.0
    assert false_alarm_rate(y, y, NORMAL) == 0.0


def test_tout_predit_comme_attaque():
    y_true = ["normal", "normal", "DoS", "Probe"]
    y_pred = ["DoS", "DoS", "DoS", "Probe"]
    assert false_alarm_rate(y_true, y_pred, NORMAL) == 1.0
    assert detection_rate(y_true, y_pred, NORMAL) == 1.0


def test_toutes_les_attaques_manquees():
    y_true = ["normal", "DoS", "R2L", "U2R"]
    y_pred = ["normal", "normal", "normal", "normal"]
    assert detection_rate(y_true, y_pred, NORMAL) == 0.0
    assert false_alarm_rate(y_true, y_pred, NORMAL) == 0.0


def test_confusion_entre_attaques_compte_comme_detection():
    # Une attaque classee dans une AUTRE classe d'attaque est tout de meme detectee.
    y_true = ["DoS", "Probe"]
    y_pred = ["Probe", "DoS"]
    assert detection_rate(y_true, y_pred, NORMAL) == 1.0


def test_nan_sans_trafic_normal():
    assert math.isnan(false_alarm_rate(["DoS", "Probe"], ["DoS", "Probe"], NORMAL))


def test_nan_sans_attaque():
    assert math.isnan(detection_rate(["normal", "normal"], ["normal", "normal"], NORMAL))


def test_fonctionne_avec_etiquettes_binaires_numeriques():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    assert false_alarm_rate(y_true, y_pred, 0) == 0.5
    assert detection_rate(y_true, y_pred, 0) == pytest.approx(2 / 3)


def test_compute_metrics_cles_et_valeurs():
    y_true = ["normal", "normal", "DoS", "DoS"]
    y_pred = ["normal", "normal", "DoS", "normal"]
    m = compute_metrics(y_true, y_pred, NORMAL)
    assert set(m) == {"accuracy", "precision_macro", "recall_macro", "f1_macro",
                      "f1_weighted", "detection_rate", "false_alarm_rate"}
    assert m["accuracy"] == 0.75
    assert m["detection_rate"] == 0.5
    assert m["false_alarm_rate"] == 0.0
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_plot_confusion_matrix_cree_le_fichier(tmp_path):
    out = tmp_path / "cm.png"
    plot_confusion_matrix(["normal", "DoS", "normal"], ["normal", "DoS", "DoS"],
                          ["normal", "DoS"], "Test", out)
    assert out.exists() and out.stat().st_size > 0

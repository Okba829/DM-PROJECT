"""Tests de bout en bout sur des données synthétiques (rapides)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ATTACK_TO_CLASS  # noqa: E402
from src.data_loader import load_nsl_kdd  # noqa: E402
from src.evaluate import detection_rate, false_alarm_rate  # noqa: E402
from src.train import run_experiment  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"),
                    "--out", str(d), "--n-train", "1500", "--n-test", "500"], check=True)
    return d / "KDDTrain+.txt", d / "KDDTest+.txt"


def test_taxonomy():
    assert ATTACK_TO_CLASS["neptune"] == "DoS"
    assert ATTACK_TO_CLASS["normal"] == "normal"


def test_loader(synth):
    df = load_nsl_kdd(synth[0])
    assert {"cls", "bin"} <= set(df.columns)
    assert set(df["bin"].unique()) <= {0, 1}


def test_metrics():
    y_true = ["normal", "normal", "DoS", "DoS"]
    y_pred = ["normal", "DoS", "DoS", "normal"]
    assert false_alarm_rate(y_true, y_pred, "normal") == 0.5
    assert detection_rate(y_true, y_pred, "normal") == 0.5


@pytest.mark.parametrize("target", ["cls", "bin"])
@pytest.mark.parametrize("smote", [False, True])
def test_training(synth, tmp_path, target, smote):
    df = run_experiment(synth[0], synth[1], target=target,
                        model_names=["DecisionTree", "RandomForest", "NaiveBayes"],
                        select_k=20, use_smote=smote, results_dir=tmp_path / "r",
                        models_dir=tmp_path / "m")
    assert df["f1_macro"].max() > 0.5

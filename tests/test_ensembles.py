"""Tests du benchmark, de l'inférence et de l'application Streamlit (données synthétiques)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import benchmark, inference  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def synth_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench")
    cols_df_train = make_synthetic(1500, seed=0)
    cols_df_test = make_synthetic(500, seed=1, shift=0.3)
    from src.config import COLS
    for name, df in (("KDDTrain+.txt", cols_df_train), ("KDDTest+.txt", cols_df_test)):
        out = df[[c for c in COLS if c != "difficulty"]].copy()
        out["difficulty"] = 10
        out[COLS].to_csv(d / name, header=False, index=False)
    return d / "KDDTrain+.txt", d / "KDDTest+.txt"


@pytest.mark.parametrize("target", ["bin", "cls"])
def test_benchmark(synth_files, tmp_path, target):
    df = benchmark.run(synth_files[0], synth_files[1], target=target,
                       model_names=["DecisionTree", "AdaBoost", "XGBoost", "LightGBM"],
                       n_latency=20, results_dir=tmp_path)
    assert {"f1_macro", "latency_ms_single", "model_size_kb", "peak_mem_mb",
            "compatible_passerelle"} <= set(df.columns)
    assert df["f1_macro"].max() > 0.5
    assert (tmp_path / f"benchmark_{target}.csv").exists()
    assert (tmp_path / f"benchmark_{target}.png").exists()


def test_voting(synth_files, tmp_path):
    df = benchmark.run(synth_files[0], synth_files[1], target="bin",
                       model_names=["Voting"], n_latency=10, results_dir=tmp_path)
    assert df.loc["Voting", "f1_macro"] > 0.5


def test_inference_demo(tmp_path, monkeypatch):
    monkeypatch.setenv("IDS_DEMO", "1")
    path = tmp_path / "m.joblib"
    model, meta = inference.load_or_train(path)
    assert path.exists() and "normal" in meta["classes"]
    res = inference.predict_one(model, meta, {"protocol_type": "tcp"})
    assert res["prediction"] in meta["classes"]
    assert 0.0 <= res["attack_score"] <= 1.0
    model2, _ = inference.load_or_train(path)      # rechargement depuis le disque
    assert model2 is not None


def test_streamlit_app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("IDS_DEMO", "1")
    monkeypatch.setenv("IDS_MODEL_PATH", str(tmp_path / "app.joblib"))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    at.button(key="ex_DoS").click().run()
    assert not at.exception, at.exception

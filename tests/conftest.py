"""Fixtures partagees par les tests : petits fichiers NSL-KDD SYNTHETIQUES (jamais de vraies donnees)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_data import make  # noqa: E402


@pytest.fixture(scope="session")
def donnees_nsl_kdd_synthetiques(tmp_path_factory):
    """Retourne (chemin_train, chemin_test) : 1500 et 600 connexions au format NSL-KDD."""
    dossier = tmp_path_factory.mktemp("nsl_kdd_synthetique")
    rng = np.random.default_rng(0)
    train, test = dossier / "KDDTrain+.txt", dossier / "KDDTest+.txt"
    make(1500, rng).to_csv(train, header=False, index=False)
    make(600, rng, shift_unseen=True).to_csv(test, header=False, index=False)
    return train, test

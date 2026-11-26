"""Tests unitaires du chargement NSL-KDD et de la taxonomie des attaques."""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ATTACK_CATEGORIES, ATTACK_TO_CLASS, COLS, NUM_COLS  # noqa: E402
from src.data_loader import class_distribution, load_nsl_kdd  # noqa: E402


def _ligne(label, difficulty=15, **valeurs):
    """Construit une ligne au format NSL-KDD (43 colonnes, sans en-tete)."""
    base = {c: 0 for c in NUM_COLS}
    base.update(protocol_type="tcp", service="http", flag="SF",
                label=label, difficulty=difficulty)
    base.update(valeurs)
    return ",".join(str(base[c]) for c in COLS)


def _ecrire(tmp_path, lignes):
    chemin = tmp_path / "echantillon.txt"
    chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return chemin


def test_colonne_difficulty_supprimee(tmp_path):
    df = load_nsl_kdd(_ecrire(tmp_path, [_ligne("normal")]))
    assert "difficulty" not in df.columns
    assert {"label", "cls", "bin"} <= set(df.columns)


def test_cibles_multiclasse_et_binaire(tmp_path):
    labels = ["normal", "neptune", "ipsweep", "guess_passwd", "buffer_overflow"]
    df = load_nsl_kdd(_ecrire(tmp_path, [_ligne(l, service=f"s{i}")
                                         for i, l in enumerate(labels)]))
    assert list(df["cls"]) == ["normal", "DoS", "Probe", "R2L", "U2R"]
    assert list(df["bin"]) == [0, 1, 1, 1, 1]


def test_attaque_inconnue_classee_other_et_comptee_comme_attaque(tmp_path):
    df = load_nsl_kdd(_ecrire(tmp_path, [_ligne("attaque_jamais_vue")]))
    assert df.loc[0, "cls"] == "Other"
    assert df.loc[0, "bin"] == 1


def test_doublons_supprimes_par_defaut(tmp_path):
    # Deux lignes identiques, sauf le niveau de difficulte (colonne ecartee) : doublons.
    chemin = _ecrire(tmp_path, [_ligne("normal", difficulty=10),
                                _ligne("normal", difficulty=20),
                                _ligne("neptune")])
    assert len(load_nsl_kdd(chemin)) == 2
    assert len(load_nsl_kdd(chemin, drop_duplicates=False)) == 3


def test_class_distribution_effectifs_et_pourcentages(tmp_path):
    lignes = ([_ligne("normal", service=f"a{i}") for i in range(3)]
              + [_ligne("neptune", service="b")])
    dist = class_distribution(load_nsl_kdd(_ecrire(tmp_path, lignes)))
    assert dist.loc["normal", "effectif"] == 3
    assert dist.loc["DoS", "effectif"] == 1
    assert dist.loc["normal", "pourcentage"] == 75.0
    assert abs(dist["pourcentage"].sum() - 100.0) < 0.1


def test_taxonomie_chaque_attaque_dans_une_seule_famille():
    noms = [a for liste in ATTACK_CATEGORIES.values() for a in liste]
    doublons = [a for a, n in Counter(noms).items() if n > 1]
    assert doublons == []
    assert set(ATTACK_TO_CLASS) == set(noms) | {"normal"}


def test_taxonomie_quatre_familles():
    assert set(ATTACK_CATEGORIES) == {"DoS", "Probe", "R2L", "U2R"}

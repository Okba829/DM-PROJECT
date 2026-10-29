"""Chargement et étiquetage des fichiers NSL-KDD."""
import pandas as pd

from .config import ATTACK_TO_CLASS, COLS


def load_nsl_kdd(path, drop_duplicates=True):
    """Lit un fichier NSL-KDD (sans en-tête) et ajoute deux cibles :
    - ``cls`` : normal / DoS / Probe / R2L / U2R (multiclasse)
    - ``bin`` : 0 = normal, 1 = attaque (binaire)
    """
    df = pd.read_csv(path, names=COLS, header=None)
    df = df.drop(columns="difficulty")
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    df["cls"] = df["label"].map(ATTACK_TO_CLASS).fillna("Other")
    df["bin"] = (df["cls"] != "normal").astype(int)
    return df


def class_distribution(df, target="cls"):
    """Effectifs et pourcentages par classe."""
    counts = df[target].value_counts()
    return pd.DataFrame({"effectif": counts, "pourcentage": (100 * counts / len(df)).round(2)})

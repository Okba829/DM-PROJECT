"""Pré-traitement : encodage one-hot, normalisation, sélection d'attributs."""
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CAT_COLS, NUM_COLS


def build_preprocessor():
    """One-hot sur les variables catégorielles, standardisation sur les numériques.
    Les paramètres sont appris sur le train uniquement (pas de fuite de données)."""
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ],
        sparse_threshold=0,
    )


def build_selector(k):
    """Sélection des k meilleurs attributs (test F de l'ANOVA). None = pas de sélection."""
    if not k:
        return None
    return SelectKBest(score_func=f_classif, k=k)

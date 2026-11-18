"""Méthodes d'ensemble : Random Forest, AdaBoost, XGBoost, LightGBM et VotingClassifier.

Le VotingClassifier combine un modèle LINÉAIRE (régression logistique) et des modèles
ARBORESCENTS (Random Forest, LightGBM) par vote pondéré sur les probabilités (soft voting).
"""
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def _lgbm(n_estimators=200):
    return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.1, num_leaves=63,
                          class_weight="balanced", random_state=RANDOM_STATE,
                          n_jobs=-1, verbose=-1)


def get_ensemble_classifiers():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                             random_state=RANDOM_STATE),
            n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
            random_state=RANDOM_STATE),
        "LightGBM": _lgbm(),
        "Voting": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                              n_jobs=-1, random_state=RANDOM_STATE)),
                ("lgbm", _lgbm(150)),
            ],
            voting="soft"),
    }

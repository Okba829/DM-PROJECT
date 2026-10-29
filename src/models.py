"""Définition des classifieurs et construction des pipelines."""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor, build_selector


def get_classifiers():
    return {
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=15, class_weight="balanced", random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "LinearSVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    }


def build_pipeline(clf, select_k=None, use_smote=False):
    """prétraitement -> [sélection] -> [SMOTE sur le train seulement] -> classifieur."""
    steps = [("prep", build_preprocessor())]
    selector = build_selector(select_k)
    if selector is not None:
        steps.append(("select", selector))
    if use_smote:
        steps.append(("smote", SMOTE(k_neighbors=3, random_state=RANDOM_STATE)))
        return ImbPipeline(steps + [("clf", clf)])
    return Pipeline(steps + [("clf", clf)])

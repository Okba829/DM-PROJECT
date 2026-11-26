"""Tests de la detection d'anomalies par Isolation Forest (src/anomaly.py)."""
from src import anomaly


def test_anomaly_metriques_et_fichiers(donnees_nsl_kdd_synthetiques, tmp_path):
    train, test = donnees_nsl_kdd_synthetiques
    m = anomaly.run(train, test, contamination=0.05, results_dir=tmp_path)
    assert {"accuracy", "f1_macro", "detection_rate", "false_alarm_rate", "auc"} <= set(m)
    assert 0.0 <= m["auc"] <= 1.0
    assert (tmp_path / "metrics_isolation_forest.csv").exists()
    assert (tmp_path / "cm_isolation_forest.png").exists()


def test_anomaly_auc_superieure_au_hasard(donnees_nsl_kdd_synthetiques, tmp_path):
    # Les attaques synthetiques sont decalees du trafic normal : le score doit les separer.
    train, test = donnees_nsl_kdd_synthetiques
    m = anomaly.run(train, test, results_dir=tmp_path)
    assert m["auc"] > 0.6


def test_anomaly_contamination_plus_haute_signale_davantage(donnees_nsl_kdd_synthetiques,
                                                            tmp_path):
    # Meme arbres (random_state fixe) : seul le seuil change, donc DR et FAR ne peuvent que monter.
    train, test = donnees_nsl_kdd_synthetiques
    bas = anomaly.run(train, test, contamination=0.02, results_dir=tmp_path / "bas")
    haut = anomaly.run(train, test, contamination=0.20, results_dir=tmp_path / "haut")
    assert haut["detection_rate"] >= bas["detection_rate"]
    assert haut["false_alarm_rate"] >= bas["false_alarm_rate"]
    assert haut["auc"] == bas["auc"]

"""Application Streamlit : test manuel de valeurs et simulation d'arrivée de trafic réseau
en temps réel avec détection d'attaques.

    streamlit run app/streamlit_app.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import inference as inf  # noqa: E402
from src.config import FEATURE_COLS, RESULTS_DIR  # noqa: E402

st.set_page_config(page_title="IDS – détection d'attaques réseau", page_icon="🛡️",
                   layout="wide")


@st.cache_resource(show_spinner="Chargement (ou entraînement) du modèle…")
def get_model():
    return inf.load_or_train()


@st.cache_data(show_spinner=False)
def get_stream_df():
    return inf.load_stream_source()


model, meta = get_model()
stream_df = get_stream_df()

# Attributs les plus parlants, exposés dans le formulaire de test manuel
KEY_NUM = {
    "duration": "Durée de la connexion (s)",
    "src_bytes": "Octets envoyés (source → destination)",
    "dst_bytes": "Octets reçus (destination → source)",
    "count": "Connexions vers le même hôte (2 dernières s)",
    "srv_count": "Connexions vers le même service (2 dernières s)",
    "dst_host_count": "Connexions vers l'hôte destination",
    "dst_host_srv_count": "Connexions vers le service de l'hôte destination",
    "num_failed_logins": "Échecs de connexion",
    "hot": "Indicateurs « hot »",
}
KEY_RATES = {
    "serror_rate": "Taux d'erreurs SYN",
    "rerror_rate": "Taux d'erreurs REJ",
    "same_srv_rate": "Taux de connexions au même service",
}
KEY_BIN = {"logged_in": "Session ouverte (logged_in)"}


def init_state():
    d = meta["defaults"]
    for c in KEY_NUM:
        st.session_state.setdefault(f"in_{c}", float(d[c]))
    for c in KEY_RATES:
        st.session_state.setdefault(f"in_{c}", float(min(max(d[c], 0.0), 1.0)))
    for c in KEY_BIN:
        st.session_state.setdefault(f"in_{c}", int(round(min(max(d[c], 0), 1))))
    for c in ("protocol_type", "service", "flag"):
        st.session_state.setdefault(f"in_{c}", d[c])


init_state()


def load_example(cls_name):
    sub = stream_df[stream_df["cls"] == cls_name]
    if sub.empty:
        return
    row = sub.sample(1, random_state=int(time.time()) % 100000).iloc[0]
    st.session_state["base"] = {c: row[c] for c in FEATURE_COLS}
    for c in KEY_NUM:
        st.session_state[f"in_{c}"] = float(row[c])
    for c in KEY_RATES:
        st.session_state[f"in_{c}"] = float(min(max(row[c], 0.0), 1.0))
    for c in KEY_BIN:
        st.session_state[f"in_{c}"] = int(row[c])
    for c in ("protocol_type", "service", "flag"):
        if row[c] in meta["choices"][c]:
            st.session_state[f"in_{c}"] = row[c]
    st.session_state["example_truth"] = row["cls"]


def show_result(res, truth=None):
    if res["is_attack"]:
        st.error(f"🚨 ALERTE – attaque détectée : **{res['prediction']}** "
                 f"(score d'attaque {res['attack_score']:.1%})")
    else:
        st.success(f"✅ Trafic normal (score d'attaque {res['attack_score']:.1%})")
    if truth:
        st.caption(f"Classe réelle de l'exemple chargé : **{truth}**")
    probs = pd.Series(res["probabilities"]).sort_values(ascending=False)
    st.bar_chart(probs.rename("probabilité"))
    st.caption(f"Latence de la prédiction : {res['latency_ms']:.2f} ms")


# ----------------------------------------------------------------- en-tête
st.title("🛡️ Détection d'attaques réseau en temps réel")
st.caption(f"Modèle : LightGBM · Données d'entraînement : {meta['source']} · "
           f"Classes : {', '.join(meta['classes'])}")
if "démo" in meta["source"]:
    st.warning("Mode démo : les fichiers NSL-KDD sont absents de `data/`, le modèle est "
               "entraîné sur des données synthétiques (voir `data/README.md`).")

tab_manual, tab_live, tab_bench = st.tabs(
    ["🧪 Test manuel", "📡 Simulation temps réel", "📊 Benchmark des modèles"])

# ------------------------------------------------------------ test manuel
with tab_manual:
    st.subheader("Saisir ou charger une connexion réseau")
    cols = st.columns(len(meta["classes"]))
    for col, cls_name in zip(cols, meta["classes"]):
        col.button(f"Exemple : {cls_name}", key=f"ex_{cls_name}",
                   on_click=load_example, args=(cls_name,), width="stretch")

    base = st.session_state.get("base", {})
    left, right = st.columns(2)
    values = dict(base)
    with left:
        for c in ("protocol_type", "service", "flag"):
            values[c] = st.selectbox(c, meta["choices"][c], key=f"in_{c}")
        for c, label in KEY_BIN.items():
            values[c] = st.selectbox(label, [0, 1], key=f"in_{c}")
        for c, label in KEY_RATES.items():
            values[c] = st.slider(label, 0.0, 1.0, step=0.01, key=f"in_{c}")
    with right:
        for c, label in KEY_NUM.items():
            values[c] = st.number_input(label, step=1.0, key=f"in_{c}")

    res = inf.predict_one(model, meta, values)
    st.divider()
    show_result(res, st.session_state.get("example_truth"))
    st.caption("Les autres attributs du formulaire utilisent la valeur médiane du jeu "
               "d'entraînement (ou ceux de l'exemple chargé).")

# --------------------------------------------------------- temps réel
with tab_live:
    st.subheader("Simulation d'arrivée de trafic")
    st.write("Les événements sont tirés au hasard dans le jeu de test et traités un par "
             "un, comme sur une passerelle qui reçoit des connexions.")
    c1, c2 = st.columns(2)
    n_events = c1.slider("Nombre d'événements", 20, 500, 100, 10)
    rate = c2.slider("Cadence (événements / seconde)", 1, 50, 10)
    if st.button("▶ Démarrer la simulation", type="primary"):
        sample = stream_df.sample(n_events, random_state=int(time.time()) % 100000)
        kpi = st.empty()
        chart = st.empty()
        table = st.empty()
        records = []
        for i, (_, row) in enumerate(sample.iterrows(), start=1):
            res = inf.predict_one(model, meta, row.to_dict())
            records.append({
                "#": i, "protocole": row["protocol_type"], "service": row["service"],
                "prédit": res["prediction"], "réel": row["cls"],
                "score attaque": round(res["attack_score"], 3),
                "latence (ms)": round(res["latency_ms"], 2),
                "statut": "🚨 ALERTE" if res["is_attack"] else "OK",
            })
            log = pd.DataFrame(records)
            alerts = int((log["statut"] != "OK").sum())
            true_att = int((log["réel"] != "normal").sum())
            detected = int(((log["réel"] != "normal") & (log["statut"] != "OK")).sum())
            false_al = int(((log["réel"] == "normal") & (log["statut"] != "OK")).sum())
            with kpi.container():
                k = st.columns(5)
                k[0].metric("Événements", i)
                k[1].metric("Alertes", alerts)
                k[2].metric("Attaques détectées", f"{detected}/{true_att}")
                k[3].metric("Fausses alarmes", false_al)
                k[4].metric("Latence moy. (ms)", f"{log['latence (ms)'].mean():.2f}")
            chart.line_chart(pd.DataFrame({
                "Alertes cumulées": (log["statut"] != "OK").cumsum(),
                "Trafic normal cumulé": (log["statut"] == "OK").cumsum()}))
            table.dataframe(log.tail(10).iloc[::-1], hide_index=True,
                            width="stretch")
            time.sleep(1 / rate)
        st.success("Simulation terminée.")

# --------------------------------------------------------------- benchmark
with tab_bench:
    st.subheader("Comparaison des algorithmes")
    found = False
    for target, title in (("bin", "Détection binaire (normal / attaque)"),
                          ("cls", "Classification à 5 classes")):
        csv = RESULTS_DIR / f"benchmark_{target}.csv"
        if csv.exists():
            found = True
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(csv), hide_index=True, width="stretch")
            png = RESULTS_DIR / f"benchmark_{target}.png"
            if png.exists():
                st.image(str(png))
    if not found:
        st.info("Aucun résultat : lancez `python -m src.benchmark --target bin` "
                "(puis `--target cls`).")

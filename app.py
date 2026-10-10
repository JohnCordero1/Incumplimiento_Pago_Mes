"""
app.py - Interfaz de Streamlit: alerta temprana de incumplimiento de pago.

Uso:  1) python train_model.py (generar models/credit_model.joblib)
      2) streamlit run app.py
"""
# ---------------------------------------------------------------------------
# LIBRERIAS
# ---------------------------------------------------------------------------
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

import project_info
from credit_utils import BILL_COLS, PAY_COLS, PAYAMT_COLS, RAW_INPUTS

BASE = Path(__file__).resolve().parent
EDU = {1: "Posgrado", 2: "Universidad", 3: "Secundaria", 4: "Otros"}
OK = [-1, 5000, 5000]  # estado de pago, facturado, pagado
PRESETS = {
    "Cliente nuevo (editable)": dict(limit=100_000, edu=2, age=35, rows=[[0, 20000, 2000]] * 6),
    "Pagador puntual": dict(limit=200_000, edu=2, age=40, rows=[OK] * 6),
    "Atrasado 3 meses": dict(limit=100_000, edu=2, age=40,
                             rows=[[3, 90000, 0], [3, 88000, 0], [2, 5000, 5000], [2, 5000, 5000], OK, OK]),
}
NAMES = {"LIMIT_BAL": "Límite de crédito", "AGE": "Edad", "util_1": "Uso de la línea, mes -1",
         "util_mean": "Uso promedio de la línea", "util_max": "Uso máximo de la línea",
         "pay_ratio_last": "Fracción pagada, último mes", "pay_ratio_mean": "Fracción pagada, promedio",
         "n_delays": "Meses con atraso", "max_delay": "Atraso máximo", "delay_trend": "Tendencia del atraso"}

st.set_page_config(page_title="Alerta temprana de incumplimiento", layout="wide")


@st.cache_resource
def load_scorer():
    path = BASE / "models" / "credit_model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def get_explainer(_model):
    try:
        return shap.TreeExplainer(_model)
    except Exception:  # modelo no basado en árboles
        return None


def nice(name):
    if name in NAMES:
        return NAMES[name]
    for pre, txt in [("PAY_AMT", "Pagado"), ("BILL_AMT", "Facturado"), ("PAY_", "Estado de pago")]:
        if name.startswith(pre):
            return f"{txt}, mes -{name[len(pre):]}"
    return "Educación: " + EDU[int(name[-1])] if name.startswith("EDUCATION_") else name


def contributions(scorer, df):
    """Aporte de cada variable a la PI de este cliente (SHAP). None si no hay explicador."""
    feats = scorer.pipeline[:-1]
    Xt = pd.DataFrame(feats.transform(df), columns=feats.named_steps["prep"].get_feature_names_out())
    mdl = scorer.pipeline.named_steps["model"]
    expl = get_explainer(mdl)
    if expl is None:
        return None
    sv = expl.shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
    base = float(np.ravel(expl.expected_value)[-1])
    prob = isinstance(mdl, RandomForestClassifier)   # RF: escala de probabilidad; boosting: log-odds
    return pd.Series(sv[0], index=Xt.columns), Xt.iloc[0], (base if prob else 1 / (1 + np.exp(-base))), prob


def plot_contrib(s, x, prob):
    top = s.reindex(s.abs().sort_values(ascending=False).head(8).index)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh([f"{nice(n)} = {x[n]:,.2f}".rstrip("0").rstrip(".") for n in top.index],
            top.values * (100 if prob else 1), color=["#E24B4A" if v > 0 else "#1D9E75" for v in top.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("puntos porcentuales de PI" if prob else "impacto (log-odds)")
    fig.tight_layout()
    return fig


def show_csv(path, title, index_name=None, note=None):
    if not path.exists():
        return
    st.markdown(f"**{title}**")
    df = pd.read_csv(path, index_col=0).reset_index() if index_name else pd.read_csv(path)
    if index_name:
        df = df.rename(columns={df.columns[0]: index_name})
    if "X" in df.columns:
        df["X"] = df["X"].astype(int).astype(str) + " : 1"
    st.dataframe(df.rename(columns=lambda c: c.replace("_", " ")).round(3), hide_index=True)
    if note:
        st.caption(note)


scorer = load_scorer()
if scorer is None:
    st.error("No se encontro `models/credit_model.joblib`. Ejecuta primero `python train_model.py`.")
    st.stop()
meta = scorer.meta
if "cost_miss" not in meta:
    st.error("Este modelo se generó con una versión anterior del proyecto. Vuelve a ejecutar `python train_model.py`.")
    st.stop()

st.title("Alerta temprana de incumplimiento de pago")
st.caption("Estima la probabilidad de que un cliente de tarjeta incumpla su pago el mes siguiente y sugiere una acción. "
           "El modelo se entrenó con datos de Taiwán (2005)")
if meta.get("cv", {}).get("fast"):
    st.warning("Modelo entrenado en modo rápido (`--fast`): sirve para probar la app, no para sacar conclusiones.")
if meta["versions"]["sklearn"] != sklearn.__version__:
    st.warning(f"El modelo se entrenó con scikit-learn {meta['versions']['sklearn']} y aquí hay {sklearn.__version__}. "
               "Reentrena con `python train_model.py` o instala la misma versión.")

tab1, tab2, tab3 = st.tabs(["Simulador de cliente", "Desempeño del modelo", "Sobre el proyecto"])

with tab1:
    preset = st.selectbox("Perfil de ejemplo", list(PRESETS))
    P, k = PRESETS[preset], list(PRESETS).index(preset)   # k en las claves: al cambiar de perfil se reinician los campos
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Datos del cliente")
        limit = st.number_input("Límite de crédito", 1_000, 2_000_000, P["limit"], 10_000, key=f"lim{k}")
        c1, c2 = st.columns(2)
        edu = c1.selectbox("Educación", list(EDU), index=P["edu"] - 1, format_func=EDU.get, key=f"edu{k}")
        age = c2.number_input("Edad", 18, 100, P["age"], key=f"age{k}")
        st.caption("Últimos 6 meses (mes -1 = el más reciente). Estado de pago: -1 pagó a tiempo; 1 o más = meses de "
                   "atraso; 0 y -2 no están documentados por la fuente (la comunidad los lee como pago mínimo y sin consumo).")
        hist = st.data_editor(
            pd.DataFrame(P["rows"], columns=["Estado", "Facturado", "Pagado"], index=[f"Mes -{i}" for i in range(1, 7)]),
            key=f"hist{k}",
            column_config={"Estado": st.column_config.NumberColumn(min_value=-2, max_value=9, step=1),
                           "Facturado": st.column_config.NumberColumn(step=1000),
                           "Pagado": st.column_config.NumberColumn(min_value=0, step=1000)})
    with right:
        st.subheader("Resultado")
        ratio = st.slider("Costo de no alertar a quien incumple frente a una alerta innecesaria (X : 1)", 1, 20,
                          int(round(meta["cost_miss"] / meta["cost_alert"])),
                          help="Se alerta si PI >= 1 / (1 + X). X debe salir de los costos de la institución; "
                               "5 es un supuesto ilustrativo.")
        thr = 1 / (1 + ratio)
        if hist.isna().any().any():
            st.warning("Completa todas las celdas del historial.")
        else:
            row = {"LIMIT_BAL": limit, "EDUCATION": edu, "AGE": age}
            for i in range(6):
                row[PAY_COLS[i]], row[BILL_COLS[i]], row[PAYAMT_COLS[i]] = hist.iloc[i].tolist()
            df = pd.DataFrame([row])[RAW_INPUTS]
            pi = float(scorer.predict_pi(df)[0])
            alert = pi >= thr
            m1, m2 = st.columns(2)
            m1.metric("Probabilidad de incumplimiento (PI)", f"{pi:.1%}")
            m2.metric("Umbral de alerta", f"{thr:.1%}")
            (st.error if alert else st.success)(
                f"Acción sugerida: {'ALERTA PREVENTIVA' if alert else 'SEGUIMIENTO NORMAL'}  "
                f"(PI {'≥' if alert else '<'} umbral)")
            st.caption("PI = probabilidad de que el cliente incumpla su pago el mes siguiente (definición del dataset). "
                       "Alerta preventiva = revisar o contactar al cliente antes de que el atraso se agrave; la acción "
                       "concreta depende de la política de la institución.")
            outside = [c for c in RAW_INPUTS
                       if not meta["input_stats"][c]["min"] <= df[c].iloc[0] <= meta["input_stats"][c]["max"]]
            if outside:
                st.warning("Valores fuera del rango visto en el entrenamiento (" + ", ".join(outside) +
                           "): la estimación puede ser poco fiable.")
            res = contributions(scorer, df)
            if res is not None:
                s, x, base, prob = res
                st.markdown("**Qué empuja la PI de este cliente**")
                fig = plot_contrib(s, x, prob)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(f"Rojo: sube la PI. Verde: la baja. Referencia del portafolio de entrenamiento: {base:.1%}.")

with tab2:
    tm, n_test = meta["test_metrics"], meta.get("data", {}).get("n_test")
    st.caption(f"Resultados en el conjunto de test ({n_test:,} clientes)." if n_test else
               "Resultados en el conjunto de test, usado una sola vez.")
    c = st.columns(4)
    c[0].metric("AUC (test)", f"{tm['auc']:.3f}", f"IC95% {tm['auc_ci95'][0]:.3f} a {tm['auc_ci95'][1]:.3f}", delta_color="off")
    c[1].metric("KS", f"{tm['ks']:.3f}", f"IC95% {tm['ks_ci95'][0]:.3f} a {tm['ks_ci95'][1]:.3f}", delta_color="off")
    c[2].metric("Gini", f"{tm['gini']:.2f}")
    c[3].metric("Brier", f"{tm['brier']:.3f}")
    out = BASE / "outputs"
    for img in ["evaluacion.png", "shap_resumen.png"]:
        if (out / img).exists():
            st.image(str(out / img))
    show_csv(out / "deciles_test.csv", "Incumplimiento observado y PI media por decil de riesgo (test)")
    show_csv(out / "politicas_test.csv", "Políticas de decisión con costos supuestos (test)", index_name="Política",
             note="Captura = % de incumplidores que se alertan; precisión = % de alertas que aciertan; el costo por cliente "
                  "está en unidades relativas, no en dinero.")
    show_csv(out / "sensibilidad_costo.csv", "Sensibilidad al costo X (no alertar : alertar)",
             note="Cada fila se evalúa con su propia matriz de costos: la reducción relativa sí es comparable entre filas; "
                  "el costo por cliente no.")
    show_csv(out / "equidad_test.csv", "Auditoría por grupo: sexo y estado civil no entran al modelo")

with tab3:
    project_info.render(meta)

"""
train_model.py
Modelo de alerta temprana: probabilidad de INCUMPLIMIENTO DE PAGO el mes siguiente (PI).

Qué predice: el objetivo del dataset (UCI "Default of Credit Card Clients") indica si el cliente
incumplió su pago en octubre de 2005. NO es un default en sentido regulatorio (90+ días de atraso)
ni una pérdida: es un incumplimiento a un mes. Por eso el caso de uso es la gestión preventiva
(alertas tempranas, cobranza preventiva, gestión de líneas).

Flujo:
  1. Datos y partición en train/test
  2. Comparación de familias de modelos con validación cruzada repetida
  3. Ajuste de hiperparámetros (búsqueda aleatoria + CV estratificada)
  4. Selección con regla de parsimonia (1 error estándar)
  5. Calibración de probabilidades
  6. Umbral de alerta según los costos
  7. Evaluación en test: IC, deciles, políticas, equidad y sensibilidad al costo
  8. SHAP y exportación del modelo para la app

Uso:  python train_model.py            (completo)
      python train_model.py --fast     (prueba rápida para verificar que todo corre)
"""
# ---------------------------------------------------------------------------
# LIBRERIAS
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from xgboost import XGBClassifier

from credit_utils import RAW_INPUTS, SENSITIVE, CreditScorer, PlattCalibrator, build_pipeline

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
try:
    BASE = Path(__file__).resolve().parent
except NameError:                      # p. ej. si se ejecuta por celdas
    BASE = Path.cwd()
SEED = 42
TARGET = "incumple"                    # 1 = incumplió su pago el mes siguiente (octubre de 2005)
CSV_NAME = "UCI_Credit_Card.csv"       # CSV de Kaggle
DATA_CANDIDATES = [BASE / CSV_NAME, BASE / "data" / CSV_NAME, BASE / "data" / "credit_default.csv"]
OUT_DIR, MODEL_DIR = BASE / "outputs", BASE / "models"

TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER = {"LogReg": 12, "RandomForest": 10, "XGBoost": 30, "LightGBM": 30}

# Costos relativos de la decisión de alertar. SON SUPUESTOS ILUSTRATIVOS: en una institución real
# salen del costo de contactar, del valor de intervenir a tiempo y de la pérdida evitable.
COST_MISS = 5.0    # no alertar a un cliente que luego incumple (se interviene tarde o nunca)
COST_ALERT = 1.0   # alertar a un cliente que habría pagado (contacto innecesario, fricción)

# Para desempatar con parsimonia: entre los modelos a menos de 1 error estándar del mejor
# AUC, se elige el de menor "complejidad". El orden entre familias es un JUICIO del modelador
# (aquí: logística < Random Forest < boosting). Si prefieres quedarte siempre con el mejor
# AUC medio entre los ensambles, ponles el mismo valor a los tres.
COMPLEXITY = {"LogReg": 1, "RandomForest": 2, "LightGBM": 3, "XGBoost": 3}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Localiza, carga y realiza comprobaciones básicas del dataset de crédito.

    Proceso
    -------
    1. Busca el archivo CSV entre las rutas definidas en DATA_CANDIDATES.
    2. Lee los datos con pandas.
    3. Homogeneiza algunos nombres de columnas.
    4. Elimina la columna ID si existe.
    5. Comprueba que estén presentes las variables requeridas.
    6. Elimina registros duplicados exactos.
    7. Guarda metadatos sobre el archivo y los duplicados en df.attrs.
    8. Imprime un resumen del tamaño de la base y de la tasa de incumplimiento.

    Retorna
    -------
    pd.DataFrame
        Dataset limpio de duplicados exactos y con nombres de columnas
        estandarizados.

    Excepciones
    -----------
    SystemExit
        Si no encuentra el archivo o si faltan columnas necesarias.

    Notas
    -----
    La función no separa los datos en entrenamiento y prueba,
    ni entrena modelos. Tampoco elimina valores faltantes de forma explícita.
    """
    path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if path is None:
        raise SystemExit(f"No encontré {CSV_NAME}. Colócalo en {BASE} (junto a los scripts).")
    df = pd.read_csv(path)
    df = df.rename(columns={"PAY_0": "PAY_1", "default.payment.next.month": TARGET,
                            "default payment next month": TARGET, "default": TARGET}).drop(columns=["ID"], errors="ignore")
    faltan = (set(RAW_INPUTS) | set(SENSITIVE) | {TARGET}) - set(df.columns)
    if faltan:
        raise SystemExit(f"Al CSV le faltan columnas: {sorted(faltan)}")
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)   # evita que un mismo registro caiga en train y test
    df.attrs.update(n_raw=n0, n_dup=n0 - len(df), file=path.name)
    print(f"{path.name}: {n0:,} filas ({n0 - len(df)} duplicados eliminados) | "
          f"incumplimiento = {df[TARGET].mean():.1%}")
    return df


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def ks_stat(y, p) -> float:
    """
    Calcula el estadístico Kolmogorov-Smirnov (KS) para un modelo binario.

    Parámetros
    ----------
    y : array-like
        Etiquetas reales: 1 para incumplimiento y 0 para no incumplimiento.
    p : array-like
        Probabilidades estimadas de incumplimiento.

    Retorna
    -------
    float
        Máxima diferencia entre la tasa de verdaderos positivos y la tasa
        de falsos positivos a través de los umbrales de clasificación.

    Interpretación
    --------------
    Un KS más alto indica una mayor capacidad de separar las dos clases.
    """
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p) -> dict:
    """
    Calcula un conjunto de métricas de evaluación del modelo.

    Parámetros
    ----------
    y : array-like
        Etiquetas reales de incumplimiento.
    p : array-like
        Probabilidades estimadas de incumplimiento.

    Retorna
    -------
    dict
        Diccionario con AUC-ROC, KS, PR-AUC, Brier Score y Log Loss.
    """
    return {"auc": roc_auc_score(y, p), 
            "ks": ks_stat(y, p), 
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), 
            "logloss": log_loss(y, p)}


def bootstrap_ci(y, p, fn, n=500, seed=SEED):
    """
    Estima un intervalo de confianza del 95 % mediante bootstrap.

    Parámetros
    ----------
    y : array-like
        Etiquetas reales.
    p : array-like
        Probabilidades estimadas.
    fn : callable
        Función que calcula la métrica que se desea evaluar.
        Debe aceptar dos argumentos: etiquetas y probabilidades.
    n : int, por defecto 500
        Número de remuestras bootstrap.
    seed : int
        Semilla para reproducibilidad de los resultados.

    Retorna
    -------
    numpy.ndarray
        Percentiles 2.5 y 97.5 de la distribución bootstrap de la métrica.
    """
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def _fit_predict(model, X_tr, y_tr, X_va):
    """
    Ajusta una copia del modelo y estima probabilidades en validación.

    Parámetros
    ----------
    model : estimador compatible con scikit-learn
        Modelo o pipeline que se desea entrenar.
    X_tr : DataFrame o matriz
        Variables predictoras de entrenamiento.
    y_tr : array-like
        Etiquetas de entrenamiento.
    X_va : DataFrame o matriz
        Variables predictoras de validación.

    Retorna
    -------
    numpy.ndarray
        Probabilidades estimadas de pertenecer a la clase 1.
    """
    return clone(model).fit(X_tr, y_tr).predict_proba(X_va)[:, 1]


def cross_val(model, X, y, cv, n_jobs):
    """
    Ejecuta validación cruzada y calcula predicciones out-of-fold.

    Parámetros
    ----------
    model : estimador compatible con scikit-learn
        Modelo o pipeline que se evaluará.
    X : DataFrame
        Variables predictoras.
    y : Series o array-like
        Etiquetas reales.
    cv : objeto de validación cruzada
        Genera los índices de entrenamiento y validación de cada fold.
    n_jobs : int
        Número de trabajos paralelos. -1 suele utilizar todos los núcleos
        disponibles.

    Retorna
    -------
    tuple
        Un DataFrame con las métricas de cada fold y un arreglo con
        las predicciones out-of-fold de cada observación.

    Notas
    -----
    Si una observación aparece en varias particiones de validación,
    sus probabilidades se promedian. Esto puede ocurrir con esquemas
    de validación cruzada repetida.
    """
    splits = list(cv.split(X, y))
    preds = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(model, X.iloc[tr], y.iloc[tr], X.iloc[va]) for tr, va in splits)
    rows = [metrics(y.iloc[va], p) for (_, va), p in zip(splits, preds)]
    oof, cnt = np.zeros(len(y)), np.zeros(len(y))
    for (_, va), p in zip(splits, preds):
        oof[va] += p
        cnt[va] += 1
    return pd.DataFrame(rows), oof / cnt


# ---------------------------------------------------------------------------
# Familias de modelos y espacios de búsqueda
# ---------------------------------------------------------------------------
def base_models() -> dict:
    """
    Construye los modelos iniciales de clasificación dentro de un pipeline.

    La función define una configuración base para cada algoritmo. Los
    hiperparámetros podrán modificarse posteriormente durante la búsqueda.

    Returns
    -------
    dict
       Diccionario que asocia el nombre de cada modelo con su pipeline.
    """
    return {
        "LogReg": build_pipeline(
            LogisticRegression(max_iter=2000), 
            scale=True),
        "RandomForest": build_pipeline(
            RandomForestClassifier(
                n_estimators=200, 
                min_samples_leaf=10,
                n_jobs=1, 
                random_state=SEED), 
            scale=False),
        "XGBoost": build_pipeline(
            XGBClassifier(
                n_estimators=300, 
                learning_rate=0.05, 
                max_depth=4, 
                tree_method="hist",
                n_jobs=1, 
                random_state=SEED, 
                eval_metric="logloss"), 
            scale=False),
        "LightGBM": build_pipeline(
            LGBMClassifier(
                n_estimators=300, 
                learning_rate=0.05, 
                num_leaves=31, 
                subsample_freq=1,
                n_jobs=1, 
                random_state=SEED, 
                verbose=-1), 
            scale=False),
    }


# Espacio de hiperparámetros
SPACES = {
    "LogReg": {"model__C": loguniform(1e-3, 1e2)},
    "RandomForest": {
        "model__n_estimators": randint(150, 350),
        "model__max_depth": [6, 10, 14, None],
        "model__min_samples_leaf": randint(5, 60),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "XGBoost": {
        "model__n_estimators": randint(150, 600),
        "model__learning_rate": loguniform(0.02, 0.2),
        "model__max_depth": randint(2, 7),
        "model__min_child_weight": loguniform(1, 50),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(0.1, 30),
    },
    "LightGBM": {
        "model__n_estimators": randint(150, 600),
        "model__learning_rate": loguniform(0.02, 0.2),
        "model__num_leaves": randint(8, 64),
        "model__min_child_samples": randint(20, 200),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(0.1, 30),
    },
}


# ---------------------------------------------------------------------------
# Decisión (alertar / no alertar) y costos
# ---------------------------------------------------------------------------
def expected_cost(y, alert, cost_miss=None, cost_alert=None) -> float:
    """Calcula el costo promedio por cliente de una política de alertas."""
    cm = COST_MISS if cost_miss is None else cost_miss
    ca = COST_ALERT if cost_alert is None else cost_alert
    y, alert = np.asarray(y), np.asarray(alert)
    miss = np.sum(~alert & (y == 1))          # no alerté y el cliente incumplió
    false_alarm = np.sum(alert & (y == 0))    # alerté y el cliente habría pagado
    return (miss * cm + false_alarm * ca) / len(y)


def policy_row(y, alert) -> dict:
    """Resume las principales métricas de una política de alertas."""
    y, alert = np.asarray(y), np.asarray(alert)
    tp = np.sum(alert & (y == 1))
    return {"tasa_alerta": alert.mean(), "captura": tp / max(y.sum(), 1),
            "precision_alerta": tp / alert.sum() if alert.any() else np.nan,
            "costo_por_cliente": expected_cost(y, alert)}


def cost_curve(y, p, grid) -> np.ndarray:
    """Calcula el costo promedio para cada umbral de alerta."""
    return np.array([expected_cost(y, np.asarray(p) >= t) for t in grid])


def cost_sensitivity(y, p, ratios=(1, 2, 3, 5, 10, 20)) -> pd.DataFrame:
    """Evalúa cómo cambia el costo al variar la importancia de detectar un incumplimiento."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for r in ratios:
        t = 1 / (1 + r)
        alert = p >= t
        tp = np.sum(alert & (y == 1))
        cost = expected_cost(y, alert, cost_miss=r, cost_alert=1.0)
        triv = min(y.mean() * r, 1 - y.mean())     # mejor política trivial: nadie o todos
        rows.append({"X": r, "umbral": t, "tasa_alerta": alert.mean(), "captura": tp / y.sum(),
                     "precision_alerta": tp / alert.sum() if alert.any() else np.nan,
                     "costo_modelo": cost, "costo_mejor_trivial": triv, "reduccion": 1 - cost / triv})
    return pd.DataFrame(rows)


def fmt_params(params: dict) -> dict:
    """Simplifica los nombres y redondea los parámetros del modelo."""
    return {k.replace("model__", ""): round(float(v), 4) if isinstance(v, (float, np.floating)) else v
            for k, v in params.items()}


def section(title: str):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="pocas iteraciones, para probar que corre")
    ap.add_argument("--jobs", type=int, default=-1, help="núcleos a usar (-1 = todos)")
    args = ap.parse_known_args()[0]      # tolera argumentos extra (p. ej. al ejecutar desde Spyder)
    n_iter = {k: 3 for k in N_ITER} if args.fast else N_ITER
    folds = 3 if args.fast else CV_FOLDS
    OUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    # 1. Datos y partición -----------------------------------------------------
    section("1. DATOS Y PARTICIÓN")
    df = load_data()
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"train = {len(X_train):,} | test = {len(X_test):,} (no se toca hasta el paso 7)")

    # Evidencia sobre qué mide realmente el objetivo (se muestra en la app y el reporte)
    al_dia = df["PAY_1"] <= 0
    por_pay1 = df.groupby("PAY_1")[TARGET].agg(["size", "mean"])
    evidence = {"pct_al_dia": float(al_dia.mean()), "tasa_en_al_dia": float(df.loc[al_dia, TARGET].mean()),
                "pct_positivos_al_dia": float(df.loc[al_dia, TARGET].sum() / df[TARGET].sum()),
                "pct_atraso_3m": float((df["PAY_1"] >= 3).mean()),
                "por_pay1": [{"PAY_1": int(k), "clientes": int(r["size"]), "tasa": float(r["mean"])}
                             for k, r in por_pay1.iterrows()]}
    print(f"Al día en septiembre: {evidence['pct_al_dia']:.1%} -> objetivo=1 en {evidence['tasa_en_al_dia']:.1%} "
          f"({evidence['pct_positivos_al_dia']:.1%} de todos los positivos)")

    # 2. Comparación de familias con hiperparámetros "razonables" ---------------
    section("2. COMPARACIÓN DE FAMILIAS (CV estratificada repetida, sin tuning)")
    cv_rep = RepeatedStratifiedKFold(n_splits=folds, n_repeats=1 if args.fast else 2, random_state=SEED)
    rows = {}
    for name, pipe in base_models().items():
        m, _ = cross_val(pipe, X_train, y_train, cv_rep, args.jobs)
        rows[name] = m.mean()
        print(f"  {name:<13} AUC={m['auc'].mean():.4f} ± {m['auc'].std():.4f}  [{time.time() - t0:5.0f}s]")
    base_table = pd.DataFrame(rows).T
    print(base_table.round(4).to_string())

    # 3. Ajuste de hiperparámetros ---------------------------------------------
    section("3. AJUSTE DE HIPERPARÁMETROS (RandomizedSearchCV, scoring = ROC-AUC)")
    cv_tune = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    searches = {}
    for name, pipe in base_models().items():
        s = RandomizedSearchCV(pipe, SPACES[name], n_iter=n_iter[name], scoring="roc_auc",
                               cv=cv_tune, n_jobs=args.jobs, random_state=SEED, refit=True)
        s.fit(X_train, y_train)
        searches[name] = s
        print(f"  {name:<13} mejor AUC en CV = {s.best_score_:.4f}  [{time.time() - t0:5.0f}s]")
        print(f"     {fmt_params(s.best_params_)}")

    # 4. Selección con CV "fresca" + regla de 1 error estándar --------------------
    section("4. SELECCIÓN DEL MODELO (CV con otra semilla + regla de parsimonia)")
    cv_confirm = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED + 1)
    conf, oofs = {}, {}
    for name, s in searches.items():
        m, oofs[name] = cross_val(s.best_estimator_, X_train, y_train, cv_confirm, args.jobs)
        conf[name] = m
    tuned_table = pd.DataFrame({n: m.mean() for n, m in conf.items()}).T
    tuned_table["auc_std"] = [conf[n]["auc"].std() for n in tuned_table.index]
    print(tuned_table.round(4).to_string())

    best = tuned_table["auc"].idxmax()
    se = conf[best]["auc"].std(ddof=1) / np.sqrt(len(conf[best]))
    eligible = [n for n in tuned_table.index if tuned_table.loc[n, "auc"] >= tuned_table.loc[best, "auc"] - se]
    chosen = min(eligible, key=lambda n: (COMPLEXITY[n], -tuned_table.loc[n, "auc"]))
    print(f"\nMejor AUC medio: {best} ({tuned_table.loc[best, 'auc']:.4f}, EE = {se:.4f})")
    print(f"Dentro de 1 EE: {eligible} -> elegido por parsimonia: {chosen}")
    comparison = [{"modelo": n, "auc_base": float(base_table.loc[n, "auc"]),
                   "auc_ajustado": float(tuned_table.loc[n, "auc"]), "auc_std": float(tuned_table.loc[n, "auc_std"]),
                   "ks": float(tuned_table.loc[n, "ks"]), "logloss": float(tuned_table.loc[n, "logloss"])}
                  for n in tuned_table.index]
    pd.DataFrame(comparison).to_csv(OUT_DIR / "comparacion_modelos.csv", index=False)

    final = searches[chosen].best_estimator_            # ya reentrenado con todo el train
    baseline = searches["LogReg"].best_estimator_       # referencia: ¿vale la pena la complejidad?

    # 5. Calibración -------------------------------------------------------------
    section("5. CALIBRACIÓN DE PROBABILIDADES")
    platt = PlattCalibrator().fit(oofs[chosen], y_train)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(oofs[chosen], y_train)
    p_raw = final.predict_proba(X_test)[:, 1]
    p_platt, p_iso = platt.predict(p_raw), iso.predict(p_raw)
    cal_table = pd.DataFrame({k: {"brier": brier_score_loss(y_test, p), "logloss": log_loss(y_test, p),
                                  "auc": roc_auc_score(y_test, p)}
                              for k, p in [("sin calibrar", p_raw), ("Platt (sigmoide)", p_platt),
                                           ("isotónica", p_iso)]}).T
    print(cal_table.round(4).to_string())
    print("(Platt es monótona: no cambia el AUC, solo corrige el nivel de las probabilidades.)")
    p_test = p_platt                                    # PI calibrada que se despliega

    # 6. Umbral de alerta por costos ------------------------------------------------
    section("6. UMBRAL DE ALERTA SEGÚN COSTOS")
    thr = CreditScorer.cutoff(COST_MISS, COST_ALERT)
    grid = np.linspace(0.03, 0.70, 68)
    pi_oof = platt.predict(oofs[chosen])
    thr_emp = float(grid[np.argmin(cost_curve(y_train, pi_oof, grid))])
    print(f"Costo de no alertar a quien incumple = {COST_MISS:g} x costo de una alerta innecesaria ({COST_ALERT:g})")
    print(f"Umbral teórico  PI* = {COST_ALERT:g}/({COST_ALERT:g}+{COST_MISS:g}) = {thr:.3f}  (alertar si PI >= PI*)")
    print(f"Umbral empírico (barrido sobre predicciones out-of-fold del train) = {thr_emp:.3f}")

    # 7. Evaluación final en test --------------------------------------------------
    section("7. EVALUACIÓN FINAL EN TEST (se mira una sola vez)")
    final_m = metrics(y_test, p_test)
    final_m["gini"] = 2 * final_m["auc"] - 1
    auc_ci = bootstrap_ci(y_test, p_test, roc_auc_score)
    ks_ci = bootstrap_ci(y_test, p_test, ks_stat)
    print(f"{chosen}: AUC = {final_m['auc']:.4f} (IC95% {auc_ci[0]:.3f}-{auc_ci[1]:.3f}) | "
          f"Gini = {final_m['gini']:.3f} | KS = {final_m['ks']:.3f} (IC95% {ks_ci[0]:.3f}-{ks_ci[1]:.3f})")
    print(f"PR-AUC = {final_m['pr_auc']:.3f} (línea base = tasa de incumplimiento = {y_test.mean():.3f}) | "
          f"Brier = {final_m['brier']:.4f} | log-loss = {final_m['logloss']:.4f}")

    p_lr_raw = baseline.predict_proba(X_test)[:, 1]
    lr_platt = PlattCalibrator().fit(oofs["LogReg"], y_train)
    p_lr = lr_platt.predict(p_lr_raw)
    lr_test = {"auc": float(roc_auc_score(y_test, p_lr)), "ks": ks_stat(y_test, p_lr)}
    print(f"Referencia LogReg: AUC = {lr_test['auc']:.4f} | KS = {lr_test['ks']:.3f}")

    print("\nPolíticas en test (umbral teórico):")
    pol = pd.DataFrame({
        "No alertar a nadie": policy_row(y_test, np.zeros(len(y_test), bool)),
        "Alertar a todos": policy_row(y_test, np.ones(len(y_test), bool)),
        "LogReg": policy_row(y_test, p_lr >= thr),
        chosen: policy_row(y_test, p_test >= thr),
    }).T
    print(pol.round(4).to_string())
    triv = min(pol.loc["No alertar a nadie", "costo_por_cliente"], pol.loc["Alertar a todos", "costo_por_cliente"])
    saving = 1 - pol.loc[chosen, "costo_por_cliente"] / triv
    print(f"Reducción de costo vs. la mejor política trivial: {saving:.1%}")
    pol.to_csv(OUT_DIR / "politicas_test.csv")

    print("\nOrdenamiento por deciles de riesgo (1 = menor PI):")
    dec = pd.qcut(pd.Series(p_test).rank(method="first"), 10, labels=False) + 1
    dec_tab = (pd.DataFrame({"decil": dec.to_numpy(), "pi": p_test, "incumple": y_test.to_numpy()})
               .groupby("decil").agg(clientes=("incumple", "size"), pi_media=("pi", "mean"),
                                     incumplimiento_observado=("incumple", "mean")))
    print(dec_tab.round(3).to_string())
    dec_tab.to_csv(OUT_DIR / "deciles_test.csv")

    # Equidad: variables sensibles NO entran al modelo, pero se auditan ----------
    print(f"\nAuditoría de equidad (umbral de alerta = {thr:.3f}):")
    groups = {"SEX": {1: "Hombre", 2: "Mujer"}, "MARRIAGE": {1: "Casado/a", 2: "Soltero/a", 3: "Otro"}}
    fr = []
    for col, labels in groups.items():
        g = X_test[col].replace({0: 3}) if col == "MARRIAGE" else X_test[col]
        for val, lab in labels.items():
            mk = (g == val).to_numpy()
            fr.append({"variable": col, "grupo": lab, "n": int(mk.sum()),
                       "incumplimiento_observado": y_test[mk].mean(), "pi_media": p_test[mk].mean(),
                       "tasa_alerta": (p_test[mk] >= thr).mean(), "auc": roc_auc_score(y_test[mk], p_test[mk])})
    fair = pd.DataFrame(fr)
    print(fair.round(3).to_string(index=False))
    fair_ratio = {}
    for col in groups:
        a = fair.loc[fair["variable"] == col, "tasa_alerta"]
        fair_ratio[col] = float(a.min() / a.max())
        print(f"  Razón de tasas de alerta mín/máx en {col}: {fair_ratio[col]:.2f}")
    fair.to_csv(OUT_DIR / "equidad_test.csv", index=False)

    sens = cost_sensitivity(y_test, p_test)
    print("\nSensibilidad al costo X (no alertar : alertar):\n" + sens.round(3).to_string(index=False))
    sens.to_csv(OUT_DIR / "sensibilidad_costo.csv", index=False)

    # Gráficos de evaluación ------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for lab, p in [(chosen, p_test), ("LogReg (referencia)", p_lr)]:
        fpr, tpr, _ = roc_curve(y_test, p)
        ax[0].plot(fpr, tpr, label=f"{lab}  AUC={roc_auc_score(y_test, p):.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(title="Curva ROC (test)", xlabel="Falsos positivos", ylabel="Verdaderos positivos")
    ax[0].legend(loc="lower right")
    for lab, p in [("Sin calibrar", p_raw), ("Calibrado (Platt)", p_test)]:
        frac, mean = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax[1].plot(mean, frac, marker="o", label=lab)
    top = max(ax[1].get_xlim()[1], ax[1].get_ylim()[1])
    ax[1].plot([0, top], [0, top], "--", color="grey", label="Calibración perfecta")
    ax[1].set(title="Calibración (test)", xlabel="PI predicha", ylabel="Incumplimiento observado")
    ax[1].legend()
    ax[2].plot(grid, cost_curve(y_test, p_test, grid), label=f"Modelo ({chosen})")
    ax[2].axhline(pol.loc["No alertar a nadie", "costo_por_cliente"], color="red", ls=":", label="No alertar a nadie")
    ax[2].axhline(pol.loc["Alertar a todos", "costo_por_cliente"], color="orange", ls=":", label="Alertar a todos")
    ax[2].axvline(thr, color="black", ls="--", label=f"Umbral = {thr:.2f}")
    ax[2].set(title=f"Costo esperado por cliente (no alertar : alertar = {COST_MISS:g}:{COST_ALERT:g})",
              xlabel="Umbral de PI (alertar si PI >= umbral)", ylabel="Costo por cliente")
    ax[2].set_ylim(top=1.35)
    ax[2].legend(loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "evaluacion.png", dpi=130)
    plt.close(fig)

    # 8. Interpretabilidad (SHAP) y exportación -------------------------------------
    section("8. INTERPRETABILIDAD Y EXPORTACIÓN")
    sample = X_test.sample(min(2000, len(X_test)), random_state=SEED)
    feats = final[:-1]
    Xt = pd.DataFrame(feats.transform(sample), columns=feats.named_steps["prep"].get_feature_names_out())
    mdl = final.named_steps["model"]
    expl = shap.LinearExplainer(mdl, Xt) if isinstance(mdl, LogisticRegression) else shap.TreeExplainer(mdl)
    sv = expl.shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
    escala = "probabilidad" if isinstance(mdl, RandomForestClassifier) else "log-odds"
    plt.figure()
    shap.summary_plot(sv, Xt, show=False, max_display=15)
    plt.gcf().axes[0].set_xlabel(f"Valor SHAP (impacto en la PI, escala {escala})")
    plt.title("¿Qué variables empujan la PI hacia arriba o hacia abajo?")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_resumen.png", dpi=130)
    plt.close()
    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xt.columns).sort_values(ascending=False)
    print("Top 8 variables por |SHAP| medio:\n" + imp.head(8).round(4).to_string())

    meta = {
        "model_name": chosen,
        "best_params": fmt_params(searches[chosen].best_params_),
        "cost_miss": COST_MISS, "cost_alert": COST_ALERT, "threshold": thr, "threshold_empirical": thr_emp,
        "test_metrics": {**final_m, "auc_ci95": auc_ci.tolist(), "ks_ci95": ks_ci.tolist()},
        "lr_test": lr_test,
        "calibration": cal_table.round(5).to_dict("index"),
        "selection": {"best": best, "se": float(se), "eligible": eligible, "comparison": comparison},
        "evidence": evidence,
        "fairness_ratio": fair_ratio,
        "shap": {"scale": escala, "top": [[k, float(v)] for k, v in imp.head(10).items()]},
        "data": {"file": df.attrs["file"], "n_raw": df.attrs["n_raw"], "n_dup": df.attrs["n_dup"],
                 "n_total": len(df), "n_train": len(X_train), "n_test": len(X_test),
                 "prevalence": float(y.mean())},
        "cv": {"folds": folds, "n_iter": dict(n_iter), "fast": bool(args.fast)},
        "dataset": "UCI - Default of Credit Card Clients (Taiwán, 2005)",
        "n_train": len(X_train), "trained_at": datetime.now().isoformat(timespec="seconds"),
        "input_stats": X_train[RAW_INPUTS].describe(percentiles=[.01, .5, .99]).T[
            ["min", "1%", "50%", "99%", "max"]].to_dict("index"),
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    joblib.dump(CreditScorer(final, platt, meta), MODEL_DIR / "credit_model.joblib")
    with open(OUT_DIR / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nModelo guardado en {MODEL_DIR / 'credit_model.joblib'}")
    print(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

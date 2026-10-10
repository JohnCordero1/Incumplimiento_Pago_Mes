"""
credit_utils.py
Código para emplear en entrenamiento (train_model.py) y la app (Streamlit).

El modelo estima la PI: probabilidad de que el cliente INCUMPLA SU PAGO el mes siguiente.
Nota: No es un default en sentido regulatorio ni una pérdida

Idea: el pipeline guardado recibe al cliente "crudo" (las columnas originales
del dataset) y hace internamente la limpieza, la ingeniería de variables y el modelo.
Así, la app solo arma un DataFrame de una fila y llama a `predict_pd`, sin duplicar
lógica (y sin riesgo de que entrenamiento y producción calculen las variables distinto).
"""
# ---------------------------------------------------------------------------
# LIBRERIAS
# ---------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Esquema de datos
# ---------------------------------------------------------------------------
PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]         # estado de pago; _1 = mes más reciente
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]    # monto facturado
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]   # monto pagado
SENSITIVE = ["SEX", "MARRIAGE"]                      # NO entran al modelo; solo se usan para auditar

RAW_INPUTS = ["LIMIT_BAL", "EDUCATION", "AGE"] + PAY_COLS + BILL_COLS + PAYAMT_COLS
ENGINEERED = ["util_1", "util_mean", "util_max", "pay_ratio_last", "pay_ratio_mean",
              "n_delays", "max_delay", "delay_trend"]
CAT_COLS = ["EDUCATION"]
NUM_COLS = [c for c in RAW_INPUTS if c not in CAT_COLS] + ENGINEERED


# ---------------------------------------------------------------------------
# Limpieza + ingeniería de variables (opera fila a fila => no hay fuga de datos)
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los datos y construye variables adicionales de comportamiento crediticio.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales de los clientes. Debe contener
        las columnas definidas en RAW_INPUTS.

    Proceso
    -------
    1. Agrupa los códigos de educación no documentados en la categoría "otros".
    2. Calcula la utilización de la línea de crédito.
    3. Calcula la proporción pagada de facturas anteriores.
    4. Resume la frecuencia, magnitud y tendencia de los atrasos.

    Retorna
    -------
    pd.DataFrame
        DataFrame con las variables numéricas y categóricas seleccionadas.
        Excluye las variables sensibles SEX y MARRIAGE y cualquier columna
        adicional que no esté especificada en NUM_COLS o CAT_COLS.

    Notas
    -----
    La transformación se realiza por cliente, sin calcular estadísticas
    entre observaciones. Por ello, no requiere estimar parámetros globales
    como medias o desviaciones estándar.
    """
    X = df.copy()
    # Existen códigos de educación no documentados en Kaggle (0, 5, 6) -> "otros" (4)
    X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4}).astype(int)
    
    #Convertimos a matrices de numpy
    limit = np.maximum(X["LIMIT_BAL"].to_numpy(float), 1.0)[:, None]
    bills = X[BILL_COLS].to_numpy(float)
    paid = X[PAYAMT_COLS].to_numpy(float)
    status = X[PAY_COLS].to_numpy(float)

    #Divide cada factura entre el límite de crédito del cliente.
    #Los saldos negativos se consideran utilización cero y se limita el indicador max 1.5 (150%)
    util = np.clip(bills / limit, 0, 1.5)
    X["util_1"] = util[:, 0]
    X["util_mean"] = util.mean(axis=1)
    X["util_max"] = util.max(axis=1)

    # Fracción de la factura anterior que se pagó (PAY_AMT_k paga la factura k+1)
    prev_bill = bills[:, 1:]
    ratio = np.where(prev_bill > 0, paid[:, :-1] / np.where(prev_bill > 0, prev_bill, 1.0), 1.0)
    ratio = np.clip(ratio, 0, 1)
    X["pay_ratio_last"] = ratio[:, 0]
    X["pay_ratio_mean"] = ratio.mean(axis=1)

    # Comportamiento de atraso (valores > 0 = meses de atraso)
    X["n_delays"] = (status > 0).sum(axis=1)
    X["max_delay"] = status.max(axis=1)
    X["delay_trend"] = status[:, 0] - status[:, 1:].mean(axis=1)   # > 0: empeorando recientemente

    return X[NUM_COLS + CAT_COLS]      # descarta SEX / MARRIAGE y cualquier columna extra


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """
    Construye un preprocesador para las variables numéricas y categóricas.

    Parámetros
    ----------
    scale : bool
        Si es True, estandariza las variables numéricas mediante StandardScaler.
        Si es False, conserva sus valores originales mediante passthrough.

    Proceso
    -------
    1. Selecciona el tratamiento de las variables numéricas (estandarización)
    2. Codifica EDUCATION mediante One-Hot Encoding.
    3. Combina ambas transformaciones en un ColumnTransformer.

    Retorna
    -------
    ColumnTransformer
        Objeto que aplica las transformaciones especificadas a las columnas
        numéricas y categóricas, respectivamente.

    Notas
    -----
    El escalado suele ser útil para modelos lineales, SVM y redes neuronales.
    Los árboles de decisión y los modelos basados en árboles generalmente
    no requieren escalado.
    """
    num = StandardScaler() if scale else "passthrough"
    cat = OneHotEncoder(categories=[[1, 2, 3, 4]], handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)],
                             verbose_feature_names_out=False)


def build_pipeline(estimator, scale: bool) -> Pipeline:
    """
    Construye un pipeline completo para predecir el incumplimiento de pagos.

    Parámetros
    ----------
    estimator : estimador compatible con scikit-learn
        Modelo de machine learning que se entrenará, por ejemplo,
        regresión logística, Random Forest, SVM o MLP.

    scale : bool
        Si es True, estandariza las variables numéricas.
        Si es False, conserva sus valores originales.

    Proceso
    -------
    1. Construye variables adicionales y limpia los datos mediante función creada add_features.
    2. Preprocesa las variables numéricas y categóricas.
    3. Entrena el modelo seleccionado con los datos transformados.

    Retorna
    -------
    Pipeline
        Pipeline de scikit-learn que integra las transformaciones
        y el estimador en un único objeto.

    Notas
    -----
    Al ajustar el pipeline con fit(), las transformaciones que necesitan
    estimar parámetros aprenden esos parámetros a partir de los datos
    de entrenamiento. Posteriormente, predict() o predict_proba()
    aplican las transformaciones necesarias antes de generar predicciones.
    """
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", make_preprocessor(scale)),
        ("model", estimator),
    ])


# ---------------------------------------------------------------------------
# Calibración y objeto final que consume la app
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """
    Calibra probabilidades mediante una transformación sigmoide.

    Ajusta una regresión logística sobre el logit de las probabilidades
    originales para aproximarlas a las frecuencias observadas.

    Métodos
    -------
    fit(p, y)
        Aprende la transformación de calibración usando las probabilidades
        predichas y los resultados observados.

    predict(p)
        Devuelve las probabilidades calibradas para nuevas observaciones.

    _logit(p)
        Transforma las probabilidades en log-odds y limita sus valores
        para evitar logaritmos y divisiones numéricamente problemáticos.

    Notas
    -----
    Si el coeficiente estimado de la regresión logística es positivo,
    la transformación es monótona creciente y conserva el orden de las
    probabilidades originales, salvo posibles empates numéricos.
    """

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(self._logit(p))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)


class CreditScorer:
    """Pipeline entrenado + calibrador + metadatos. Es lo que se guarda con joblib."""

    def __init__(self, pipeline, calibrator, meta: dict):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.meta = meta

    def predict_pi(self, df: pd.DataFrame) -> np.ndarray:
        """PI calibrada (probabilidad de incumplir el pago el mes siguiente) para cada fila de `df`."""
        return self.calibrator.predict(self.pipeline.predict_proba(df)[:, 1])

    @staticmethod
    def cutoff(cost_miss: float, cost_alert: float) -> float:
        """Alertar si PI >= cost_alert / (cost_alert + cost_miss). Óptimo solo si la PI está calibrada."""
        return cost_alert / (cost_alert + cost_miss)

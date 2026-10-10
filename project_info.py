"""
project_info.py - Contenido de la pestaña "Sobre el proyecto".
Las cifras salen de lo que guardó el entrenamiento (meta), no están escritas a mano.
"""
import pandas as pd
import streamlit as st

VARIABLES = pd.DataFrame([
    ["LIMIT_BAL", "Límite de crédito otorgado (incluye el crédito individual y el familiar o suplementario)",
     "NT$ (dólares taiwaneses)", "Original", "Sí"],
    ["SEX", "Sexo", "1 = hombre, 2 = mujer", "Original", "No: solo auditoría de equidad"],
    ["EDUCATION", "Nivel educativo",
     "1 posgrado, 2 universidad, 3 secundaria, 4 otros (los códigos 0, 5 y 6 no están documentados y se agrupan en 4)",
     "Original", "Sí (one-hot)"],
    ["MARRIAGE", "Estado civil", "1 casado, 2 soltero, 3 otros (el 0 no está documentado)", "Original",
     "No: solo auditoría de equidad"],
    ["AGE", "Edad", "años", "Original", "Sí"],
    ["PAY_1 … PAY_6", "Estado de pago de cada mes. PAY_1 = septiembre de 2005 (el más reciente), PAY_6 = abril",
     "−1 = pagó a tiempo; 1 a 9 = meses de atraso. Los valores 0 y −2 no están documentados "
     "(la comunidad los lee como pago mínimo y sin consumo)", "Original", "Sí"],
    ["BILL_AMT1 … 6", "Monto del estado de cuenta de cada mes (septiembre → abril)",
     "NT$; puede ser negativo (saldo a favor)", "Original", "Sí"],
    ["PAY_AMT1 … 6", "Monto pagado en cada mes (septiembre → abril)", "NT$", "Original", "Sí"],
    ["util_1, util_mean, util_max", "Uso de la línea = factura / límite (mes más reciente, promedio y máximo de 6 meses)",
     "proporción, acotada a [0; 1,5]", "Derivada", "Sí"],
    ["pay_ratio_last, pay_ratio_mean", "Fracción de la factura del mes anterior que se pagó (último mes y promedio)",
     "proporción, acotada a [0; 1]", "Derivada", "Sí"],
    ["n_delays", "Cantidad de meses con atraso (PAY > 0) entre los 6", "0 a 6", "Derivada", "Sí"],
    ["max_delay", "Mayor valor de PAY en los 6 meses", "meses", "Derivada", "Sí"],
    ["delay_trend", "PAY_1 menos el promedio de PAY_2 a PAY_6: positivo = el atraso está empeorando", "meses",
     "Derivada", "Sí"],
    ["incumple", "Objetivo: 1 si el cliente incumplió su pago en octubre de 2005 (el dataset lo llama "
                 "«default payment next month»)", "0 / 1", "Objetivo", "—"],
], columns=["Variable", "Qué es", "Valores / unidad", "Tipo", "¿Entra al modelo?"])

TERMINOS = pd.DataFrame([
    ["Incumplimiento de pago a un mes", "No pagar lo exigido en el mes siguiente", "Sí, es lo que mide este dataset"],
    ["Mora", "Atraso en el pago, medido en días o meses (30, 60, 90…)",
     "Parcialmente, el objetivo parece ser un atraso de un mes o más"],
    ["Default regulatorio (Basilea)", "90 días de atraso, o improbabilidad de que el deudor pague", "No"],
    ["Castigo (charge-off)", "El banco da de baja la deuda por considerarla irrecuperable", "No"],
], columns=["Concepto", "Definición habitual", "¿Lo predice este modelo?"])

REFERENCIAS = """
- Yeh, I. C. y Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480. Dataset: UCI Machine Learning Repository, «Default of Credit Card Clients».
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
- Chen, T. y Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.
- Lundberg, S. M. y Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 30*.
"""


def render(meta: dict):
    d, ev = meta.get("data", {}), meta.get("evidence", {})
    sel, tm = meta.get("selection", {}), meta["test_metrics"]
    cm, ca, thr = meta["cost_miss"], meta["cost_alert"], meta["threshold"]
    model = meta["model_name"]
    pct = lambda x, k=1: f"{100 * x:.{k}f}%"

    st.header("Sobre el proyecto")

    st.subheader("1. ¿Qué es y para qué sirve?")
    st.markdown(
        "El proyecto estima, para un cliente de tarjeta de crédito, la **probabilidad de que incumpla su pago el mes "
        "siguiente (PI)** a partir de su historial de seis meses, y la convierte en una acción sugerida: "
        "**alerta preventiva**. El proyecto está pensado como herramienta de **alerta temprana**, para priorizar "
        "o revisar antes de que el atraso se agrave. Es una ayuda para que una persona decida."
        "\n\n"
        "**Qué no es:** no decide a quién prestar (originación), no estima la probabilidad de default que se usa para "
        "provisiones y no mide pérdidas. El porqué está en la sección siguiente.")

    st.subheader("2. ¿Qué mide la variable objetivo?")
    st.markdown(
        "La variable objetivo del dataset es *default payment next month*: 1 si el cliente incumplió su pago en el mes "
        "siguiente (octubre de 2005) y 0 si no. La documentación revisada en Kaggle y el paper de Yeh, I. C. y Lien, C. H. (2009), " 
        "no precisa qué cuenta como incumplimiento, así que aquí esta variable es interpretada como " 
        "**incumplimiento de pago a un mes** y no *default*, porque son conceptos distintos.")
    st.dataframe(TERMINOS, hide_index=True)
    if ev:
        st.markdown(
            f"**Evidencia en los datos.** El {pct(ev['pct_al_dia'])} de los clientes estaba al día en septiembre "
            f"(`PAY_1` ≤ 0) y aun así el {pct(ev['tasa_en_al_dia'])} de ellos tiene objetivo = 1; el "
            f"{pct(ev['pct_positivos_al_dia'])} de todos los casos positivos venía de clientes al día. Como el atraso "
            "avanza como máximo un mes por mes, eso es incompatible con un objetivo de 3 o más meses de atraso y "
            "coherente con un incumplimiento de un mes. Es una inferencia a partir de los datos, no una definición "
            f"oficial. Solo el {pct(ev['pct_atraso_3m'])} de los clientes tenía 3 o más meses de atraso en septiembre.")
        t = pd.DataFrame(ev["por_pay1"]).rename(columns={
            "PAY_1": "Estado de pago en septiembre (PAY_1)", "clientes": "Clientes", "tasa": "% con objetivo = 1"})
        t["% con objetivo = 1"] = (100 * t["% con objetivo = 1"]).round(1)
        st.dataframe(t, hide_index=True)
    st.markdown(
        f"**Consecuencias.** La tasa de {pct(d.get('prevalence', 0.22))} no es una tasa de default. Un incumplimiento a "
        "un mes no es necesariamente una pérdida: muchos clientes que financian su saldo son rentables, y lo que se "
        "busca anticipar es el deterioro (pasar de pagar a dejar de pagar) para intervenir a tiempo. Si esa "
        "intervención conviene depende de costos que cada institución debe estimar (sección 6).")

    st.subheader("3. Datos y variables")
    st.markdown(
        f"Fuente: *Default of Credit Card Clients* (UCI / Kaggle), clientes de tarjeta de crédito de un banco de Taiwán; "
        f"historial de abril a septiembre de 2005 y objetivo en octubre de 2005. Se usó el archivo `{d.get('file', 'UCI_Credit_Card.csv')}`: "
        f"{d.get('n_raw', 0):,} filas, {d.get('n_dup', 0)} duplicados eliminados, {d.get('n_total', 0):,} clientes "
        f"({d.get('n_train', 0):,} para entrenar y {d.get('n_test', 0):,} para el test).")
    st.dataframe(VARIABLES, hide_index=True)
    st.caption("Sexo y estado civil se excluyen del modelo y se usan solo para auditar si el modelo trata distinto a "
               "algún grupo. La edad sí entra, aunque su uso es discutible en algunos contextos regulatorios.")

    st.subheader("4. Los algoritmos")
    st.markdown(f"Se compararon cuatro familias de modelos. **Modelo final: {model}.**")
    with st.expander("Regresión logística (línea base)"):
        st.markdown(
            "Modela el logaritmo de las probabilidades (log-odds) como una combinación lineal de las variables: "
            "`P(incumple) = 1 / (1 + e^−(β₀ + β₁x₁ + …))`. Cada coeficiente indica cuánto cambian las log-odds al subir "
            "una desviación estándar de esa variable. Es el estándar histórico de las scorecards porque es transparente y "
            "estable; su límite es que solo captura efectos lineales y aditivos. Se usó con regularización L2 y variables "
            "estandarizadas, ajustando la intensidad (`C`). Sirve de referencia: ¿cuánto aportan los modelos complejos?")
    with st.expander("Random Forest"):
        st.markdown(
            "Entrena cientos de árboles de decisión, cada uno con una muestra bootstrap de los clientes y, en cada "
            "división, un subconjunto aleatorio de variables; la probabilidad es el promedio de los árboles. Promediar "
            "árboles poco correlacionados reduce la varianza (el sobreajuste de un árbol individual) y captura "
            "interacciones y no linealidades sin transformar las variables. Es robusto a valores extremos y no necesita "
            "escalado. Hiperparámetros ajustados: `n_estimators`, `max_depth`, `min_samples_leaf` "
            "y `max_features`.")
    with st.expander("Gradient boosting: XGBoost y LightGBM"):
        st.markdown(
            "Construyen árboles pequeños uno tras otro: cada árbol nuevo corrige los errores (el gradiente de la pérdida "
            "logarítmica) del conjunto anterior y su aporte se reduce con una tasa de aprendizaje. Con regularización "
            "(`reg_lambda`, tamaño mínimo de hoja, submuestreo de filas y columnas) suelen ser los más precisos en datos "
            "tabulares, a costa de más hiperparámetros. XGBoost crece los árboles por niveles (`max_depth`) y LightGBM por "
            "hojas (`num_leaves`); ambos usan histogramas para buscar cortes más rápido. En este problema la diferencia "
            "práctica entre ambos es pequeña.")
    with st.expander("Validación, ajuste y selección"):
        folds, n_iter = meta.get("cv", {}).get("folds", 5), meta.get("cv", {}).get("n_iter", {})
        st.markdown(
            f"- **Partición:** el test se separa al inicio y se usa una sola vez.\n"
            f"- **Validación cruzada estratificada** de {folds} folds sobre el train: conserva la proporción de "
            "incumplidores en cada fold. Para la comparación inicial se repite dos veces.\n"
            f"- **Búsqueda aleatoria de hiperparámetros** (combinaciones por modelo: {n_iter}), optimizando AUC.\n"
            "- **Confirmación con otra partición** de los mejores de cada familia, para no depender del azar de una sola.\n"
            "- **Regla de 1 error estándar:** entre los modelos cuyo AUC medio está a menos de un error estándar del mejor, "
            "se elige el más simple. El orden de simplicidad es un juicio del modelador y está en el código (`COMPLEXITY`).")
        if sel.get("comparison"):
            c = pd.DataFrame(sel["comparison"]).rename(columns={
                "modelo": "Modelo", "auc_base": "AUC sin ajuste", "auc_ajustado": "AUC ajustado (CV)",
                "auc_std": "± DE entre folds", "ks": "KS", "logloss": "Log-loss"})
            st.dataframe(c.round(4), hide_index=True)
            st.caption(f"Mejor AUC medio: {sel.get('best')} (error estándar {sel.get('se', 0):.4f}). "
                       f"Dentro de 1 error estándar: {', '.join(sel.get('eligible', []))}. Elegido: {model}.")
    with st.expander("Calibración (Platt)"):
        st.markdown(
            "Un modelo puede ordenar bien a los clientes y aun así dar probabilidades poco confiables. La calibración de "
            "Platt ajusta una regresión logística sobre la salida del modelo (en escala logit) usando predicciones "
            "*out-of-fold*. Es una transformación creciente: **no cambia el orden** (AUC y KS iguales), solo corrige el nivel "
            "de las probabilidades. Importa porque la regla de decisión por costos supone probabilidades correctas.")
    with st.expander("Interpretabilidad (SHAP)"):
        st.markdown(
            "SHAP descompone la predicción de cada cliente en el aporte de cada variable respecto a un valor base (el "
            "promedio del portafolio de entrenamiento), con base en los valores de Shapley de la teoría de juegos. Los aportes "
            "suman la predicción del modelo antes de calibrar y, en árboles, se calculan de forma exacta (TreeSHAP). Rojo o "
            "positivo: sube la probabilidad; verde o negativo: la baja. Son explicaciones **del modelo**, no relaciones "
            f"causales. Escala de los aportes en este modelo: {meta.get('shap', {}).get('scale', '—')}.")
    with st.expander("Métricas: cómo leerlas"):
        st.markdown(
            "- **AUC:** probabilidad de que un cliente que incumple reciba una PI mayor que uno que no; 0,5 es azar.\n"
            "- **Gini:** 2·AUC − 1.\n"
            "- **KS:** máxima distancia entre las distribuciones acumuladas de PI de incumplidores y no incumplidores.\n"
            "- **PR-AUC:** área bajo la curva precisión–recall; su referencia es la tasa base de incumplimiento.\n"
            "- **Brier y log-loss:** calidad de las probabilidades (menor es mejor).\n"
            "- **Captura y precisión de la alerta:** qué parte de los incumplidores se alerta y qué parte de las alertas "
            "acierta.")

    st.subheader("5. ¿Cómo se entrenó?")
    st.markdown(
        "1. Limpieza: duplicados fuera; códigos de educación sin documentar agrupados; facturas negativas tratadas como "
        "uso de línea cero (saldo a favor).\n"
        "2. Variables derivadas calculadas cliente por cliente (sin mezclar información entre train y test), dentro de un "
        "`Pipeline` de scikit-learn.\n"
        "3. Comparación, ajuste y selección de modelos (sección 4).\n"
        "4. Calibración de las probabilidades.\n"
        "5. Umbral de alerta por costos y contraste con un barrido empírico "
        f"({pct(meta.get('threshold_empirical', thr))} frente a {pct(thr)} teórico).\n"
        "6. Evaluación final en test, con intervalos de confianza por bootstrap, deciles de riesgo, auditoría por grupo y "
        "sensibilidad al costo.\n"
        "7. Exportación del modelo completo (limpieza, variables, modelo y calibrador) en un solo archivo, que usa esta app.")

    st.subheader("6. Costos y umbral de alerta")
    st.markdown(
        "Para convertir la PI en una acción hay que decir cuánto cuesta equivocarse en cada dirección. Son dos números "
        "**relativos** (no son dinero) y en este proyecto son **supuestos ilustrativos**:\n\n"
        f"- **Costo de no alertar** a un cliente que luego incumple: `{cm:g}`. Es intervenir tarde o nunca.\n"
        f"- **Costo de una alerta innecesaria** (el cliente habría pagado): `{ca:g}`. Es el contacto y la fricción con un "
        "buen cliente.")
    st.latex(r"\text{alertar} \iff PI\cdot c_{\text{falta}} > (1-PI)\cdot c_{\text{alerta}} \iff "
             r"PI > \frac{c_{\text{alerta}}}{c_{\text{alerta}}+c_{\text{falta}}} = \frac{1}{1+X}, \qquad "
             r"X=\frac{c_{\text{falta}}}{c_{\text{alerta}}}")
    st.markdown(
        f"Con X = {cm / ca:g} el umbral es **{pct(thr)}**: se alerta si la PI es al menos esa. X se lee como cuántas alertas "
        "innecesarias se aceptan, en el margen, por cada incumplidor detectado. La regla solo es óptima si la PI está "
        "calibrada.\n\n"
        "Estos costos **no** son la pérdida de capital de un default: el objetivo es un incumplimiento a un mes. Son "
        "uniformes por cliente (no ponderan por monto) y en la práctica el umbral suele fijarse también por la capacidad "
        "del equipo de contacto, por ejemplo alertar al 10% de mayor riesgo. La pestaña «Desempeño del modelo» muestra "
        "cuánto cambian el umbral y el beneficio según X.")

    st.subheader("7. Cómo leer el simulador")
    st.markdown(
        "- **PI:** probabilidad calibrada de incumplir el pago el mes siguiente, según el modelo.\n"
        "- **Acción sugerida:** alerta preventiva si la PI alcanza el umbral que fija el costo X que elijas.\n"
        "- **Gráfico de factores:** qué variables empujan la PI de ese cliente hacia arriba (rojo) o hacia abajo (verde).\n"
        "- **Avisos de rango:** si un dato cae fuera de lo visto en el entrenamiento, la estimación puede ser poco fiable.")

    st.subheader("8. Limitaciones")
    st.markdown(
        "- El objetivo es un **incumplimiento a un mes**, no un default ni una pérdida: no sirve para originación ni "
        "provisiones.\n"
        "- Datos de tarjetas de un banco de Taiwán de 2005: la metodología se transfiere, las cifras no."
        "- Sin validación fuera de tiempo (una sola ventana, sin fechas) y con sesgo de selección: solo se observan "
        "clientes que ya tenían crédito.\n"
        "- Sexo y estado civil no son variables del modelo, pero otras variables pueden actuar como proxies; la "
        "auditoría por grupo es introductoria y algunos grupos son pequeños.\n"
        "- Los costos del umbral son supuestos. No se midió cuánto aportan las variables derivadas frente a las originales.\n"
        "- Los valores 0 y −2 de `PAY_x` no están documentados por la fuente.")

    st.subheader("9. Ficha técnica")
    st.dataframe(pd.DataFrame([
        ["Modelo final", model],
        ["Hiperparámetros", ", ".join(f"{k} = {v}" for k, v in meta["best_params"].items())],
        ["AUC en test", f"{tm['auc']:.3f} (IC95% {tm['auc_ci95'][0]:.3f} a {tm['auc_ci95'][1]:.3f})"],
        ["KS / Gini / PR-AUC / Brier", f"{tm['ks']:.3f} / {tm['gini']:.2f} / {tm['pr_auc']:.2f} / {tm['brier']:.3f}"],
        ["Umbral de alerta", f"PI ≥ {pct(thr)} (X = {cm / ca:g} : 1)"],
        ["Versiones", ", ".join(f"{k} {v}" for k, v in meta["versions"].items())],
    ], columns=["Dato", "Valor"]), hide_index=True)

    st.subheader("10. Reproducir el proyecto")
    st.code("pip install -r requirements.txt\n"
            "python train_model.py          # entrena y guarda models/credit_model.joblib (usa UCI_Credit_Card.csv)\n"
            "streamlit run app.py           # esta aplicación\n"
            "python build_report.py         # opcional: genera reporte_modelo_riesgo.html", language="bash")

    st.subheader("Referencias")
    st.markdown(REFERENCIAS)

"""
build_report.py - Genera reporte_modelo_riesgo.html a partir de lo que guardó train_model.py.
Todas las cifras salen de outputs/metricas.json y de las tablas CSV; no hay números escritos a mano.
Uso: python build_report.py    (después de python train_model.py)
"""
import base64
import json
from pathlib import Path

import pandas as pd

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()
O = BASE / "outputs"
meta = json.load(open(O / "metricas.json", encoding="utf-8"))
tm, d, ev, sel = meta["test_metrics"], meta["data"], meta["evidence"], meta["selection"]
cm, ca, thr, model = meta["cost_miss"], meta["cost_alert"], meta["threshold"], meta["model_name"]
f = lambda x, k=3: f"{x:.{k}f}".replace(".", ",")
pc = lambda x, k=1: f"{100 * x:.{k}f}%".replace(".", ",")
nf = lambda n: f"{int(n):,}".replace(",", ".")


def tbl(head, rows, hl=None):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join(f"<tr{' class=hl' if r[0] == hl else ''}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="sc"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def img(name):
    p = O / name
    return f'<img alt="{name}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">' if p.exists() else ""


# --- tablas ---------------------------------------------------------------------------
t_cmp = tbl(["Modelo", "AUC sin ajuste (CV)", "AUC ajustado (CV, ± DE)", "KS", "Log-loss"],
            [[r["modelo"], f(r["auc_base"]), f"{f(r['auc_ajustado'])} ± {f(r['auc_std'])}", f(r["ks"]), f(r["logloss"], 4)]
             for r in sel["comparison"]], hl=model)
pol = pd.read_csv(O / "politicas_test.csv", index_col=0)
t_pol = tbl(["Política (test)", "Tasa de alerta", "Captura de incumplidores", "Precisión de la alerta", "Costo por cliente"],
            [[i, pc(r.tasa_alerta), pc(r.captura), pc(r.precision_alerta) if r.tasa_alerta > 0 else "–", f(r.costo_por_cliente)]
             for i, r in pol.iterrows()], hl=model)
dec = pd.read_csv(O / "deciles_test.csv")
t_dec = tbl(["Decil de riesgo", "Clientes", "PI media", "Incumplimiento observado"],
            [[int(r.decil), int(r.clientes), pc(r.pi_media), pc(r.incumplimiento_observado)] for r in dec.itertuples()])
eq = pd.read_csv(O / "equidad_test.csv")
t_eq = tbl(["Variable", "Grupo", "n", "Incumplimiento observado", "PI media", "Tasa de alerta", "AUC"],
           [[r.variable, r.grupo, int(r.n), pc(r.incumplimiento_observado), pc(r.pi_media), pc(r.tasa_alerta), f(r.auc)]
            for r in eq.itertuples()])
sens = pd.read_csv(O / "sensibilidad_costo.csv")
t_sens = tbl(["Costo X : 1", "Umbral", "Tasa de alerta", "Captura", "Precisión", "Costo/cliente modelo",
              "Costo/cliente mejor trivial", "Reducción"],
             [[f"{int(r.X)} : 1", pc(r.umbral), pc(r.tasa_alerta), pc(r.captura), pc(r.precision_alerta), f(r.costo_modelo),
               f(r.costo_mejor_trivial), pc(r.reduccion)] for r in sens.itertuples()], hl=f"{int(round(cm / ca))} : 1")
t_pay = tbl(["Estado de pago en septiembre (PAY_1)", "Clientes", "% con objetivo = 1"],
            [[r["PAY_1"], nf(r["clientes"]), pc(r["tasa"])] for r in ev["por_pay1"]])

triv_none, triv_all = pol.loc["No alertar a nadie", "costo_por_cliente"], pol.loc["Alertar a todos", "costo_por_cliente"]
best_triv, m_cost = min(triv_none, triv_all), pol.loc[model, "costo_por_cliente"]
saving, prev = 1 - m_cost / best_triv, d["prevalence"]
bp = ", ".join(f"{k} = {v}" for k, v in meta["best_params"].items())
cal = meta["calibration"]
fast = ('<p class="warn">Este reporte viene de una corrida rápida (<code>--fast</code>): no usar sus cifras como resultado.</p>'
        if meta["cv"]["fast"] else "")

CSS = """:root{--bg:#fff;--fg:#1d2330;--mut:#5b6475;--line:#dde1ea;--card:#f5f7fb;--acc:#2f5fd0}
@media(prefers-color-scheme:dark){:root{--bg:#14171f;--fg:#e6e9f0;--mut:#9aa3b5;--line:#2c3242;--card:#1b2030;--acc:#7da2ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 20px 64px}h1{font-size:28px;margin:0 0 4px}
h2{font-size:21px;margin:40px 0 10px;padding-top:12px;border-top:1px solid var(--line)}h3{font-size:16px;margin:22px 0 6px}
.sub,.note{color:var(--mut)}.note{font-size:14px}.warn{background:var(--card);border-left:3px solid #c23b3b;padding:8px 12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:10px;margin:14px 0}
.card{background:var(--card);border-radius:8px;padding:10px 12px}.card b{display:block;font-size:22px}.card span{color:var(--mut);font-size:13px}
.sc{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0 4px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:600}
tr.hl td{font-weight:600;background:var(--card)}.eq{background:var(--card);border-left:3px solid var(--acc);padding:10px 14px;margin:12px 0;font-family:ui-monospace,Consolas,monospace;font-size:14.5px;overflow-x:auto;line-height:1.9}
img{max-width:100%;border-radius:6px;margin:10px 0}code{background:var(--card);padding:1px 5px;border-radius:4px}"""

HTML = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alerta temprana de incumplimiento de pago: costos y resultados</title><style>{CSS}</style></head><body><main>
<h1>Alerta temprana de incumplimiento de pago: costos y resultados</h1>
<p class="sub">Dataset UCI <i>Default of Credit Card Clients</i> (Taiwán, 2005) · entrenado el {meta['trained_at'][:10]} · modelo final: {model}</p>{fast}
<p>Modelo que estima la probabilidad de que un cliente de tarjeta <b>incumpla su pago el mes siguiente</b> (PI), con probabilidades calibradas y un umbral de alerta derivado de una matriz de costos. En el test ({nf(d['n_test'])} clientes, usado una sola vez) alcanza AUC {f(tm['auc'])} y KS {f(tm['ks'])}. Con costos {cm:g} : {ca:g}, alertar si PI ≥ {pc(thr)} reduce el costo esperado por cliente en {pc(saving)} frente a la mejor política trivial.</p>
<div class="cards"><div class="card"><b>{f(tm['auc'])}</b><span>AUC · IC95% {f(tm['auc_ci95'][0])} a {f(tm['auc_ci95'][1])}</span></div>
<div class="card"><b>{f(tm['ks'])}</b><span>KS · IC95% {f(tm['ks_ci95'][0])} a {f(tm['ks_ci95'][1])}</span></div>
<div class="card"><b>{f(tm['gini'], 2)}</b><span>Gini</span></div><div class="card"><b>{f(tm['pr_auc'], 2)}</b><span>PR-AUC (base {f(prev, 2)})</span></div>
<div class="card"><b>{f(tm['brier'])}</b><span>Brier (PI calibrada)</span></div><div class="card"><b>{pc(saving)}</b><span>menos costo vs. mejor trivial</span></div></div>

<h2>1. Qué se hizo</h2>
<ul><li><b>Datos:</b> {nf(d['n_raw'])} filas, {nf(d['n_total'])} tras quitar {d['n_dup']} duplicados; incumplimiento de {pc(prev)}. Partición 80/20 estratificada: {nf(d['n_train'])} en train y {nf(d['n_test'])} en test.</li>
<li><b>Variables:</b> las originales más uso de la línea, fracción pagada de la factura anterior, meses con atraso, atraso máximo y tendencia del atraso. Sexo y estado civil no entran al modelo; solo se usan para auditar.</li>
<li><b>Selección:</b> logística, Random Forest, XGBoost y LightGBM. Comparación con CV repetida, búsqueda aleatoria sobre AUC ({', '.join(f'{k}: {v}' for k, v in meta['cv']['n_iter'].items())} combinaciones), reconfirmación con otra partición y regla de 1 error estándar.</li>
<li><b>Probabilidades:</b> calibración de Platt sobre predicciones out-of-fold. <b>Decisión:</b> umbral de alerta por costos (sección 3), contrastado con un barrido empírico.</li>
<li><b>Evaluación:</b> test usado una vez; intervalos por bootstrap (500 remuestreos).</li></ul>

<h2>2. Qué predice realmente el objetivo</h2>
<p>El dataset llama a su objetivo <i>default payment next month</i>, sin definir qué cuenta como incumplimiento. Aquí se le llama <b>incumplimiento de pago a un mes</b> y no <i>default</i>: no es un default regulatorio (90 días de atraso o improbabilidad de pago) ni una pérdida.</p>
<p>La evidencia en los datos: el {pc(ev['pct_al_dia'])} de los clientes estaba al día en septiembre (<code>PAY_1</code> ≤ 0) y aun así el {pc(ev['tasa_en_al_dia'])} de ellos tiene objetivo = 1; el {pc(ev['pct_positivos_al_dia'])} de todos los positivos venía de clientes al día. Como el atraso avanza como máximo un mes por mes, eso es incompatible con un objetivo de 3 o más meses de atraso y coherente con un incumplimiento de un mes. Es una inferencia, no una definición oficial. Solo el {pc(ev['pct_atraso_3m'])} tenía 3 o más meses de atraso en septiembre.</p>{t_pay}
<p>Consecuencia: el caso de uso es la <b>gestión preventiva</b> (alertas tempranas, cobranza preventiva, gestión de líneas), no la originación ni las provisiones. Un incumplimiento a un mes no siempre es una pérdida: muchos clientes que financian su saldo son rentables, y lo que se busca anticipar es el deterioro para intervenir a tiempo.</p>

<h2>3. Los costos</h2>
<p>El modelo entrega una PI. Para convertirla en alertar o no alertar hay que decir cuánto cuesta equivocarse en cada dirección. Son dos números en unidades relativas (no en dinero), y aquí son supuestos ilustrativos:</p>
<ul><li><code>c_falta = {cm:g}</code>: costo de <b>no alertar</b> a un cliente que luego incumple (se interviene tarde o nunca).</li>
<li><code>c_alerta = {ca:g}</code>: costo de una <b>alerta innecesaria</b> (contacto y fricción con un cliente que habría pagado).</li></ul>
<div class="eq">costo esperado de no alertar = PI · c_falta<br>costo esperado de alertar = (1 − PI) · c_alerta<br>alertar ⇔ PI · c_falta &gt; (1 − PI) · c_alerta ⇔ <b>PI &gt; c_alerta / (c_alerta + c_falta) = 1/(1+X) = {pc(thr)}</b>, con X = c_falta / c_alerta</div>
<p>X se lee como cuántas alertas innecesarias se aceptan, en el margen, por cada incumplidor detectado. La regla solo es óptima si la PI está calibrada, y por eso se calibró. El barrido empírico sobre las predicciones out-of-fold del train dio un umbral de {pc(meta['threshold_empirical'])}, frente a {pc(thr)} teórico.</p>
<h3>De dónde sale el {cm:g} : {ca:g}</h3>
<p>Es un supuesto, no un dato del dataset. En una institución habría que cuantificar el costo de contactar y el valor de intervenir a tiempo (por ejemplo, la menor probabilidad de que el atraso se agrave). Estos costos <b>no</b> son la pérdida de capital de un default: el objetivo es un incumplimiento a un mes.</p>
<h3>Cómo se mide el resultado</h3>
<div class="eq">C(t) = (FN · c_falta + FP · c_alerta) / N &nbsp;&nbsp; FN = incumplidores no alertados · FP = alertados que habrían pagado</div>
<p>Referencias en test con {cm:g} : {ca:g}: no alertar a nadie cuesta {f(triv_none)} por cliente ({pc(prev)} × {cm:g}) y alertar a todos {f(triv_all)} ({pc(1 - prev)} × {ca:g}). El modelo se compara contra la de menor costo.</p>
<h3>Cautelas</h3>
<ul><li>Los costos son uniformes por cliente: no ponderan por exposición (límite o saldo).</li>
<li>El costo por cliente está en unidades relativas: sirve para comparar políticas, no para estimar ganancias.</li>
<li>En la práctica el umbral suele fijarse por la capacidad del equipo de contacto (por ejemplo, alertar al 10 % de mayor riesgo); la captura y la precisión de la tabla de políticas ayudan a ese análisis.</li>
<li>El resultado depende de X; la sección 5 muestra cuánto.</li></ul>

<h2>4. Resultados</h2>
<h3>Comparación de modelos (AUC en validación cruzada, solo train)</h3>{t_cmp}
<p class="note">Mejor AUC medio: {sel['best']} (error estándar {f(sel['se'], 4)}). Dentro de 1 error estándar: {', '.join(sel['eligible'])}. La regla de parsimonia, con un orden de complejidad que fijé antes de ver resultados, eligió {model}. Hiperparámetros: {bp}.</p>
<h3>Test</h3><p>Referencia: la regresión logística ajustada obtiene AUC {f(meta['lr_test']['auc'])} y KS {f(meta['lr_test']['ks'])} en test. Brier del modelo sin calibrar: {f(cal['sin calibrar']['brier'], 4)}; con Platt: {f(cal['Platt (sigmoide)']['brier'], 4)}; log-loss {f(cal['sin calibrar']['logloss'], 4)} frente a {f(cal['Platt (sigmoide)']['logloss'], 4)}.</p>
<h3>Políticas de decisión</h3>{t_pol}
<p class="note">Con {cm:g} : {ca:g}, el modelo alerta al {pc(pol.loc[model, 'tasa_alerta'])} de los clientes y captura el {pc(pol.loc[model, 'captura'])} de los incumplidores; el {pc(pol.loc[model, 'precision_alerta'])} de las alertas acierta. Costo por cliente: {f(m_cost)} frente a {f(best_triv)} de la mejor política trivial.</p>
<h3>Ordenamiento y calibración por decil</h3>{t_dec}
<h3>Curvas de evaluación (test)</h3>{img('evaluacion.png')}
<h3>Variables más influyentes (SHAP)</h3><p>Mayor impacto medio: {', '.join(f'<code>{k}</code>' for k, _ in meta['shap']['top'][:5])} (escala de los aportes: {meta['shap']['scale']}).</p>{img('shap_resumen.png')}
<h3>Auditoría por grupo</h3>{t_eq}
<p class="note">Sexo y estado civil no son variables del modelo. Razón mínima/máxima de tasas de alerta entre grupos: {', '.join(f'{k} {f(v, 2)}' for k, v in meta['fairness_ratio'].items())}. Algunos grupos son pequeños y no permiten concluir.</p>

<h2>5. Sensibilidad al costo</h2>
<p>Mismo modelo y mismo test, cambiando X. Cada fila se evalúa con su propia matriz de costos, así que los costos por cliente no son comparables entre filas; la reducción relativa sí indica cuánto depende el beneficio del supuesto.</p>{t_sens}

<h2>6. Límites</h2>
<ul><li>El objetivo es un incumplimiento a un mes, no un default ni una pérdida: no sirve para originación ni provisiones.</li>
<li>Datos de tarjetas de Taiwán de 2005: la metodología se transfiere, las cifras no. No usar para decisiones reales.</li>
<li>Sin validación fuera de tiempo (el dataset no trae fechas) y con sesgo de selección: solo se observan clientes que ya tenían crédito.</li>
<li>Los valores 0 y −2 de <code>PAY_x</code> no están documentados por la fuente; la lectura habitual (pago mínimo y sin consumo) es de la comunidad.</li>
<li>No se midió cuánto aportan las variables derivadas frente a las originales.</li></ul>
<p class="note">Fuente de los datos: Yeh, I. C. y Lien, C. H. (2009), <i>Expert Systems with Applications</i> 36(2), 2473–2480; UCI Machine Learning Repository.</p>
</main></body></html>"""
out = BASE / "reporte_modelo_riesgo.html"
out.write_text(HTML, encoding="utf-8")
print("OK", out, f"{out.stat().st_size / 1024:.0f} KB")

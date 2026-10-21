"""Funciones compartidas por notebook2 (auditoría), notebook3 (estimación causal)
y notebook4 (validación semisintética).

Estimador principal: Double ML parcialmente lineal (PLR, Chernozhukov et al., 2018)
    Y = theta * T + g(X) + e,   T = m(X) + v
con cross-fitting agrupado por edificio (ningún edificio aparece a la vez en
entrenamiento y evaluación) y errores estándar agrupados (cluster).
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RES = Path("resultados")
FIG = RES / "figuras"
TAB = RES / "tablas"

CLASE = {"miraflores": "clase1", "san-isidro": "clase1", "barranco": "clase1", "lince": "clase1",
         "magdalena": "clase2", "surquillo": "clase2", "jesus-maria": "clase2", "san-borja": "clase2",
         "surco": "clase2", "cercado": "clase3", "san-miguel": "clase3", "chorrillos": "clase3",
         "la-molina": "clase3"}

# Conjunto de ajuste principal (S1), fijado por el DAG antes de estimar
NUM = ["area_log", "dormitorios", "antig", "antig_na", "latitud", "longitud", "dias_pub"]
CAT = ["distrito", "tipo_ubic"]


# --------------------------------------------------------------------------- datos
def normalizar_texto(s):
    """Minúsculas, sin tildes ni signos: para detectar avisos re-publicados."""
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", s)).strip()


def motivo_filtro_principal(df):
    """Reglas de depuración del notebook1, aplicadas en el mismo orden.
    Devuelve, por fila, el primer motivo de exclusión (o None si se conserva)."""
    motivo = pd.Series(None, index=df.index, dtype=object)
    vivo = pd.Series(True, index=df.index)

    def marcar(mask, texto):
        nonlocal vivo
        m = mask & vivo
        motivo[m] = texto
        vivo &= ~m

    marcar((df.dormitorios > 6) | (df.area > 600) | (df.precioxm2 > 120) | (df.precioxm2 < 8),
           "F1 rangos (dorm>6, área>600, precio/m2 fuera de [8,120])")
    marcar((df.dormitorios <= 2) & (df.banos == 1) & (df.area >= 160), "F2 área incoherente (<=2 dorm, 1 baño, >=160 m2)")
    marcar((df.dormitorios <= 2) & (df.banos == 2) & (df.area >= 250), "F3 área incoherente (<=2 dorm, 2 baños, >=250 m2)")
    marcar(df.precio < 600, "F4 precio < 600")
    cnt = df[vivo].groupby(["dormitorios", "banos"]).size()
    raras = set(cnt[cnt <= 3].index)
    marcar(pd.Series([(d, b) in raras for d, b in zip(df.dormitorios, df.banos)], index=df.index),
           "F5 combinación dorm-baños rara (<=3 casos)")
    lim = df[vivo].groupby("distrito")["precioxm2"].apply(
        lambda x: x.quantile(.75) + 2.5 * (x.quantile(.75) - x.quantile(.25)))
    marcar(df.precioxm2 > df.distrito.map(lim), "F6 precio/m2 atípico en su distrito")
    return motivo


def pasa_filtro_minimo(df):
    """Filtro mínimo predefinido: solo errores groseros; no usa precio/m2 ni reglas por distrito."""
    return (df.area.between(15, 1000) & df.precio.between(300, 30000)
            & df.dormitorios.between(1, 6) & df.banos.between(1, 6))


def preparar_variables(df):
    """Variables de análisis (sin imputar con información de otras filas: antigüedad
    faltante se codifica 0 + indicador, así no hay fuga entre folds)."""
    d = df.copy()
    d["Y"] = np.log(d["precio"])
    d["T"] = d["banos"].astype(float)
    d["area_log"] = np.log(d["area"])
    d["antig_na"] = d["antiguedad"].isna().astype(int)
    d["antig"] = d["antiguedad"].fillna(0.0)
    d["dias_pub"] = d["dias_pub"].astype(float)
    d["distrito_clase"] = d["distrito"].map(CLASE)
    tipos = d["type"].value_counts()
    d["tipo_ubic"] = d["type"].where(d["type"].map(tipos) >= 30, "otro")
    d["banos_dorm"] = d["banos"] / d["dormitorios"]
    return d


def muestra(base, dedup=True, filtro="principal", excluir_aprox=False):
    """Selecciona la muestra analítica a partir de la base auditada (notebook2)."""
    d = base.copy()
    if "geo_incoherente" in d:
        d = d[d["geo_incoherente"] == 0]
    if dedup:
        d = d[d["es_duplicado"] == 0]
    if filtro == "principal":
        d = d[motivo_filtro_principal(d).isna()]
    elif filtro == "minimo":
        d = d[pasa_filtro_minimo(d)]
    if excluir_aprox:
        d = d[d["precision"] != "APPROXIMATE"]
    return preparar_variables(d).reset_index(drop=True)


def matriz(d, num=NUM, cat=CAT, extra=()):
    partes = [d[list(num) + list(extra)]]
    if cat:
        partes.append(pd.get_dummies(d[list(cat)], drop_first=True))
    return pd.concat(partes, axis=1).astype(float)


# --------------------------------------------------------------------------- learners
LEARNERS = {
    "HGB": lambda s: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                                   min_samples_leaf=20, l2_regularization=1.0, random_state=s),
    "GBR": lambda s: GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                               subsample=0.8, random_state=s),
    "RF": lambda s: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=s),
    "Lasso": lambda s: make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=s, max_iter=20000)),
}


def folds_agrupados(grupos, k=5, seed=SEED):
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(gkf.split(np.zeros(len(grupos)), groups=grupos))


def predecir_cf(learner, X, v, folds, seed):
    pred = np.empty(len(v))
    for tr, te in folds:
        m = LEARNERS[learner](seed)
        m.fit(X[tr], v[tr])
        pred[te] = m.predict(X[te])
    return pred


def se_score(psi, J, clusters=None):
    """EE del score ortogonal; con clusters usa la varianza agrupada (CR1)."""
    n = len(psi)
    if clusters is None:
        return np.sqrt(np.mean(psi ** 2) / n) / J
    s = pd.Series(psi).groupby(np.asarray(clusters)).sum().values
    G = len(s)
    return np.sqrt(G / (G - 1) * np.sum(s ** 2)) / (n * J)


def dml_plr(Y, T, X, grupos, clusters=None, learner="HGB", k=5, n_rep=1, seed=SEED):
    """DML-PLR con cross-fitting agrupado y repetido.

    grupos: unidad que no puede repartirse entre folds (edificio).
    clusters: dict nombre -> array para errores estándar agrupados.
    Con n_rep > 1, theta = mediana de las repeticiones y el EE incorpora la
    dispersión entre particiones (Chernozhukov et al., 2018, def. 3.3).
    """
    Y, T = np.asarray(Y, float), np.asarray(T, float)
    X = np.asarray(X, float)
    clusters = clusters or {}
    tipos = ["iid"] + list(clusters)
    thetas, ses, yhats, thats = [], {c: [] for c in tipos}, [], []
    for r in range(n_rep):
        folds = folds_agrupados(grupos, k, seed + r)
        yh = predecir_cf(learner, X, Y, folds, seed + r)
        th = predecir_cf(learner, X, T, folds, seed + r)
        ry, rt = Y - yh, T - th
        theta = (rt @ ry) / (rt @ rt)
        psi = (ry - theta * rt) * rt
        J = np.mean(rt ** 2)
        thetas.append(theta)
        ses["iid"].append(se_score(psi, J))
        for c, cl in clusters.items():
            ses[c].append(se_score(psi, J, cl))
        yhats.append(yh)
        thats.append(th)
    thetas = np.array(thetas)
    theta = float(np.median(thetas))
    se = {c: float(np.sqrt(np.median(np.array(v) ** 2 + (thetas - theta) ** 2))) for c, v in ses.items()}
    # residuos con las predicciones promediadas (todas fuera de fold) para diagnósticos
    ry = Y - np.mean(yhats, axis=0)
    rt = T - np.mean(thats, axis=0)
    eps = ry - theta * rt
    return {"theta": theta, "se": se, "thetas": thetas, "ry": ry, "rt": rt, "eps": eps,
            "r2_y": 1 - np.var(ry) / np.var(Y), "r2_t": 1 - np.var(rt) / np.var(T),
            "mae_y": float(np.mean(np.abs(ry))), "mae_t": float(np.mean(np.abs(rt))), "n": len(Y)}


def ols_efecto(Y, T, X=None, clusters=None):
    """OLS de Y sobre T (+X) con EE agrupado (CR1) o HC1."""
    import statsmodels.api as sm
    Z = np.column_stack([T] if X is None else [T, np.asarray(X, float)])
    Z = sm.add_constant(Z, has_constant="add")
    if clusters is not None:
        r = sm.OLS(np.asarray(Y, float), Z).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(clusters)[0]})
    else:
        r = sm.OLS(np.asarray(Y, float), Z).fit(cov_type="HC1")
    return float(r.params[1]), float(r.bse[1])


def pct(theta):
    return 100 * (np.exp(theta) - 1)


# --------------------------------------------------------------------------- heterogeneidad
def gate(rt, ry, segmento, clusters):
    """Efectos por grupo predefinido (GATE) en el modelo PLR, reutilizando los residuos
    fuera de fold de la muestra completa. Covarianza agrupada conjunta entre grupos
    (un mismo edificio puede aportar a varios grupos) y prueba de Wald de igualdad."""
    segmento = np.asarray(segmento)
    niveles = [g for g in pd.unique(segmento) if pd.notna(g)]
    niveles = sorted(niveles, key=str)
    n = len(rt)
    phi = np.zeros((n, len(niveles)))
    th = []
    for j, g in enumerate(niveles):
        m = segmento == g
        S = np.sum(rt[m] ** 2)
        t = np.sum(rt[m] * ry[m]) / S
        th.append(t)
        phi[m, j] = rt[m] * (ry[m] - t * rt[m]) / S
    th = np.array(th)
    sums = pd.DataFrame(phi).groupby(np.asarray(clusters)).sum().values
    G = sums.shape[0]
    V = G / (G - 1) * sums.T @ sums
    se = np.sqrt(np.diag(V))
    C = np.column_stack([np.ones(len(th) - 1), -np.eye(len(th) - 1)])  # theta_1 - theta_j = 0
    diff = C @ th
    wald = float(diff @ np.linalg.pinv(C @ V @ C.T) @ diff)
    from scipy import stats
    p = float(stats.chi2.sf(wald, len(th) - 1))
    zb = stats.norm.ppf(1 - 0.05 / (2 * len(th)))   # Bonferroni (simultáneo)
    tab = pd.DataFrame({"grupo": niveles, "n": [int(np.sum(segmento == g)) for g in niveles],
                        "theta": th, "ee": se, "ic_inf": th - 1.96 * se, "ic_sup": th + 1.96 * se,
                        "ic_sim_inf": th - zb * se, "ic_sim_sup": th + zb * se})
    return tab, wald, p


# --------------------------------------------------------------------------- sensibilidad
def sesgo_max(r2y, r2d, var_eps, var_rt):
    """Cota del sesgo por un confusor omitido U con R2 parcial r2y (con Y | T,X) y r2d
    (con T | X), en el modelo PLR (Chernozhukov, Cinelli et al., 2022; rho = 1)."""
    r2d = np.minimum(r2d, 0.999)
    return np.sqrt(r2y * r2d / (1 - r2d) * var_eps / var_rt)


def valor_robustez(theta, var_eps, var_rt, se=0.0, z=0.0):
    """RV: fuerza común (R2 con T y con Y) que lleva el efecto (o su límite del IC) a 0."""
    efecto = max(abs(theta) - z * se, 0.0)
    f2 = efecto ** 2 * var_rt / var_eps
    return 0.5 * (np.sqrt(f2 ** 2 + 4 * f2) - f2)


def benchmark(largo, corto):
    """Fuerza implícita de un bloque de confusores observados: cuánto explica de la
    varianza residual de T y de Y (R2 parciales) comparando el modelo largo con el corto."""
    r2d = max((np.var(corto["rt"]) - np.var(largo["rt"])) / np.var(corto["rt"]), 0.0)
    r2y = max((np.var(corto["eps"]) - np.var(largo["eps"])) / np.var(corto["eps"]), 0.0)
    return r2d, r2y


# --------------------------------------------------------------------------- espacial
def coords_km(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = (lon - lon.mean()) * 111.32 * np.cos(np.deg2rad(lat.mean()))
    y = (lat - lat.mean()) * 110.57
    return np.column_stack([x, y])


def vecinos_knn(lat, lon, k=10):
    P = coords_km(lat, lon)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P)
    out = np.empty((len(P), k), dtype=int)
    for i, fila in enumerate(idx):
        fila = fila[fila != i]
        out[i] = fila[:k]
    return out


def moran_i(z, vecinos, nperm=999, seed=SEED):
    """I de Moran con pesos k-vecinos estandarizados por fila y p-valor por permutación."""
    z = np.asarray(z, float) - np.mean(z)
    den = np.sum(z ** 2)
    I = np.sum(z * z[vecinos].mean(axis=1)) / den
    rng = np.random.default_rng(seed)
    perm = np.empty(nperm)
    for b in range(nperm):
        zp = rng.permutation(z)
        perm[b] = np.sum(zp * zp[vecinos].mean(axis=1)) / den
    p = (1 + np.sum(perm >= I)) / (nperm + 1)
    return float(I), float(p), float(perm.mean())


# --------------------------------------------------------------------------- salidas
def _esc(x):
    s = str(x)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def guardar_tabla(df, nombre, decimales=3, origen=""):
    """Guarda la tabla como CSV y como tabular LaTeX (booktabs) para el manuscrito."""
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{nombre}.csv", index=False)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:,.{decimales}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return _esc(v)

    alin = "".join("l" if df[c].dtype == object else "r" for c in df.columns)
    lineas = [f"% Generado por {origen} -> {nombre}.csv", r"\begin{tabular}{" + alin + "}", r"\toprule",
              " & ".join(_esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lineas += [" & ".join(fmt(v) for v in fila) + r" \\" for fila in df.itertuples(index=False)]
    lineas += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"{nombre}.tex").write_text("\n".join(lineas), encoding="utf-8")
    return df

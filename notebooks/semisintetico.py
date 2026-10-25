"""Validación semisintética (OE6 · R6): DGP anclado en las covariables reales.

    T* = m0(X) + a*sd_v*U + sqrt(1-a^2)*sd_v*z_T + c_T[celda]   ;  T = clip(round(T*), 1, 5)
    Y  = g0(X) + theta(X)*T + b*sd_e*U + sqrt(1-b^2)*sd_e*z_Y + c_Y[celda]

U ~ N(0,1) no se entrega a los estimadores (confusor omitido). m0 y g0 se ajustan una vez
sobre los datos reales; el efecto verdadero theta(X) lo fija el escenario.
"""
import numpy as np
import pandas as pd

from causal_utils import benchmark_irm, dml_irm, dml_plr, gate, ols_efecto, sesgo_max

ESCENARIOS = {
    "E1 Sin U (efecto homogéneo)": dict(r2u=0.00, het=False, tau=0.0),
    "E2 U débil (R² ≈ 0,02)": dict(r2u=0.02, het=False, tau=0.0),
    "E3 U moderado (R² ≈ 0,07)": dict(r2u=0.07, het=False, tau=0.0),
    "E4 U fuerte (R² ≈ 0,15)": dict(r2u=0.15, het=False, tau=0.0),
    "E5 Efecto heterogéneo por clase": dict(r2u=0.00, het=True, tau=0.0),
    "E6 Dependencia espacial (bloques 1 km)": dict(r2u=0.00, het=False, tau=0.6),
}
THETA0 = 0.05
THETA_CLASE = {"clase1": 0.03, "clase2": 0.05, "clase3": 0.08}


def theta_x(clase, het):
    if not het:
        return np.full(len(clase), THETA0)
    return np.array([THETA_CLASE[c] for c in clase])


def simular(ins, esc, rng):
    """Genera (T, Y, U) para un escenario. ins: dict con m0, g0, sd_v, sd_e, clase, celda_codes."""
    p = ESCENARIOS[esc]
    n = len(ins["m0"])
    a = b = np.sqrt(p["r2u"])
    U = rng.standard_normal(n)
    ncel = ins["celda_codes"].max() + 1
    cT = rng.standard_normal(ncel)[ins["celda_codes"]] * p["tau"] * ins["sd_v"]
    cY = rng.standard_normal(ncel)[ins["celda_codes"]] * p["tau"] * ins["sd_e"]
    Ts = ins["m0"] + a * ins["sd_v"] * U + np.sqrt(1 - a ** 2) * ins["sd_v"] * rng.standard_normal(n) + cT
    T = np.clip(np.round(Ts), 1, 5)
    th = theta_x(ins["clase"], p["het"])
    Y = ins["g0"] + th * T + b * ins["sd_e"] * U + np.sqrt(1 - b ** 2) * ins["sd_e"] * rng.standard_normal(n) + cY
    return T, Y, U, th


def theta_plr_verdadero(ins, esc, n_mc=300, seed=0):
    """Estimando PLR verdadero bajo heterogeneidad: promedio de theta(X) ponderado por Var(T|X),
    calculado por Monte Carlo sobre la distribución de T dado X (sin U ni celdas)."""
    p = ESCENARIOS[esc]
    th = theta_x(ins["clase"], p["het"])
    if not p["het"]:
        return THETA0
    rng = np.random.default_rng(seed)
    n = len(ins["m0"])
    Ts = ins["m0"][None, :] + ins["sd_v"] * rng.standard_normal((n_mc, n))
    v = np.clip(np.round(Ts), 1, 5).var(axis=0)
    return float(np.sum(v * th) / np.sum(v))


def replica(ins, esc, rep, Xv, area, edificio, celda, seed=1000):
    rng = np.random.default_rng(seed + rep)
    T, Y, U, th = simular(ins, esc, rng)
    p = ESCENARIOS[esc]
    out = []
    b0, s0 = ols_efecto(Y, T, None, edificio)
    out.append(dict(estimador="OLS sin ajustar", theta=b0, se_iid=np.nan, se_edif=s0, se_celda=np.nan))
    b1, s1 = ols_efecto(Y, T, Xv, edificio)
    out.append(dict(estimador="Hedónico OLS (S1)", theta=b1, se_iid=np.nan, se_edif=s1, se_celda=np.nan))
    r = dml_plr(Y, T, Xv, edificio, {"edif": edificio, "celda": celda}, learner="HGB", n_rep=1, seed=seed + rep)
    out.append(dict(estimador="DML PLR (S1)", theta=r["theta"], se_iid=r["se"]["iid"], se_edif=r["se"]["edif"],
                    se_celda=r["se"]["celda"]))
    extra = {}
    if esc.startswith("E1"):
        Xb = np.column_stack([Xv, np.exp(Y) / area])          # control inadmisible: precio/m2
        rb = dml_plr(Y, T, Xb, edificio, {"edif": edificio}, learner="HGB", n_rep=1, seed=seed + rep)
        out.append(dict(estimador="DML + precio/m² (inadmisible)", theta=rb["theta"], se_iid=rb["se"]["iid"],
                        se_edif=rb["se"]["edif"], se_celda=np.nan))
    if p["r2u"] > 0:
        # fuerza real de U en esta réplica y cota de sensibilidad evaluada en esa fuerza
        rt, eps = r["rt"], r["eps"]
        r2d = np.corrcoef(rt, U)[0, 1] ** 2
        Ures = U - np.cov(U, rt)[0, 1] / np.var(rt) * rt
        r2y = np.corrcoef(eps, Ures)[0, 1] ** 2
        bmax = sesgo_max(r2y, r2d, np.var(eps), np.var(rt))
        se = r["se"]["edif"]
        extra = dict(r2d_real=r2d, r2y_real=r2y, cota=bmax,
                     cubre_sens=(r["theta"] - bmax - 1.96 * se <= THETA0 <= r["theta"] + bmax + 1.96 * se))
    if p["het"]:
        tb, _, pw = gate(r["rt"], r["ry"], ins["clase"], edificio)
        tb["verdad"] = tb["grupo"].map(THETA_CLASE)
        extra["gate"] = tb[["grupo", "theta", "ee", "verdad"]].to_dict("records")
        extra["p_wald"] = pw
    for o in out:
        o.update(escenario=esc, rep=rep)
    return out, dict(escenario=esc, rep=rep, **extra)


# --------------------------------------------------------------------------- tratamiento binario (cochera)
#   logit P(D=1|X,U) = logit m0(X) + a*U ;  Y = g0(X) + THETA_B*D + b*sd_e*U + sqrt(1-b^2)*sd_e*Z
ESCENARIOS_B = {
    "B1 Binario sin U": dict(a=0.0, b2=0.00),
    "B2 Binario con U moderado": dict(a=0.8, b2=0.07),
}
THETA_B = 0.10


def replica_binaria(ins, esc, rep, Xv, edificio, seed=5000):
    p = ESCENARIOS_B[esc]
    rng = np.random.default_rng(seed + rep)
    n = len(ins["logit_m0"])
    U = rng.standard_normal(n)
    D = rng.binomial(1, 1 / (1 + np.exp(-(ins["logit_m0"] + p["a"] * U))))
    b = np.sqrt(p["b2"])
    Y = ins["g0"] + THETA_B * D + b * ins["sd_e"] * U + np.sqrt(1 - b ** 2) * ins["sd_e"] * rng.standard_normal(n)
    out = []
    b1, s1 = ols_efecto(Y, D, Xv, edificio)
    out.append(dict(estimador="Hedónico OLS", theta=b1, se=s1))
    r = dml_plr(Y, D, Xv, edificio, {"edif": edificio}, n_rep=1, seed=seed + rep)
    out.append(dict(estimador="DML PLR", theta=r["theta"], se=r["se"]["edif"]))
    irm = dml_irm(Y, D, Xv, edificio, {"edif": edificio}, n_rep=1, seed=seed + rep)
    rc = irm["recortada"]
    out.append(dict(estimador="DML-IRM (soporte común)", theta=rc["theta"], se=rc["se"]["edif"]))
    extra = dict(pct_fuera=float(1 - rc["n"] / n))
    if p["a"] > 0:
        # fuerza real de U: modelo "largo" que sí observa U, comparado con el principal (definición de benchmark)
        largo = dml_irm(Y, D, np.column_stack([Xv, U]), edificio, {"edif": edificio}, n_rep=1, seed=seed + rep)["recortada"]
        cf_d, cf_y = benchmark_irm(largo, rc)
        bmax = sesgo_max(cf_y, cf_d, rc["sigma2"], 1 / rc["nu2"])
        se = rc["se"]["edif"]
        extra.update(cf_d=cf_d, cf_y=cf_y, cota=bmax,
                     cubre_sens=(rc["theta"] - bmax - 1.96 * se <= THETA_B <= rc["theta"] + bmax + 1.96 * se))
    for o in out:
        o.update(escenario=esc, rep=rep)
    return out, dict(escenario=esc, rep=rep, **extra)

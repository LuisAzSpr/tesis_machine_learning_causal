"""Genera las tablas compactas del manuscrito a partir de los CSV de resultados.

Uso (desde la carpeta informe/):  ../.venv/Scripts/python.exe tablas_manuscrito.py
Cada tabla indica en un comentario el CSV y el notebook de origen.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("../notebooks/resultados/tablas")
OUT = Path("tablas")
OUT.mkdir(exist_ok=True)
NB = {"R1": "notebook2_auditoria", "R3": "notebook3_estimacion_causal", "R4": "notebook3_estimacion_causal",
      "R5": "notebook3_estimacion_causal", "R7": "notebook3_estimacion_causal", "E7": "notebook3_estimacion_causal",
      "R6": "notebook4_semisintetico"}


def esc(s):
    s = str(s)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("²", r"$^2$"), ("θ", r"$\theta$"),
                 ("≈", r"$\approx$"), ("×", r"$\times$"), ("→", r"$\rightarrow$"), ("±", r"$\pm$"), ("α", r"$\alpha$")]:
        s = s.replace(a, b)
    return s


def num(v, d=3):
    if isinstance(v, str):
        return esc(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer() and abs(v) >= 100):
        return f"{int(round(v)):,}"
    return f"{v:,.{d}f}"


def tabla(df, nombre, origen, alin=None, dec=None):
    dec = dec or {}
    alin = alin or "l" + "r" * (df.shape[1] - 1)
    filas = [f"% Origen: {origen} (generado por informe/tablas_manuscrito.py)",
             r"\begin{tabular}{" + alin + "}", r"\toprule", " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        filas.append(" & ".join(num(r[c], dec.get(c, 3)) for c in df.columns) + r" \\")
    filas += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{nombre}.tex").write_text("\n".join(filas), encoding="utf-8")


def leer(n):
    return pd.read_csv(SRC / f"{n}.csv"), f"{NB[n[:2]]}.ipynb -> resultados/tablas/{n}.csv"


# R1 ---------------------------------------------------------------------------------
df, o = leer("R1_exclusiones")
df = df.rename(columns={"paso": "Paso", "N restante": "N restante", "excluidos": "Excluidos", "% del crudo": "% del crudo"})
tabla(df, "t_exclusiones", o, "lrrr", {"% del crudo": 1})

df, o = leer("R1_variantes_muestra")
tabla(df.rename(columns={"variante de muestra": "Variante de muestra"}), "t_variantes", o, "lr")

df, o = leer("R1_cobertura_distrito")
df = df[["distrito", "clase", "N", "% muestra", "precio mediano", "área mediana", "baños medio", "% ROOFTOP", "% antig. faltante"]]
df.columns = ["Distrito", "Clase", "N", "% muestra", "Precio med. (S/)", "Área med. (m²)", "Baños prom.", "% ROOFTOP", "% antig. faltante"]
tabla(df, "t_cobertura", o, "llrrrrrrr", {"% muestra": 1, "Baños prom.": 2, "% ROOFTOP": 1, "% antig. faltante": 1,
                                           "Precio med. (S/)": 0, "Área med. (m²)": 0})

df, o = leer("R1_agrupamientos")
df.columns = ["Agrupamiento", "N grupos", "Tamaño mediano", "Tamaño máximo", "% avisos en grupos >1"]
tabla(df, "t_agrupamientos", o, "lrrrr", {"Tamaño mediano": 1, "% avisos en grupos >1": 1})

df, o = leer("R1_diccionario")
df = df[["variable", "faltantes", "% faltante", "descripción"]]
df.columns = ["Variable", "Faltantes", "% faltante", "Descripción"]
tabla(df, "t_diccionario", o, "lrrp{8.2cm}", {"% faltante": 1})

# R3 ---------------------------------------------------------------------------------
df, o = leer("R3_principal")
t = pd.DataFrame({"Estimador": df["estimador"], "θ": df["theta"], "EE": df["EE (cluster edificio)"],
                  "IC 95% (θ)": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(df["IC95 inf"], df["IC95 sup"])],
                  "% por baño": df["% por baño"],
                  "IC 95% (%)": [f"[{a:.1f}, {b:.1f}]" for a, b in zip(df["% IC inf"], df["% IC sup"])],
                  "S/ al mes": df["S/ al mes (ref.)"]})
tabla(t, "t_principal", o, "lrrcrcr", {"θ": 4, "EE": 4, "% por baño": 1, "S/ al mes": 0})

df, o = leer("R3_learners")
df.columns = ["Learner", "R² E[Y|X]", "MAE E[Y|X]", "R² E[T|X]", "MAE E[T|X]", "θ", "EE", "% por baño"]
tabla(df, "t_learners", o, "lrrrrrrr", {"θ": 4, "EE": 4, "% por baño": 1})

# R4 ---------------------------------------------------------------------------------
df, o = leer("R4_especificaciones")
t = pd.DataFrame({"Cód.": df["código"], "Especificación": df["especificación"], "Tipo": df["tipo"],
                  "N": df["N"], "θ": df["theta"], "EE": df["EE"],
                  "IC 95%": ["--" if np.isnan(a) else f"[{a:.3f}, {b:.3f}]" for a, b in zip(df["IC95 inf"], df["IC95 sup"])],
                  "% por baño": df["% por baño"]})
tabla(t, "t_especificaciones", o, "llp{3.2cm}rrrcr", {"θ": 4, "EE": 4, "% por baño": 1})

# R5 ---------------------------------------------------------------------------------
df, o = leer("R5_sensibilidad")
df.columns = ["Escenario de confusor omitido", "R² con T", "R² con Y", "θ ajustado", "IC inf. ajustado", "% ajustado"]
tabla(df, "t_sensibilidad", o, "lrrrrr", {"θ ajustado": 4, "IC inf. ajustado": 4, "% ajustado": 1})
df, o = leer("R5_robustez")
df.columns = ["Indicador", "Valor"]
tabla(df, "t_robustez", o, "lr", {"Valor": 3})

# R7 ---------------------------------------------------------------------------------
df, o = leer("R7_gate")
t = pd.DataFrame({"Segmentación": df["segmentación"], "Grupo": df["grupo"], "n": df["n"], "θ_g": df["theta"], "EE": df["ee"],
                  "IC 95%": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(df["ic_inf"], df["ic_sup"])],
                  "IC simultáneo": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(df["ic_sim_inf"], df["ic_sim_sup"])],
                  "% por baño": df["% por baño"], "p Wald": df["p-valor Wald"]})
tabla(t, "t_gate", o, "llrrrccrr", {"θ_g": 4, "EE": 4, "% por baño": 1})

# Espacial ----------------------------------------------------------------------------
df, o = leer("E7_moran")
df.columns = ["Serie", "I de Moran", "E[I] (permutación)", "p-valor"]
tabla(df, "t_moran", o, "lrrr", {"I de Moran": 4, "E[I] (permutación)": 4})
df, o = leer("E7_inferencia")
df.columns = ["Inferencia", "EE", "IC inf.", "IC sup.", "Razón vs iid"]
tabla(df, "t_inferencia", o, "lrrrr", {"EE": 4, "IC inf.": 4, "IC sup.": 4, "Razón vs iid": 2})

# R6 ---------------------------------------------------------------------------------
if (SRC / "R6_simulacion.csv").exists():
    df, o = leer("R6_simulacion")
    t = pd.DataFrame({"Escenario": df["escenario"].str.split(" ").str[0], "Estimador": df["estimador"],
                      "θ verdadero": df["verdad"], "Sesgo": df["sesgo"], "RMSE": df["RMSE"],
                      "Cob. iid": df["cobertura EE iid"], "Cob. edificio": df["cobertura EE edificio"],
                      "Cob. celda": df["cobertura EE celda"], "Long. IC": df["long. media IC"]})
    tabla(t, "t_sim", o, "llrrrrrrr", {"θ verdadero": 4, "Sesgo": 4, "RMSE": 4, "Cob. iid": 0,
                                        "Cob. edificio": 0, "Cob. celda": 0, "Long. IC": 3})
    df, o = leer("R6_sensibilidad_calibracion")
    df["escenario"] = df["escenario"].str.split(" ").str[0]
    df.columns = ["Escenario", "R² real U-T", "R² real U-Y", "Sesgo medio DML", "Cota media", "% cubre IC", "% cubre IC + cota"]
    tabla(df, "t_sim_sens", o, "lrrrrrr", {"Sesgo medio DML": 4, "Cota media": 4, "% cubre IC": 0, "% cubre IC + cota": 0})
    df, o = leer("R6_gate")
    df.columns = ["Grupo", "θ verdadero", "Media GATE", "Sesgo", "RMSE", "Cobertura IC 95%", "Potencia Wald (%)"]
    tabla(df, "t_sim_gate", o, "lrrrrrr", {"Cobertura IC 95%": 0, "Potencia Wald (%)": 0})
print("Tablas escritas en", OUT.resolve())

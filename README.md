# Efecto causal de la cochera en el alquiler de departamentos en Lima

Tesis de Luis Angel Azaña Vega (Ciencia de la Computación, UNI). Asesor: Cesar Jesús Lara Ávila.

**Pregunta:** ¿cuánto aumenta el precio de oferta de un alquiler en Lima por incluir cochera, comparando departamentos equivalentes en todo lo demás?
**Resultado principal:** +7.5 % (IC 95 %: 3.7–11.4 %), unos S/ 199 al mes. Es casi nulo donde la cochera es habitual y llega a ≈ 18 % donde escasea.

## Notebooks (`notebooks/`), en orden de ejecución

| Notebook | Qué hace |
|---|---|
| `notebook2_auditoria` | Limpia el crudo: duplicados, geocodificación, solo ROOFTOP. Genera `csv/data_causal_base.csv` |
| `notebook2b_extraccion_ia` | Mide la cochera desde el texto (regex + `gpt-4.1-mini`, con caché). Necesita `API_KEY_OPENAI` en `.env` |
| `notebook3b_efecto_cochera` | **Resultado principal**: DML-IRM, soporte común, grilla, sensibilidad y GATE |
| `notebook3_estimacion_causal` | Análisis secundario: efecto de un baño adicional |
| `notebook4_semisintetico` | Validación con verdad conocida (≈ 8 min) |
| `notebook_resumen` | **Recorrido de toda la tesis** con el EDA resumido (lee los resultados; corre en segundos) |

Módulos: `causal_utils.py` (estimadores), `extraccion_ia.py` (prompt y API), `semisintetico.py` (simulación).
Trabajo previo (EDA): `notebook0_data_wrangling`, `notebook1`.

## Informe (`informe/`)

- `tesis.tex` → `tesis.pdf` (manuscrito), `presentacion.tex` → `presentacion.pdf`, `guia_explicativa.pdf`, `resumen_modelo_irm.pdf`.
- Para regenerar las tablas y compilar (desde `informe/`): `../.venv/Scripts/python.exe tablas_manuscrito.py` y luego `./tectonic.exe tesis.tex`.

Los datos (`data/`, `csv/`) y la clave (`.env`) no se versionan.

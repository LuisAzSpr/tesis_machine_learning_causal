# Solución — ML Causal sobre alquileres en Lima

- `notebooks/notebook_causal.ipynb`: notebook ejecutado (efecto causal de un baño adicional sobre el precio, Double ML + CATE + descubrimiento de estructura + robustez).
- `ESTADO_DEL_ARTE.md`: resumen de estado del arte (también incluido en el notebook).
- `figuras/`: 12 gráficos (incluye 5 grafos de dependencia: DAG propuesto, tipos de variables, equivalencia observacional, PC, LiNGAM).
- `_build_notebook.py`: regenera el notebook. Ejecutar: `py _build_notebook.py` y luego abrir/ejecutar el notebook desde esta carpeta.
- `requirements.txt`: dependencias.

Notas de entorno (Windows): si `numba/llvmlite` están bloqueados por la política de control de aplicaciones, el notebook usa un stub de `sparse`; `dowhy 0.8` solo se usa para identificación (su estimación falla con pandas 3), y los refutadores se ejecutan sobre el estimador DML.

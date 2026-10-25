"""Extracción de la variable cochera desde el texto de los avisos (OE5).

Flujo: regex -> ventanas de texto alrededor de cada mención -> enmascarar datos personales
-> OpenAI (salida JSON con esquema fijo, temperatura 0) -> caché local para no repetir llamadas.
La clave se lee de ../.env (API_KEY_OPENAI) y nunca se imprime.
"""
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

MODELO = "gpt-4.1-mini-2025-04-14"
VERSION_PROMPT = "v2"
VENTANA = 200
CACHE = Path("../csv/cochera_llm_cache.jsonl")   # contiene fragmentos de texto: carpeta privada (no se versiona)
# Precio de referencia de gpt-4.1-mini (USD por millón de tokens) para estimar el costo
USD_M_IN, USD_M_OUT = 0.40, 1.60

PATRON = re.compile(r"cocheras?|estacionamientos?|parqueos?|parking|garajes?|garages?|aparcamientos?|\bestac\.",
                    re.IGNORECASE)

PROMPT_SISTEMA = """Eres un asistente que extrae información estructurada de avisos de alquiler de departamentos en Lima, Perú.
Recibirás uno o varios fragmentos del texto de un aviso, separados por " [...] ", en los que se menciona cochera o estacionamiento.
Determina si el DEPARTAMENTO ofrecido cuenta con cochera/estacionamiento para el inquilino y si esta tiene un costo adicional al alquiler.

Reglas:
1. cochera = "si" si el departamento ofrecido cuenta con al menos una cochera o estacionamiento asignado o disponible para el inquilino (incluye "cochera opcional", "con o sin cochera" y "cochera en alquiler aparte").
2. cochera = "no" si el texto dice que no tiene cochera, o si solo menciona estacionamiento de visitas, en la calle, público, cercano, para bicicletas o motos, o si no permite afirmar que el departamento tiene cochera.
3. tipo:
   - "incluida": tiene cochera y NO se indica ningún pago adicional por ella, aunque no diga "incluida" de forma explícita. Ejemplos: "1 estacionamiento", "Cochera: si", "tiene 2 cocheras", "cuenta con cochera", "cochera" dentro de la lista de características del departamento, "ALQUILER CON COCHERA $700" (ese monto es el alquiler total que ya incluye la cochera).
   - "costo_extra": se indica que la cochera se paga aparte o tiene un monto propio adicional al alquiler ("cochera S/200 adicional", "alquiler de cochera: S/300", "posibilidad de alquilar un estacionamiento, costo adicional"), el aviso ofrece "con/sin cochera" con precios distintos, o dice que el departamento no tiene cochera pero se puede alquilar una en el mismo edificio.
   - "no_tiene": cochera = "no".
   - "ambigua": SOLO cuando no se puede saber si la cochera corresponde al departamento (por ejemplo, "el edificio cuenta con cocheras" sin decir si una es del departamento). No uses "ambigua" porque falte el precio de la cochera: en ese caso es "incluida".
4. precio_extra: 0 si tipo = "incluida"; el monto mensual adicional de la cochera (solo el número) si tipo = "costo_extra" y el monto aparece (si es un rango, usa el punto medio); null si no tiene cochera, si es ambigua o si el costo extra no indica monto. Nunca uses como precio_extra el alquiler total del departamento.
5. moneda: "PEN" (S/, soles) o "USD" ($, US$, dólares) del precio_extra; null si no hay monto.
6. incluida_explicita: true solo si el texto dice explícitamente que la cochera está incluida en el precio o alquiler; false en otro caso.
7. num_cocheras: número de cocheras del departamento (1 si dice "cochera" sin número); 0 si no tiene; null si es ambigua.
8. motivo: una frase breve (máximo 25 palabras) que cite la evidencia del texto.

No confundas el pago de mantenimiento, arbitrios, servicios o garantía con el precio de la cochera. No inventes información que no esté en el texto."""

ESQUEMA = {
    "name": "cochera_aviso",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cochera": {"type": "string", "enum": ["si", "no"]},
            "tipo": {"type": "string", "enum": ["incluida", "costo_extra", "no_tiene", "ambigua"]},
            "precio_extra": {"type": ["number", "null"]},
            "moneda": {"type": ["string", "null"], "enum": ["PEN", "USD", None]},
            "incluida_explicita": {"type": "boolean"},
            "num_cocheras": {"type": ["integer", "null"]},
            "motivo": {"type": "string"},
        },
        "required": ["cochera", "tipo", "precio_extra", "moneda", "incluida_explicita", "num_cocheras", "motivo"],
    },
}


def leer_clave(ruta="../.env"):
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        if linea.startswith("API_KEY_OPENAI="):
            return linea.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("No se encontró API_KEY_OPENAI en .env")


def enmascarar(texto):
    """Quita teléfonos, correos y enlaces antes de enviar el texto a la API."""
    texto = re.sub(r"\S+@\S+\.\S+", "[CORREO]", texto)
    texto = re.sub(r"(https?://|www\.)\S+", "[ENLACE]", texto)
    texto = re.sub(r"(\+?51[\s-]?)?9\d{2}[\s-]?\d{3}[\s-]?\d{3}", "[TELEFONO]", texto)
    texto = re.sub(r"\b\d{7,}\b", "[NUMERO]", texto)
    return texto


def ventanas(texto, ancho=VENTANA):
    """Fragmentos texto[x-ancho : x+ancho] alrededor de cada mención, uniendo los que se solapan."""
    texto = texto or ""
    tramos = []
    for m in PATRON.finditer(texto):
        ini, fin = max(0, m.start() - ancho), min(len(texto), m.end() + ancho)
        if tramos and ini <= tramos[-1][1]:
            tramos[-1][1] = max(tramos[-1][1], fin)
        else:
            tramos.append([ini, fin])
    return " [...] ".join(texto[a:b].strip() for a, b in tramos)


def _clave_cache(fragmento):
    return hashlib.sha256(f"{MODELO}|{VERSION_PROMPT}|{fragmento}".encode("utf-8")).hexdigest()


def cargar_cache():
    cache = {}
    if CACHE.exists():
        for linea in CACHE.read_text(encoding="utf-8").splitlines():
            reg = json.loads(linea)
            cache[reg["clave"]] = reg
    return cache


def _llamar(cliente, clave_api, fragmento, intentos=5):
    cuerpo = {"model": MODELO, "temperature": 0, "max_tokens": 250,
              "response_format": {"type": "json_schema", "json_schema": ESQUEMA},
              "messages": [{"role": "system", "content": PROMPT_SISTEMA},
                           {"role": "user", "content": f"Fragmentos del aviso:\n{fragmento}"}]}
    for k in range(intentos):
        r = cliente.post("https://api.openai.com/v1/chat/completions", json=cuerpo,
                         headers={"Authorization": f"Bearer {clave_api}"}, timeout=60)
        if r.status_code == 200:
            j = r.json()
            return json.loads(j["choices"][0]["message"]["content"]), j["usage"]
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** k)
            continue
        raise RuntimeError(f"Error de la API {r.status_code}: {r.text[:300]}")
    raise RuntimeError("La API no respondió tras varios intentos")


def extraer(fragmentos, hilos=8):
    """Devuelve una lista de dicts (uno por fragmento). Solo llama a la API para lo que no está en caché."""
    cache = cargar_cache()
    pendientes = sorted({f for f in fragmentos if _clave_cache(f) not in cache})
    if pendientes:
        clave_api = leer_clave()
        with httpx.Client() as cliente, CACHE.open("a", encoding="utf-8") as fh:
            def tarea(f):
                return f, *_llamar(cliente, clave_api, f)
            with ThreadPoolExecutor(hilos) as ex:
                for f, res, uso in ex.map(tarea, pendientes):
                    reg = {"clave": _clave_cache(f), "modelo": MODELO, "version_prompt": VERSION_PROMPT,
                           "resultado": res, "tokens_in": uso["prompt_tokens"], "tokens_out": uso["completion_tokens"]}
                    fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
                    cache[reg["clave"]] = reg
    return [cache[_clave_cache(f)] for f in fragmentos], len(pendientes)


def costo(registros):
    tin = sum(r["tokens_in"] for r in registros)
    tout = sum(r["tokens_out"] for r in registros)
    return tin, tout, tin / 1e6 * USD_M_IN + tout / 1e6 * USD_M_OUT

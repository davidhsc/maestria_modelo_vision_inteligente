"""
preparar_modelo.py  —  Fase 1 del pipeline: preparación del modelo entrenado
============================================================================
Toma el modelo final de la Actividad 3.3 y lo deja listo para producción:

  1. Lo copia a models/ con un nombre que incluye su VERSIÓN.
  2. Escribe clases.txt (una clase por línea, en el mismo orden del entrenamiento).
  3. Escribe model_card.json: la "ficha técnica" del modelo (qué entrada espera,
     cómo se preprocesa, métricas obtenidas, versiones de librerías y su huella SHA-256).

La huella SHA-256 es como una "huella digital" del archivo: si alguien cambia
un solo byte del modelo, la huella cambia y el pipeline lo detecta al cargarlo.

Uso (desde la raíz del proyecto):
    python herramientas/preparar_modelo.py
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ORIGEN = RAIZ / "entrenamiento" / "resultados"
DESTINO = RAIZ / "models"
VERSION = "1.0.0"
NOMBRE = "modelo_figuras_v1.keras"


def sha256(ruta):
    """Calcula la huella SHA-256 de un archivo leyendo por bloques."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(65536), b""):
            h.update(bloque)
    return h.hexdigest()


def main():
    import keras                      # se importa aquí para que el script arranque rápido
    import tensorflow as tf

    DESTINO.mkdir(exist_ok=True)
    metricas = json.loads((ORIGEN / "metricas_finales.json").read_text(encoding="utf-8"))
    clases = metricas["clases"]

    # 1) Copiar el modelo con nombre versionado
    ruta_modelo = DESTINO / NOMBRE
    shutil.copy2(ORIGEN / "modelo_aumento.keras", ruta_modelo)

    # 2) Archivo de clases (el orden importa: índice 0 = primera línea)
    (DESTINO / "clases.txt").write_text("\n".join(clases) + "\n", encoding="utf-8")

    # 3) Revisar el modelo para documentar su entrada/salida reales
    modelo = keras.models.load_model(ruta_modelo)
    capas_aumento = [c.name for c in modelo.layers if c.name.startswith("random_")]

    tarjeta = {
        "nombre": "Clasificador de figuras geométricas",
        "version": VERSION,
        "archivo": NOMBRE,
        "sha256": sha256(ruta_modelo),
        "fecha_preparacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origen": "Actividad 3.3 — Construcción del modelo de visión (modelo con aumento de datos)",
        "tarea": "clasificación (una etiqueta por imagen)",
        "arquitectura": "CNN 3 bloques Conv2D+MaxPooling (16-32-64) + Dense(64) + Dropout(0.3) + Dense(3, softmax)",
        "parametros": int(modelo.count_params()),
        "capas_solo_entrenamiento": capas_aumento,
        "entrada": {
            "forma": list(modelo.input_shape[1:]),           # (64, 64, 1)
            "color": "escala de grises (1 canal)",
            "normalizacion": "pixel / 255  ->  rango [0, 1]",
            "dominio": "figura oscura sobre fondo claro, una figura por imagen",
        },
        "salida": {"forma": list(modelo.output_shape[1:]), "tipo": "probabilidades softmax"},
        "clases": clases,
        "metricas_prueba": {
            "exactitud": metricas["aumento"]["prueba"]["exactitud"],
            "perdida": metricas["aumento"]["prueba"]["perdida"],
            "macro_f1": metricas["aumento"]["macro_f1"],
            "imagenes": metricas["imagenes"]["prueba"],
        },
        "versiones": {"tensorflow": tf.__version__, "keras": keras.__version__},
    }
    (DESTINO / "model_card.json").write_text(
        json.dumps(tarjeta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Modelo listo: {ruta_modelo.relative_to(RAIZ)}  (v{VERSION})")
    print(f"SHA-256: {tarjeta['sha256']}")
    print(f"Capas que solo actúan al entrenar (inactivas en inferencia): {capas_aumento}")


if __name__ == "__main__":
    main()

"""
modelo.py  —  Fases 5 y 6: carga del modelo e inferencia
=========================================================
Carga del modelo (Fase 5):
  1. Verificar que el archivo existe y que su huella SHA-256 coincide con la ficha técnica.
  2. Cargar arquitectura + pesos.
  3. Revisar que la entrada que espera la red es la que dice la configuración.
  4. Seleccionar el dispositivo (CPU o GPU).
  5. Prueba de integridad: una inferencia "de práctica" con una imagen en blanco.

Inferencia (Fase 6):
  * Forward-pass del lote preprocesado, midiendo el tiempo.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

# Silenciar los mensajes informativos de TensorFlow (deben ir ANTES de importarlo)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import keras             # noqa: E402


class ErrorModelo(Exception):
    """Problema grave con el modelo: sin él no se puede continuar."""


def huella_sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(65536), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_clases(ruta):
    """Lee clases.txt -> ['circulo', 'cuadrado', 'triangulo'] (el orden importa)."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise ErrorModelo(f"No se encuentra el archivo de clases: {ruta}")
    return [linea.strip() for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]


def dispositivo_disponible():
    """Devuelve 'GPU' si TensorFlow ve una tarjeta gráfica; si no, 'CPU'."""
    return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"


def cargar_modelo(ruta_modelo, ruta_tarjeta, forma_esperada, verificar_hash=True, log=None):
    """Carga y valida el modelo. Devuelve (modelo, tarjeta)."""
    ruta_modelo, ruta_tarjeta = Path(ruta_modelo), Path(ruta_tarjeta)
    info = log.info if log else print

    # 1) Existencia y huella digital
    if not ruta_modelo.exists():
        raise ErrorModelo(f"No se encuentra el modelo en: {ruta_modelo}")
    tarjeta = json.loads(ruta_tarjeta.read_text(encoding="utf-8")) if ruta_tarjeta.exists() else {}
    if verificar_hash and tarjeta:
        huella = huella_sha256(ruta_modelo)
        if huella != tarjeta.get("sha256"):
            raise ErrorModelo("La huella SHA-256 del modelo NO coincide con model_card.json: "
                              "el archivo fue modificado o es otra versión.")
        info(f"Huella SHA-256 verificada ({huella[:12]}...)")

    # 2) Cargar arquitectura + pesos. compile=False: para predecir no hace falta
    #    el optimizador ni la función de pérdida (carga más rápido y con menos memoria).
    inicio = time.perf_counter()
    modelo = keras.models.load_model(ruta_modelo, compile=False)
    info(f"Modelo cargado: {ruta_modelo.name} v{tarjeta.get('version', '?')} "
         f"en {time.perf_counter() - inicio:.2f} s")

    # 3) ¿La red espera la entrada que dice la configuración?
    forma_real = tuple(modelo.input_shape[1:])
    if forma_real != tuple(forma_esperada):
        raise ErrorModelo(f"El modelo espera {forma_real} pero la configuración indica {tuple(forma_esperada)}")

    # 4) Dispositivo
    info(f"Dispositivo de ejecución: {dispositivo_disponible()}")

    # 5) Prueba de integridad (también "calienta" el modelo: la primera llamada
    #    es lenta porque TensorFlow prepara el grafo; así no ensucia las mediciones)
    prueba = np.ones((1, *forma_real), dtype=np.float32)          # imagen en blanco
    salida = np.asarray(modelo(prueba, training=False))
    if salida.shape[0] != 1 or not np.isclose(salida.sum(), 1.0, atol=1e-3):
        raise ErrorModelo(f"La prueba de integridad falló: salida inesperada {salida}")
    info(f"Prueba de integridad correcta (salida {salida.shape}, suma de probabilidades = {salida.sum():.3f})")

    return modelo, tarjeta


def inferir(modelo, lote):
    """Forward-pass. Recibe (N, 64, 64, 1) y devuelve (probabilidades (N, 3), milisegundos).

    Se usa modelo(lote, training=False) en vez de modelo.predict():
      * training=False garantiza que Dropout y las capas de aumento estén APAGADAS.
      * Para lotes pequeños es más rápido que predict(), que está pensado para datasets grandes.
    """
    inicio = time.perf_counter()
    try:
        probabilidades = np.asarray(modelo(lote, training=False))
    except (tf.errors.ResourceExhaustedError, MemoryError) as e:
        raise ErrorModelo(f"Sin memoria al procesar un lote de {len(lote)} imágenes; "
                          f"reduce 'tamano_lote' en la configuración. Detalle: {e}") from e
    milisegundos = (time.perf_counter() - inicio) * 1000
    return probabilidades, milisegundos

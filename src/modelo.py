"""
modelo.py  —  Fases 5 y 6: carga del modelo e inferencia
=========================================================
Carga del modelo (Fase 5):
  1. Verificar que el archivo existe y que su huella SHA-256 coincide con la ficha técnica.
  2. Cargar el modelo con el MOTOR adecuado según su formato:
       .keras  -> Keras / TensorFlow   (servidor o PC; librería grande, ~600 MB instalada)
       .tflite -> LiteRT               (dispositivos edge; librería pequeña, ~30 MB)
  3. Revisar que la entrada que espera la red es la que dice la configuración.
  4. Prueba de integridad: una inferencia "de práctica" con una imagen en blanco.

Inferencia (Fase 6):
  * Forward-pass del lote preprocesado, midiendo el tiempo.

Los dos motores ofrecen la MISMA interfaz:  modelo(lote) -> probabilidades.
Así el resto del pipeline no sabe (ni necesita saber) cuál se está usando.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

# Silenciar los mensajes informativos de TensorFlow (si se llega a importar)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class ErrorModelo(Exception):
    """Problema grave con el modelo: sin él no se puede continuar."""


# ---------------------------------------------------------------------------- motores
class ModeloKeras:
    """Motor Keras/TensorFlow. Las librerías se importan aquí (y no arriba) para que
    una instalación edge, que no tiene TensorFlow, pueda usar este archivo igual."""

    motor = "keras"

    def __init__(self, ruta):
        import keras
        import tensorflow as tf
        self.red = keras.models.load_model(ruta, compile=False)
        self.forma_entrada = tuple(self.red.input_shape[1:])
        self.dispositivo = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

    def __call__(self, lote, training=False):
        # training=False garantiza que Dropout y las capas de aumento estén APAGADAS
        return np.asarray(self.red(lote, training=False))


class ModeloTFLite:
    """Motor LiteRT (TensorFlow Lite) para dispositivos edge."""

    motor = "tflite"

    def __init__(self, ruta, hilos=None):
        from ai_edge_litert.interpreter import Interpreter
        self.interprete = Interpreter(model_path=str(ruta), num_threads=hilos)
        self.interprete.allocate_tensors()
        self.entrada = self.interprete.get_input_details()[0]
        self.salida = self.interprete.get_output_details()[0]
        self.forma_entrada = tuple(int(n) for n in self.entrada["shape"][1:])
        self.lote_actual = int(self.entrada["shape"][0])
        self.dispositivo = f"CPU (LiteRT, {hilos or 'auto'} hilos)"

    def __call__(self, lote, training=False):
        lote = np.ascontiguousarray(lote, dtype=np.float32)
        if lote.shape[0] != self.lote_actual:          # LiteRT necesita saber el tamaño del lote
            self.interprete.resize_tensor_input(self.entrada["index"], lote.shape)
            self.interprete.allocate_tensors()
            self.lote_actual = lote.shape[0]
        self.interprete.set_tensor(self.entrada["index"], lote)
        self.interprete.invoke()
        return self.interprete.get_tensor(self.salida["index"]).copy()


# ---------------------------------------------------------------------------- utilidades
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


def huella_esperada(tarjeta, nombre_archivo):
    """Busca en la ficha técnica la huella del archivo concreto (Keras o alguna variante TFLite)."""
    artefacto = tarjeta.get("artefactos", {}).get(nombre_archivo)
    if artefacto:
        return artefacto["sha256"]
    return tarjeta.get("sha256") if tarjeta.get("archivo") == nombre_archivo else None


# ---------------------------------------------------------------------------- carga
def cargar_modelo(ruta_modelo, ruta_tarjeta, forma_esperada, verificar_hash=True, log=None, hilos=None):
    """Carga y valida el modelo. Devuelve (modelo, tarjeta)."""
    ruta_modelo, ruta_tarjeta = Path(ruta_modelo), Path(ruta_tarjeta)
    info = log.info if log else print

    # 1) Existencia y huella digital
    if not ruta_modelo.exists():
        raise ErrorModelo(f"No se encuentra el modelo en: {ruta_modelo}")
    tarjeta = json.loads(ruta_tarjeta.read_text(encoding="utf-8")) if ruta_tarjeta.exists() else {}
    if verificar_hash and tarjeta:
        esperada = huella_esperada(tarjeta, ruta_modelo.name)
        huella = huella_sha256(ruta_modelo)
        if huella != esperada:
            raise ErrorModelo("La huella SHA-256 del modelo NO coincide con model_card.json: "
                              "el archivo fue modificado o es otra versión.")
        info(f"Huella SHA-256 verificada ({huella[:12]}...)")

    # 2) Elegir el motor según la extensión del archivo
    inicio = time.perf_counter()
    if ruta_modelo.suffix == ".keras":
        modelo = ModeloKeras(ruta_modelo)
    elif ruta_modelo.suffix == ".tflite":
        modelo = ModeloTFLite(ruta_modelo, hilos)
    else:
        raise ErrorModelo(f"Formato de modelo no soportado: {ruta_modelo.suffix} (use .keras o .tflite)")
    info(f"Modelo cargado: {ruta_modelo.name} v{tarjeta.get('version', '?')} "
         f"(motor {modelo.motor}) en {time.perf_counter() - inicio:.2f} s")

    # 3) ¿La red espera la entrada que dice la configuración?
    if modelo.forma_entrada != tuple(forma_esperada):
        raise ErrorModelo(f"El modelo espera {modelo.forma_entrada} pero la configuración "
                          f"indica {tuple(forma_esperada)}")
    info(f"Dispositivo de ejecución: {modelo.dispositivo}")

    # 4) Prueba de integridad (también "calienta" el modelo: la primera llamada
    #    es lenta porque el motor prepara sus operaciones; así no ensucia las mediciones)
    prueba = np.ones((1, *modelo.forma_entrada), dtype=np.float32)        # imagen en blanco
    salida = modelo(prueba)
    if salida.shape[0] != 1 or not np.isclose(salida.sum(), 1.0, atol=1e-2):
        raise ErrorModelo(f"La prueba de integridad falló: salida inesperada {salida}")
    info(f"Prueba de integridad correcta (salida {salida.shape}, suma de probabilidades = {salida.sum():.3f})")

    return modelo, tarjeta


def inferir(modelo, lote):
    """Forward-pass. Recibe (N, 64, 64, 1) y devuelve (probabilidades (N, 3), milisegundos)."""
    inicio = time.perf_counter()
    try:
        probabilidades = modelo(lote)
    except (MemoryError, RuntimeError) as e:
        raise ErrorModelo(f"Falló la inferencia de un lote de {len(lote)} imágenes "
                          f"(¿memoria insuficiente? reduce 'tamano_lote'). Detalle: {e}") from e
    milisegundos = (time.perf_counter() - inicio) * 1000
    return probabilidades, milisegundos

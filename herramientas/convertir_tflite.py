"""
convertir_tflite.py  —  Fase 1 (optimización): modelo ligero para dispositivos edge
=====================================================================================
Convierte el modelo Keras a TensorFlow Lite (.tflite), el formato que usan
teléfonos, cámaras inteligentes y tarjetas como Raspberry Pi o Jetson.

Pasos:
  1. Quitar las capas que SOLO sirven para entrenar (aumento de datos). En inferencia
     no hacen nada, pero ocupan espacio y algunas no se pueden convertir.
  2. Convertir en tres variantes:
       float32  -> mismos números que Keras, solo cambia el formato.
       float16  -> pesos a media precisión: el archivo pesa la mitad.
       int8_pesos    -> cuantización de rango dinámico: solo los PESOS se guardan en
                        enteros de 8 bits; los cálculos y la salida siguen en float.
       int8_completo -> cuantización completa: pesos Y operaciones en enteros de 8 bits
                        (lo que exigen aceleradores como Coral/NPU). Necesita un "conjunto
                        representativo" de imágenes para calibrar los rangos de valores.
     Ambas int8 pesan ~4 veces menos que float32.
  3. Medir la exactitud de cada variante con las 180 imágenes de prueba de la 3.3.
  4. Registrar cada archivo (tamaño, SHA-256, exactitud) en models/model_card.json.

Uso (desde la raíz del proyecto):
    python herramientas/convertir_tflite.py
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import keras             # noqa: E402
from ai_edge_litert.interpreter import Interpreter  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "models"


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def cargar_carpeta(carpeta, clases):
    """Lee imágenes como en la 3.3: gris, /255, forma (N, 64, 64, 1)."""
    X, y = [], []
    for i, clase in enumerate(clases):
        for archivo in sorted((carpeta / clase).glob("*.png")):
            X.append(np.array(Image.open(archivo).convert("L"), dtype=np.float32) / 255.0)
            y.append(i)
    return np.array(X)[..., None], np.array(y)


def modelo_solo_inferencia(modelo):
    """Reconstruye la red SIN las capas de aumento de datos (comparten los mismos pesos)."""
    capas = [c for c in modelo.layers if not c.name.startswith("random_")]
    return keras.Sequential([keras.Input(shape=modelo.input_shape[1:]), *capas], name="cnn_inferencia")


def predecir_tflite(ruta, X):
    """Ejecuta un .tflite con LiteRT imagen por imagen (como en un dispositivo edge)."""
    inter = Interpreter(model_path=str(ruta))
    inter.allocate_tensors()
    ent, sal = inter.get_input_details()[0], inter.get_output_details()[0]
    salida = []
    for x in X:
        inter.set_tensor(ent["index"], x[None].astype(np.float32))
        inter.invoke()
        salida.append(inter.get_tensor(sal["index"])[0].copy())
    return np.array(salida)


def main():
    tarjeta_ruta = MODELOS / "model_card.json"
    tarjeta = json.loads(tarjeta_ruta.read_text(encoding="utf-8"))
    clases = tarjeta["clases"]
    modelo = keras.models.load_model(MODELOS / tarjeta["archivo"], compile=False)

    # ---- 1) Modelo sin capas de entrenamiento: debe dar EXACTAMENTE lo mismo ----
    inferencia = modelo_solo_inferencia(modelo)
    X_prueba, y_prueba = cargar_carpeta(RAIZ / "data" / "conjunto_prueba", clases)
    p_original = np.asarray(modelo(X_prueba, training=False))
    p_inferencia = np.asarray(inferencia(X_prueba, training=False))
    print(f"Capas eliminadas: {[c.name for c in modelo.layers if c.name.startswith('random_')]}")
    print(f"Diferencia máxima original vs. solo-inferencia: {np.abs(p_original - p_inferencia).max():.2e}")

    # Conjunto representativo para calibrar int8: 300 imágenes de ENTRENAMIENTO (no de prueba)
    X_entr, _ = cargar_carpeta(RAIZ / "entrenamiento" / "dataset_figuras", clases)
    rng = np.random.default_rng(42)
    representativo = X_entr[rng.choice(len(X_entr), 300, replace=False)]

    # ---- 2) Exportar a SavedModel (formato intermedio que entiende el convertidor) ----
    carpeta_tmp = Path(tempfile.mkdtemp())
    inferencia.export(str(carpeta_tmp / "saved_model"), format="tf_saved_model", verbose=False)

    def convertidor():
        return tf.lite.TFLiteConverter.from_saved_model(str(carpeta_tmp / "saved_model"))

    variantes = {}
    c = convertidor()                                                   # float32
    variantes["float32"] = c.convert()

    c = convertidor()                                                   # float16
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_types = [tf.float16]
    variantes["float16"] = c.convert()

    c = convertidor()                                                   # int8: solo pesos
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    variantes["int8_pesos"] = c.convert()

    c = convertidor()                                                   # int8: cuantización completa
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = lambda: ([x[None]] for x in representativo)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Entrada y salida se mantienen en float32 para que el pipeline no cambie
    variantes["int8_completo"] = c.convert()
    shutil.rmtree(carpeta_tmp, ignore_errors=True)

    # ---- 3) y 4) Guardar, medir y registrar ----
    base = Path(tarjeta["archivo"]).stem                                # modelo_figuras_v1
    artefactos = {tarjeta["archivo"]: {
        "formato": "keras", "precision": "float32",
        "bytes": (MODELOS / tarjeta["archivo"]).stat().st_size, "sha256": tarjeta["sha256"],
        "exactitud_prueba": round(float((p_original.argmax(1) == y_prueba).mean()), 4)}}
    for viejo in MODELOS.glob(f"{Path(tarjeta['archivo']).stem}_*.tflite"):
        viejo.unlink()                                                  # quitar conversiones anteriores
    print(f"\n{'archivo':<34}{'KB':>8}{'exactitud':>11}{'dif. máx. vs Keras':>21}")
    print(f"{tarjeta['archivo']:<34}{artefactos[tarjeta['archivo']]['bytes'] / 1024:>8.0f}"
          f"{artefactos[tarjeta['archivo']]['exactitud_prueba']:>11.2%}{'-':>21}")
    for precision, contenido in variantes.items():
        nombre = f"{base}_{precision}.tflite"
        ruta = MODELOS / nombre
        ruta.write_bytes(contenido)
        p = predecir_tflite(ruta, X_prueba)
        exactitud = float((p.argmax(1) == y_prueba).mean())
        orden = np.sort(p, axis=1)
        empates = int((orden[:, -1] - orden[:, -2] < 1e-6).sum())      # 1.ª y 2.ª clase con la misma prob.
        artefactos[nombre] = {"formato": "tflite", "precision": precision, "bytes": ruta.stat().st_size,
                              "sha256": sha256(ruta), "exactitud_prueba": round(exactitud, 4),
                              "dif_max_prob_vs_keras": round(float(np.abs(p - p_original).max()), 5),
                              "empates_prueba": empates}
        print(f"{nombre:<34}{ruta.stat().st_size / 1024:>8.0f}{exactitud:>11.2%}"
              f"{artefactos[nombre]['dif_max_prob_vs_keras']:>21.5f}   empates: {empates}")

    tarjeta["artefactos"] = artefactos
    tarjeta["versiones"]["ai_edge_litert"] = __import__("importlib.metadata").metadata.version("ai-edge-litert")
    tarjeta_ruta.write_text(json.dumps(tarjeta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nmodel_card.json actualizado con los artefactos.")


if __name__ == "__main__":
    sys.exit(main())

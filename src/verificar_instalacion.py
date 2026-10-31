"""
verificar_instalacion.py  —  ¿Quedó bien instalado el sistema?
================================================================
Revisa, en orden, todo lo que el sistema necesita y lo reporta con ✔ / ✘:

  1. Versión de Python.
  2. Librerías instaladas (según el motor del perfil: Keras o LiteRT).
  3. Archivos del modelo y su huella SHA-256 (que no estén alterados).
  4. Carga del modelo y prueba de integridad.
  5. Una predicción real: dibuja un círculo, un cuadrado y un triángulo y los clasifica.

Uso (desde la raíz del proyecto, o dentro del contenedor):
    python src/verificar_instalacion.py                                   # perfil completo
    python src/verificar_instalacion.py --config config/config_edge.yaml  # perfil edge
    docker run --rm vision-figuras:1.0-edge verificar
Código de salida: 0 si todo está bien, 1 si algo falló.
"""

import argparse
import importlib
import importlib.metadata as metadatos
import logging
import os
import sys
from pathlib import Path

import numpy as np

RESULTADOS = []


def revisar(nombre, funcion):
    """Ejecuta una comprobación y guarda ✔ o ✘ con el detalle."""
    try:
        detalle = funcion()
        RESULTADOS.append(True)
        print(f"  ✔ {nombre}: {detalle}")
        return True
    except Exception as e:                                   # noqa: BLE001 (queremos reportar cualquier falla)
        RESULTADOS.append(False)
        print(f"  ✘ {nombre}: {e}")
        return False


def dibujar(clase, lado=128):
    """Figura oscura sobre fondo claro, como las del dataset (no requiere archivos)."""
    import cv2
    img = np.full((lado, lado), 235, np.uint8)
    c, r = lado // 2, lado // 3
    if clase == "circulo":
        cv2.circle(img, (c, c), r, 40, -1)
    elif clase == "cuadrado":
        cv2.rectangle(img, (c - r, c - r), (c + r, c + r), 40, -1)
    else:
        puntos = np.array([[c, c - r], [c + r, c + r // 2 + 10], [c - r, c + r // 2 + 10]], np.int32)
        cv2.fillPoly(img, [puntos], 40)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verifica la instalación del sistema")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from configuracion import cargar_config, ruta_absoluta
    cfg = cargar_config(args.config or os.environ.get("PIPELINE_CONFIG"))
    m, p = cfg["modelo"], cfg["preprocesamiento"]
    ruta_modelo = ruta_absoluta(m["ruta"])
    motor = "tflite" if ruta_modelo.suffix == ".tflite" else "keras"
    print(f"Verificando la instalación (perfil: {Path(args.config or os.environ.get('PIPELINE_CONFIG') or 'config/config.yaml').name}, motor: {motor})\n")

    # 1) Python
    revisar("Python", lambda: f"{sys.version.split()[0]}" + ("" if sys.version_info >= (3, 10)
                                                              else "  (se recomienda 3.10 o superior)"))

    # 2) Librerías
    paquetes = ["numpy", "opencv-python-headless", "pyyaml"] + (
        ["ai-edge-litert"] if motor == "tflite" else ["keras", "tensorflow-cpu|tensorflow"])
    for paquete in paquetes:
        def version(nombres=paquete.split("|")):
            for n in nombres:
                try:
                    return metadatos.version(n)
                except metadatos.PackageNotFoundError:
                    continue
            raise RuntimeError("no está instalado (pip install -r requirements" +
                               ("-edge" if motor == "tflite" else "") + ".txt)")
        revisar(f"Librería {paquete.split('|')[0]}", version)

    # 3) y 4) Modelo
    from modelo import cargar_modelo, leer_clases
    estado = {}

    def cargar():
        estado["clases"] = leer_clases(ruta_absoluta(m["clases"]))
        estado["modelo"], tarjeta = cargar_modelo(ruta_modelo, ruta_absoluta(m["tarjeta"]),
                                                  (p["alto"], p["ancho"], p["canales"]), m.get("verificar_hash", True),
                                                  log=logging.getLogger("verificar"), hilos=m.get("hilos"))
        return f"{ruta_modelo.name} v{tarjeta.get('version')} (huella SHA-256 e integridad correctas)"
    if revisar("Modelo", cargar):
        # 5) Predicción real
        from preprocesamiento import crear_lote, preprocesar

        def predecir():
            clases = estado["clases"]
            esperadas = ["circulo", "cuadrado", "triangulo"]
            lote = crear_lote([preprocesar(dibujar(c), p["alto"], p["ancho"], p["divisor"]) for c in esperadas])
            obtenidas = [clases[i] for i in estado["modelo"](lote).argmax(1)]
            if obtenidas != esperadas:
                raise RuntimeError(f"se esperaba {esperadas} y se obtuvo {obtenidas}")
            return "círculo, cuadrado y triángulo clasificados correctamente"
        revisar("Predicción de prueba", predecir)

    correctas = sum(RESULTADOS)
    print(f"\n{'LISTO' if all(RESULTADOS) else 'HAY PROBLEMAS'}: {correctas} de {len(RESULTADOS)} comprobaciones correctas.")
    return 0 if all(RESULTADOS) else 1


if __name__ == "__main__":
    sys.exit(main())

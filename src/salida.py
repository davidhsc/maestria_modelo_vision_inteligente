"""
salida.py  —  Fase 8: generación de salida
===========================================
Formas de entregar el resultado:
  * Consola: para quien ejecuta el programa.
  * JSON: para otros sistemas (APIs, bases de datos).
  * CSV: para abrir en Excel y analizar.
  * Imagen / video anotado: para que una persona vea qué decidió el modelo.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

# Colores en formato BGR (el que usa OpenCV)
VERDE, NARANJA, ROJO, BLANCO, NEGRO = (40, 160, 40), (0, 140, 255), (40, 40, 220), (255, 255, 255), (0, 0, 0)


def guardar_json(datos, ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def guardar_csv(filas, ruta):
    """filas: lista de diccionarios con las mismas llaves."""
    if not filas:
        return
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


def a_color(img):
    """Asegura una imagen BGR de 3 canales para poder dibujar en color."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def anotar(img, resultado, lado_minimo=256, extra=""):
    """Dibuja una franja inferior con la clase y la confianza sobre la imagen."""
    lienzo = a_color(img)
    alto, ancho = lienzo.shape[:2]
    if min(alto, ancho) < lado_minimo:                       # agrandar imágenes chicas (64x64)
        escala = lado_minimo / min(alto, ancho)
        lienzo = cv2.resize(lienzo, (round(ancho * escala), round(alto * escala)),
                            interpolation=cv2.INTER_NEAREST)
        alto, ancho = lienzo.shape[:2]

    color = VERDE if resultado["estado"] == "ok" else NARANJA
    franja = max(28, alto // 9)
    cv2.rectangle(lienzo, (0, alto - franja), (ancho, alto), color, thickness=-1)
    texto = f"{resultado['clase']} {resultado['confianza']:.0%}"
    if resultado["estado"] != "ok":
        texto += " (revisar)"
    escala_txt = franja / 45
    # Si el texto no cabe en el ancho de la imagen, se reduce la letra hasta que quepa
    while escala_txt > 0.3 and cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala_txt,
                                               max(1, round(escala_txt * 2)))[0][0] > ancho - 16:
        escala_txt *= 0.9
    cv2.putText(lienzo, texto, (8, alto - franja // 3), cv2.FONT_HERSHEY_SIMPLEX,
                escala_txt, BLANCO, max(1, round(escala_txt * 2)), cv2.LINE_AA)
    if extra:                                                # texto adicional arriba (p. ej. FPS)
        cv2.putText(lienzo, extra, (8, franja // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    escala_txt * 0.8, NEGRO, 3, cv2.LINE_AA)
        cv2.putText(lienzo, extra, (8, franja // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    escala_txt * 0.8, BLANCO, 1, cv2.LINE_AA)
    return lienzo


def guardar_imagen(img, ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    ok, datos = cv2.imencode(Path(ruta).suffix or ".png", img)
    if ok:
        datos.tofile(str(ruta))


def crear_escritor_video(ruta, fps, tamano):
    """Abre un archivo .mp4 para ir escribiendo cuadros anotados."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, tamano)


def resumen_latencias(milisegundos):
    """Estadísticos de tiempo: la mediana (p50) es el caso típico, p95 el 'caso malo'."""
    ms = np.asarray(milisegundos, dtype=float)
    if ms.size == 0:
        return {}
    return {
        "media_ms": round(float(ms.mean()), 2),
        "p50_ms": round(float(np.percentile(ms, 50)), 2),
        "p95_ms": round(float(np.percentile(ms, 95)), 2),
        "max_ms": round(float(ms.max()), 2),
    }

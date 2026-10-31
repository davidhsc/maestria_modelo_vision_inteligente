"""
generar_dataset.py
==================
Crea un DATASET PROPIO de imágenes sencillas: círculos, cuadrados y triángulos.

¿Por qué generamos nuestras propias imágenes?
  - No dependemos de descargar nada de internet.
  - Sabemos exactamente qué hay en cada imagen (la etiqueta es 100 % correcta).
  - Es un problema fácil de entender, ideal para aprender cómo se entrena una CNN.

Cada imagen:
  - Mide 64 x 64 píxeles, en escala de grises (1 canal).
  - Tiene UNA figura con tamaño, posición, rotación, tono de gris y grosor aleatorios.
  - Lleva un poco de "ruido" (puntitos) para que no sea demasiado perfecta.

El resultado queda organizado así (una carpeta por clase):

    dataset_figuras/
        circulo/    circulo_0000.png, circulo_0001.png, ...
        cuadrado/   cuadrado_0000.png, ...
        triangulo/  triangulo_0000.png, ...

Uso:
    python generar_dataset.py                      # 400 imágenes por clase
    python generar_dataset.py --por-clase 200      # otra cantidad
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TAMANO = 64                                   # lado de la imagen en píxeles
CLASES = ["circulo", "cuadrado", "triangulo"]  # nombres de las carpetas / clases


def puntos_poligono(cx, cy, radio, lados, angulo):
    """Calcula los vértices de un polígono regular (3 lados = triángulo, 4 = cuadrado).

    cx, cy  -> centro de la figura
    radio   -> distancia del centro a cada vértice
    angulo  -> rotación de la figura en radianes
    """
    puntos = []
    for i in range(lados):
        a = angulo + 2 * math.pi * i / lados
        puntos.append((cx + radio * math.cos(a), cy + radio * math.sin(a)))
    return puntos


def dibujar_figura(clase):
    """Devuelve una imagen PIL de 64x64 con una figura de la clase indicada."""
    # 1) Fondo claro con un tono aleatorio (entre gris muy claro y blanco)
    fondo = random.randint(200, 255)
    img = Image.new("L", (TAMANO, TAMANO), color=fondo)   # "L" = escala de grises
    lapiz = ImageDraw.Draw(img)

    # 2) Parámetros aleatorios de la figura
    radio = random.randint(12, 26)                       # qué tan grande es
    cx = random.randint(radio + 2, TAMANO - radio - 2)   # posición horizontal
    cy = random.randint(radio + 2, TAMANO - radio - 2)   # posición vertical
    tono = random.randint(0, 110)                        # color oscuro de la figura
    rellena = random.random() < 0.5                      # rellena o solo contorno
    grosor = random.randint(2, 4)                        # grosor del contorno
    angulo = random.uniform(0, 2 * math.pi)              # rotación

    relleno = tono if rellena else None

    # 3) Dibujamos según la clase
    if clase == "circulo":
        caja = [cx - radio, cy - radio, cx + radio, cy + radio]
        lapiz.ellipse(caja, fill=relleno, outline=tono, width=grosor)
    elif clase == "cuadrado":
        lapiz.polygon(puntos_poligono(cx, cy, radio, 4, angulo),
                      fill=relleno, outline=tono, width=grosor)
    elif clase == "triangulo":
        lapiz.polygon(puntos_poligono(cx, cy, radio, 3, angulo),
                      fill=relleno, outline=tono, width=grosor)

    # 4) Añadimos ruido gaussiano suave para hacerla más "realista"
    arreglo = np.array(img).astype(np.float32)
    arreglo += np.random.normal(0, 12, arreglo.shape)    # media 0, desviación 12
    arreglo = np.clip(arreglo, 0, 255).astype(np.uint8)  # mantenemos el rango 0-255
    return Image.fromarray(arreglo)


def main():
    parser = argparse.ArgumentParser(description="Genera el dataset de figuras.")
    parser.add_argument("--por-clase", type=int, default=400, help="imágenes por clase")
    parser.add_argument("--salida", default="dataset_figuras", help="carpeta de salida")
    parser.add_argument("--semilla", type=int, default=42, help="semilla aleatoria")
    args = parser.parse_args()

    # Fijar la semilla hace que SIEMPRE se generen las mismas imágenes (reproducible)
    random.seed(args.semilla)
    np.random.seed(args.semilla)

    for clase in CLASES:
        carpeta = Path(args.salida) / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        for i in range(args.por_clase):
            dibujar_figura(clase).save(carpeta / f"{clase}_{i:04d}.png")
        print(f"{clase:>10}: {args.por_clase} imágenes en {carpeta}")

    print(f"Listo. Total: {args.por_clase * len(CLASES)} imágenes.")


if __name__ == "__main__":
    main()

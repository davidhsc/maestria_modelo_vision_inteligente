"""
preprocesamiento.py  —  Fase 4: preprocesamiento
=================================================
REGLA DE ORO: la imagen debe llegar al modelo EXACTAMENTE como llegaban las
imágenes de entrenamiento. En la Actividad 3.3 se hizo:

    Image.open(archivo).convert("L")      # 1) escala de grises
    imgs.astype("float32") / 255.0        # 2) normalizar a [0, 1]
    imgs[..., np.newaxis]                 # 3) agregar el canal -> (64, 64, 1)

Las imágenes del dataset ya medían 64x64. En producción pueden llegar de
cualquier tamaño, así que se agrega un paso de redimensionado.
"""

import cv2
import numpy as np


def a_escala_de_grises(img):
    """Convierte cualquier imagen (gris, BGR o BGRA) a escala de grises.

    OpenCV usa la misma fórmula que PIL en .convert("L"):
        gris = 0.299*R + 0.587*G + 0.114*B
    así que el resultado es igual al del entrenamiento.
    """
    if img.ndim == 2:                                   # ya es gris
        return img
    if img.shape[2] == 4:                               # tiene transparencia (BGRA)
        # Lo transparente se pinta de blanco: en el dataset el fondo es claro
        alfa = img[:, :, 3:4].astype(np.float32) / 255.0
        color = img[:, :, :3].astype(np.float32)
        img = (color * alfa + 255.0 * (1 - alfa)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def preprocesar(img, alto=64, ancho=64, divisor=255.0):
    """Imagen cruda (cualquier tamaño y color)  ->  arreglo (alto, ancho, 1) listo para la red."""
    gris = a_escala_de_grises(img)                                   # 1) color -> gris

    if gris.shape != (alto, ancho):                                  # 2) tamaño -> 64x64
        # INTER_AREA es la interpolación recomendada para REDUCIR imágenes:
        # promedia los píxeles y no "pierde" líneas delgadas.
        interpolacion = cv2.INTER_AREA if gris.shape[0] > alto else cv2.INTER_LINEAR
        gris = cv2.resize(gris, (ancho, alto), interpolation=interpolacion)

    x = gris.astype(np.float32) / divisor                            # 3) 0..255 -> 0..1
    return x[..., np.newaxis]                                        # 4) (64, 64) -> (64, 64, 1)


def crear_lote(imagenes_preprocesadas):
    """Junta varias imágenes (64, 64, 1) en un solo arreglo (N, 64, 64, 1).
    La red procesa un lote completo de una vez: es más eficiente que una por una."""
    return np.stack(imagenes_preprocesadas, axis=0)

"""
entrada.py  —  Fase 3: ingesta de datos (Input Layer)
======================================================
Responsabilidades:
  * Saber de dónde vienen los datos: una imagen, una carpeta, un video o una cámara.
  * Validar cada archivo ANTES de que llegue al modelo (formato, tamaño, canales).
  * Convertir los problemas en errores claros, en vez de que el programa "truene".
"""

from pathlib import Path

import cv2
import numpy as np

# OpenCV imprime sus propios avisos al leer archivos dañados; los silenciamos
# porque el pipeline ya reporta esos casos con un mensaje claro en su log.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)


class ErrorEntrada(Exception):
    """Error de una entrada concreta (archivo corrupto, formato no válido...).
    En el modo lote se registra y se continúa con la siguiente imagen."""


def leer_imagen(ruta, extensiones_validas, tamano_minimo=16):
    """Lee una imagen del disco y la valida.

    Devuelve un arreglo NumPy:
        (alto, ancho)      si la imagen es en escala de grises
        (alto, ancho, 3)   si es a color (OpenCV usa el orden BGR, no RGB)
        (alto, ancho, 4)   si además tiene transparencia (BGRA)
    """
    ruta = Path(ruta)

    # 1) ¿Existe la ruta?
    if not ruta.exists():
        raise ErrorEntrada(f"ruta inexistente: {ruta}")

    # 2) ¿La extensión es de imagen?
    if ruta.suffix.lower() not in extensiones_validas:
        raise ErrorEntrada(f"formato no soportado '{ruta.suffix}'")

    # 3) ¿El archivo tiene contenido?
    if ruta.stat().st_size == 0:
        raise ErrorEntrada("archivo vacío (0 bytes)")

    # 4) Intentar decodificarla. IMREAD_UNCHANGED respeta canales y transparencia.
    #    Se lee con np.fromfile para que funcionen rutas con acentos en Windows.
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ErrorEntrada("archivo corrupto o no es una imagen válida")

    # 5) Profundidad: convertimos 16 bits (p. ej. algunos TIFF) a 8 bits (0..255)
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ErrorEntrada(f"profundidad de color no soportada ({img.dtype})")

    # 6) Tamaño y canales
    alto, ancho = img.shape[:2]
    if min(alto, ancho) < tamano_minimo:
        raise ErrorEntrada(f"imagen demasiado pequeña ({ancho}x{alto})")
    canales = 1 if img.ndim == 2 else img.shape[2]
    if canales not in (1, 3, 4):
        raise ErrorEntrada(f"número de canales no soportado ({canales})")

    return img


def listar_imagenes(carpeta):
    """Lista TODOS los archivos de una carpeta (incluye subcarpetas), ordenados.

    No se filtra por extensión a propósito: queremos que los archivos inválidos
    lleguen a leer_imagen() y queden registrados como error, no que se ignoren en silencio.
    """
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise ErrorEntrada(f"no es una carpeta válida: {carpeta}")
    return sorted(p for p in carpeta.rglob("*") if p.is_file() and not p.name.startswith("."))


def abrir_fuente_video(fuente):
    """Abre un video (ruta de archivo) o una cámara (número: 0, 1, ...).

    Devuelve un 'generador': cada vez que se le pide, entrega el siguiente cuadro.
    Así el resto del pipeline no necesita saber si los cuadros vienen de un archivo
    o de una cámara real.
    """
    es_camara = str(fuente).isdigit()
    if not es_camara and not Path(fuente).exists():
        raise ErrorEntrada(f"no se encuentra el video: {fuente}")

    captura = cv2.VideoCapture(int(fuente) if es_camara else str(fuente))
    if not captura.isOpened():
        raise ErrorEntrada(f"no se pudo abrir la fuente de video: {fuente}")

    fps_origen = captura.get(cv2.CAP_PROP_FPS) or 0.0

    def cuadros():
        try:
            while True:
                ok, cuadro = captura.read()
                if not ok:                  # fin del video (o la cámara dejó de enviar)
                    break
                yield cuadro
        finally:
            captura.release()               # liberar el archivo / cámara siempre

    return cuadros(), fps_origen

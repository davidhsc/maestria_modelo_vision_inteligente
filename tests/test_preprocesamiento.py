"""Fase 4: el preprocesamiento debe ser idéntico al del entrenamiento (Actividad 3.3)."""

import cv2
import numpy as np
from PIL import Image

from preprocesamiento import a_escala_de_grises, crear_lote, preprocesar


def test_gris_igual_que_pil(tmp_path):
    """OpenCV (pipeline) y PIL .convert('L') (entrenamiento) dan la misma imagen en gris."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    ruta = tmp_path / "color.png"
    Image.fromarray(rgb).save(ruta)
    gris_pil = np.array(Image.open(ruta).convert("L"))
    gris_cv = a_escala_de_grises(cv2.imread(str(ruta)))
    assert np.abs(gris_pil.astype(int) - gris_cv.astype(int)).max() <= 1


def test_transparencia_se_vuelve_blanca():
    """Un PNG con fondo transparente no debe quedar con fondo negro."""
    bgra = np.zeros((32, 32, 4), dtype=np.uint8)         # todo transparente
    assert a_escala_de_grises(bgra).min() == 255


def test_forma_y_rango():
    img = np.full((300, 200, 3), 128, dtype=np.uint8)
    x = preprocesar(img)
    assert x.shape == (64, 64, 1)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_lote():
    lote = crear_lote([np.zeros((64, 64, 1), np.float32)] * 5)
    assert lote.shape == (5, 64, 64, 1)

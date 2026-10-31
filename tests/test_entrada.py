"""Fase 3: la entrada rechaza archivos inválidos con un error claro."""

import cv2
import numpy as np
import pytest

from entrada import ErrorEntrada, leer_imagen

VALIDAS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def test_lee_png_valido(tmp_path, figura_sintetica):
    ruta = tmp_path / "ok.png"
    cv2.imwrite(str(ruta), figura_sintetica)
    assert leer_imagen(ruta, VALIDAS).shape == (64, 64)


@pytest.mark.parametrize("nombre, contenido, mensaje", [
    ("vacia.png", b"", "vacío"),
    ("notas.txt", b"hola", "formato no soportado"),
    ("corrupta.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "corrupto"),
])
def test_rechaza_archivos_invalidos(tmp_path, nombre, contenido, mensaje):
    ruta = tmp_path / nombre
    ruta.write_bytes(contenido)
    with pytest.raises(ErrorEntrada, match=mensaje):
        leer_imagen(ruta, VALIDAS)


def test_rechaza_ruta_inexistente(tmp_path):
    with pytest.raises(ErrorEntrada, match="inexistente"):
        leer_imagen(tmp_path / "no_existe.png", VALIDAS)


def test_rechaza_imagen_diminuta(tmp_path):
    ruta = tmp_path / "chica.png"
    cv2.imwrite(str(ruta), np.full((8, 8), 255, np.uint8))
    with pytest.raises(ErrorEntrada, match="pequeña"):
        leer_imagen(ruta, VALIDAS, tamano_minimo=16)


def test_convierte_16_bits_a_8(tmp_path):
    ruta = tmp_path / "profunda.tif"
    cv2.imwrite(str(ruta), np.full((32, 32), 65535, np.uint16))
    img = leer_imagen(ruta, VALIDAS)
    assert img.dtype == np.uint8 and img.max() == 255

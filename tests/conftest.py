"""
Configuración común de las pruebas automáticas (pytest).

Ejecutar todas las pruebas desde la raíz del proyecto:
    pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))          # para importar los módulos del pipeline


@pytest.fixture(scope="session")
def raiz():
    return RAIZ


@pytest.fixture
def figura_sintetica():
    """Un cuadrado oscuro sobre fondo claro, de 64x64 en gris (no depende de archivos)."""
    img = np.full((64, 64), 235, dtype=np.uint8)
    img[16:48, 16:48] = 40
    return img

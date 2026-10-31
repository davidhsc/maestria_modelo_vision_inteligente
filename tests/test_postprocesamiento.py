"""Fase 7: interpretación de la salida, umbral, empates y suavizado temporal."""

import numpy as np

from postprocesamiento import INCIERTO, SuavizadorTemporal, interpretar

CLASES = ["circulo", "cuadrado", "triangulo"]


def test_clase_y_confianza():
    r = interpretar(np.array([0.1, 0.85, 0.05]), CLASES, umbral=0.8)
    assert r["clase"] == "cuadrado" and r["estado"] == "ok"
    assert [t["clase"] for t in r["top"]] == ["cuadrado", "circulo", "triangulo"]


def test_baja_confianza():
    r = interpretar(np.array([0.3, 0.4, 0.3]), CLASES, umbral=0.8)
    assert r["estado"] == "baja_confianza"


def test_empate_gana_la_primera_clase():
    """Con modelos cuantizados pueden aparecer empates exactos: se resuelven como argmax."""
    r = interpretar(np.array([0.0, 0.5, 0.5]), CLASES, umbral=0.8)
    assert r["clase"] == "cuadrado"


def test_suavizado_vota_y_marca_incierto():
    s = SuavizadorTemporal(ventana=5)
    assert s.actualizar("circulo", confiable=False) == INCIERTO      # solo dudas -> incierto
    for _ in range(3):
        salida = s.actualizar("cuadrado", confiable=True)
    assert salida == "cuadrado"

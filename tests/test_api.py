"""Servicio web: rutas /health, /info y /predict (con el perfil edge, que carga rápido)."""

import cv2
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("ai_edge_litert")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def cliente(raiz, tmp_path_factory):
    import os
    os.environ["PIPELINE_CONFIG"] = str(raiz / "config" / "config_edge.yaml")
    import api
    with TestClient(api.app) as c:          # "with" ejecuta el arranque (carga del modelo)
        yield c
    os.environ.pop("PIPELINE_CONFIG", None)


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["estado"] == "ok"


def test_info(cliente):
    assert cliente.get("/info").json()["clases"] == ["circulo", "cuadrado", "triangulo"]


def test_predict_figura(cliente, figura_sintetica):
    ok, png = cv2.imencode(".png", figura_sintetica)
    r = cliente.post("/predict", files={"imagen": ("cuadrado.png", png.tobytes(), "image/png")})
    assert r.status_code == 200
    assert r.json()["clase"] == "cuadrado"


@pytest.mark.parametrize("contenido, codigo", [(b"", 400), (b"no es una imagen", 400)])
def test_predict_rechaza_invalidos(cliente, contenido, codigo):
    r = cliente.post("/predict", files={"imagen": ("x.png", contenido, "image/png")})
    assert r.status_code == codigo

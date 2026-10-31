"""Fases 5-6 y prueba de paridad: el pipeline reproduce la exactitud de la Actividad 3.3."""

import json
import shutil

import numpy as np
import pytest

from modelo import ErrorModelo, cargar_modelo
from pipeline_operativo import PipelineOperativo


def test_modelo_alterado_se_rechaza(raiz, tmp_path):
    """Si el archivo del modelo cambia un solo byte, su huella SHA-256 ya no coincide."""
    copia = tmp_path / "modelo_figuras_v1_int8_pesos.tflite"
    shutil.copy(raiz / "models" / copia.name, copia)
    datos = bytearray(copia.read_bytes())
    datos[-50] ^= 0xFF
    copia.write_bytes(datos)
    with pytest.raises(ErrorModelo, match="SHA-256"):
        cargar_modelo(copia, raiz / "models" / "model_card.json", (64, 64, 1), log=None)


@pytest.mark.parametrize("config, motor", [
    ("config/config.yaml", "keras"),
    ("config/config_edge.yaml", "tflite"),
])
def test_paridad_con_actividad_3_3(raiz, tmp_path, config, motor):
    """Las 180 imágenes de prueba de la 3.3 deben dar la misma exactitud (99.44 %)."""
    pytest.importorskip("keras" if motor == "keras" else "ai_edge_litert")
    pipe = PipelineOperativo(raiz / config, modo="prueba", carpeta_salida=tmp_path)
    assert pipe.modelo.motor == motor
    resumen = pipe.ejecutar_lote(raiz / "data" / "conjunto_prueba")
    tarjeta = json.loads((raiz / "models" / "model_card.json").read_text(encoding="utf-8"))
    assert resumen["errores"] == 0 and resumen["procesadas"] == 180
    assert resumen["exactitud"] == pytest.approx(tarjeta["metricas_prueba"]["exactitud"], abs=1e-4)


def test_keras_y_tflite_float32_dan_lo_mismo(raiz):
    pytest.importorskip("keras")
    pytest.importorskip("ai_edge_litert")
    tarjeta = raiz / "models" / "model_card.json"
    keras_m, _ = cargar_modelo(raiz / "models" / "modelo_figuras_v1.keras", tarjeta, (64, 64, 1), log=None)
    lite_m, _ = cargar_modelo(raiz / "models" / "modelo_figuras_v1_float32.tflite", tarjeta, (64, 64, 1), log=None)
    x = np.random.default_rng(1).random((8, 64, 64, 1), dtype=np.float32)
    assert np.abs(keras_m(x) - lite_m(x)).max() < 1e-4

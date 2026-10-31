"""
api.py  —  El pipeline como SERVICIO WEB (despliegue en contenedor)
====================================================================
En lugar de ejecutar un script por cada imagen, el modelo se carga UNA vez y
queda esperando peticiones por HTTP. Así cualquier sistema (una app, una cámara,
otro microservicio) puede pedir una clasificación enviando una imagen.

Rutas:
    GET  /          Página de prueba: dibujar o subir una figura y clasificarla.
    GET  /health    Estado del servicio (lo usa Docker para saber si está "sano").
    GET  /info      Versión del modelo, motor, clases y métricas.
    POST /predict   Recibe una imagen (multipart, campo "imagen") y devuelve la predicción en JSON.

Ejecutar (desde la raíz del proyecto):
    uvicorn --app-dir src api:app --host 0.0.0.0 --port 8000
La configuración se elige con la variable de entorno PIPELINE_CONFIG (por defecto config/config.yaml).
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from pipeline_operativo import PipelineOperativo

ESTADO = {}                      # aquí vive el pipeline cargado (uno por proceso)
MAX_BYTES = 5 * 1024 * 1024      # 5 MB por imagen como máximo


@asynccontextmanager
async def ciclo_de_vida(app):
    """Se ejecuta al arrancar: carga y valida el modelo UNA sola vez."""
    inicio = time.perf_counter()
    ESTADO["pipe"] = PipelineOperativo(modo="api")
    ESTADO["arranque_s"] = round(time.perf_counter() - inicio, 2)
    ESTADO["peticiones"] = 0
    yield
    ESTADO.clear()


app = FastAPI(title="Clasificador de figuras geométricas", version="1.0.0", lifespan=ciclo_de_vida)


@app.get("/", include_in_schema=False)
def pagina():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def salud():
    pipe = ESTADO.get("pipe")
    if pipe is None:
        raise HTTPException(503, "El modelo aún no está cargado")
    return {"estado": "ok", "modelo": pipe.tarjeta.get("version"), "motor": pipe.modelo.motor,
            "peticiones_atendidas": ESTADO["peticiones"]}


@app.get("/info")
def info():
    pipe = ESTADO["pipe"]
    archivo = Path(pipe.cfg["modelo"]["ruta"]).name
    return {"modelo": pipe.tarjeta.get("nombre"), "version": pipe.tarjeta.get("version"),
            "archivo": archivo, "motor": pipe.modelo.motor, "dispositivo": pipe.modelo.dispositivo,
            "clases": pipe.clases, "entrada": pipe.tarjeta.get("entrada"),
            "metricas": pipe.tarjeta.get("artefactos", {}).get(archivo, pipe.tarjeta.get("metricas_prueba")),
            "umbral_confianza": pipe.cfg["postprocesamiento"]["umbral_confianza"],
            "arranque_s": ESTADO["arranque_s"]}


@app.post("/predict")
async def predecir(imagen: UploadFile = File(...)):
    pipe = ESTADO["pipe"]
    datos = await imagen.read()
    # --- Validación de la entrada (Fase 3), igual que en el pipeline por archivos ---
    if not datos:
        raise HTTPException(400, "El archivo está vacío")
    if len(datos) > MAX_BYTES:
        raise HTTPException(413, "La imagen supera los 5 MB")
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "El archivo no es una imagen válida o está corrupto")
    if min(img.shape[:2]) < pipe.cfg["entrada"]["tamano_minimo"]:
        raise HTTPException(400, f"Imagen demasiado pequeña ({img.shape[1]}x{img.shape[0]})")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)

    resultado = pipe.predecir_arreglo(img)                       # Fases 4 a 7
    ESTADO["peticiones"] += 1
    pipe.log.info(f"API /predict '{imagen.filename}': {resultado['clase']} "
                  f"({resultado['confianza']:.1%}) en {resultado['total_ms']:.1f} ms")
    return {"archivo": imagen.filename, **resultado}

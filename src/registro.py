"""
registro.py  —  Fase 9: registro, logs y control de versiones
==============================================================
Cada ejecución recibe un identificador único (run_id) y deja:
  * outputs/logs/<run_id>.log         -> bitácora detallada (qué pasó y cuándo)
  * outputs/registro_ejecuciones.csv  -> una línea por ejecución (auditoría)

Con esto se puede responder después: ¿qué versión del modelo se usó?,
¿cuántas imágenes fallaron?, ¿qué tan rápido fue?
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path


def nuevo_run_id(modo):
    """Ejemplo: 20261031-153012_lote"""
    return f"{datetime.now():%Y%m%d-%H%M%S}_{modo}"


def configurar_log(run_id, carpeta_salida):
    """Crea un logger que escribe en consola y en archivo al mismo tiempo."""
    carpeta_logs = Path(carpeta_salida) / "logs"
    carpeta_logs.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(f"pipeline.{run_id}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    formato = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(formato)
    archivo = logging.FileHandler(carpeta_logs / f"{run_id}.log", encoding="utf-8")
    archivo.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    log.addHandler(consola)
    log.addHandler(archivo)
    return log


def registrar_ejecucion(carpeta_salida, fila):
    """Agrega una línea al registro de auditoría (crea el archivo con encabezados si no existe)."""
    ruta = Path(carpeta_salida) / "registro_ejecuciones.csv"
    nuevo = not ruta.exists()
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(fila.keys()))
        if nuevo:
            escritor.writeheader()
        escritor.writerow(fila)

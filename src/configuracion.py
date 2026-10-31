"""
configuracion.py  —  Lee config/config.yaml y resuelve las rutas
==================================================================
El código nunca escribe rutas "a mano" (hardcodeadas). Todas salen de aquí.
"""

from pathlib import Path

import yaml

# Carpeta raíz del proyecto: la que contiene src/, models/, config/ ...
RAIZ = Path(__file__).resolve().parents[1]
CONFIG_POR_DEFECTO = RAIZ / "config" / "config.yaml"


def ruta_absoluta(ruta):
    """Convierte una ruta relativa del config en absoluta (desde la raíz del proyecto)."""
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else RAIZ / ruta


def cargar_config(ruta=None):
    """Devuelve la configuración como diccionario de Python."""
    ruta = ruta_absoluta(ruta) if ruta else CONFIG_POR_DEFECTO
    if not ruta.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de configuración: {ruta}")
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)

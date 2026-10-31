#!/bin/sh
# Punto de entrada del contenedor: un mismo contenedor sirve para dos usos.
#   docker run IMAGEN                 -> levanta el servicio web (API) en el puerto 8000
#   docker run IMAGEN api             -> igual que el anterior
#   docker run IMAGEN --modo lote ... -> ejecuta el pipeline por línea de comandos
#   docker run IMAGEN benchmark ...   -> ejecuta la medición de rendimiento
set -e
case "$1" in
  ""|api)    shift 2>/dev/null || true
             exec uvicorn --app-dir src api:app --host 0.0.0.0 --port 8000 "$@" ;;
  benchmark) shift; exec python3 src/benchmark.py "$@" ;;
  *)         exec python3 src/pipeline_operativo.py "$@" ;;
esac

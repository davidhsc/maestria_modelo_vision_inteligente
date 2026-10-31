# Despliegue con Docker

El sistema se distribuye en **dos imágenes** construidas con el mismo código:

| Imagen | Dockerfile | Modelo y motor | Uso recomendado |
|---|---|---|---|
| `vision-figuras:1.0-completo` | `Dockerfile` | `.keras` con TensorFlow (CPU) | Servidor o PC con recursos de sobra |
| `vision-figuras:1.0-edge` | `Dockerfile.edge` | `.tflite` con pesos int8 y LiteRT, 1 hilo | Dispositivos pequeños (gateway, cámara inteligente, Raspberry Pi) |

## 1. Construir

```bash
docker build -t vision-figuras:1.0-completo .
docker build -f Dockerfile.edge -t vision-figuras:1.0-edge .
# o las dos a la vez:
docker compose build
```

> La imagen base es `python:3.12-slim`. Si no hay acceso a Docker Hub, se puede usar otra base
> equivalente con `--build-arg BASE_IMAGE=<imagen>`. Debe tener Python 3.12 y el módulo `venv`.

## 2. Ejecutar como servicio web

```bash
docker run -d -p 8000:8000 --name figuras vision-figuras:1.0-edge
# Página de prueba (dibujar o subir una figura):  http://localhost:8000
curl http://localhost:8000/health
curl -F imagen=@data/ejemplos/nuevas/cuadrado_1.png http://localhost:8000/predict
```

Las dos imágenes a la vez: `docker compose up -d` (completo → puerto 8000, edge → puerto 8001,
este último con medio núcleo y 256 MB de RAM).

## 3. Ejecutar el pipeline por línea de comandos

El mismo contenedor acepta los argumentos de `src/pipeline_operativo.py`. Los datos y los
resultados se **montan como volúmenes**: la imagen no contiene datos.

```bash
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/outputs:/app/outputs" \
    vision-figuras:1.0-edge --modo lote --entrada data/conjunto_prueba

docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/outputs:/app/outputs" \
    vision-figuras:1.0-edge --modo tiempo-real --entrada data/video/figuras.mp4 \
    --etiquetas data/video/figuras_etiquetas.csv --fps 15
```

En Windows (PowerShell) se usa `${PWD}` en lugar de `$PWD`.

## 4. Simular un dispositivo edge

Docker puede limitar los núcleos (`--cpus`) y la memoria (`--memory`) del contenedor:

```bash
docker run --rm --cpus=0.25 --memory=128m --memory-swap=128m \
    -v "$PWD/data:/app/data:ro" vision-figuras:1.0-edge benchmark --etiqueta edge_limitado
```

Para medir todos los escenarios (host, contenedor completo y edge, con y sin límites):

```bash
python herramientas/benchmark_despliegue.py      # resultados en benchmarks/resultados/
```

## 5. Requisitos mínimos medidos

| | Completo | Edge |
|---|---|---|
| Tamaño de la imagen | ≈ 2.3 GB | ≈ 0.6 GB |
| Memoria mínima | 512 MB (con 256 MB Docker detiene el proceso) | 128 MB (con 64 MB Docker lo detiene) |
| `docker run` hasta la primera predicción | ≈ 4.4 s | ≈ 0.6 s |
| Latencia por imagen (p50) | ≈ 12 ms | ≈ 0.2 ms |
| Exactitud (180 imágenes de prueba) | 99.44 % | 99.44 % |

## 6. Variables y configuración

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `PIPELINE_CONFIG` | `config/config.yaml` (completo) · `config/config_edge.yaml` (edge) | Perfil de configuración |

Para cambiar un parámetro sin reconstruir la imagen, se monta otro archivo de configuración:
`-v "$PWD/mi_config.yaml:/app/config/mi_config.yaml" -e PIPELINE_CONFIG=config/mi_config.yaml`.

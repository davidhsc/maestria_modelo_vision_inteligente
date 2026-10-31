# =============================================================================
# Dockerfile  —  Imagen COMPLETA (servidor / PC)
# -----------------------------------------------------------------------------
# Contiene el pipeline con el modelo Keras original y TensorFlow (CPU).
# Construir:   docker build -t vision-figuras:1.0-completo .
# Servicio:    docker run -p 8000:8000 vision-figuras:1.0-completo         -> http://localhost:8000
# Por lote:    docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
#                  vision-figuras:1.0-completo --modo lote --entrada data/conjunto_prueba
# =============================================================================

# Imagen base oficial de Python. Se puede cambiar al construir con --build-arg BASE_IMAGE=...
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Buenas prácticas de Python en contenedores:
#   no generar archivos .pyc, mostrar los logs al instante y no guardar caché de pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIPELINE_CONFIG=config/config.yaml

WORKDIR /app

# 1) Dependencias primero: Docker guarda esta capa en caché y no la repite
#    mientras requirements*.txt no cambien (los cambios de código no la invalidan).
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements.txt -r requirements-api.txt

# 2) Solo lo necesario para operar: código, configuración y el modelo Keras
COPY config/ config/
COPY models/clases.txt models/model_card.json models/modelo_figuras_v1.keras models/
COPY src/ src/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# 3) Seguridad: el proceso NO corre como root
RUN useradd --create-home --uid 1000 app && mkdir -p outputs data && chown -R app:app /app
USER app

EXPOSE 8000
# Docker consulta /health cada 30 s para saber si el servicio está "sano"
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

ENTRYPOINT ["entrypoint.sh"]
CMD ["api"]

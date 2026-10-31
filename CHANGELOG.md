# Historial de versiones

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/). Cada versión tiene una etiqueta de git.

## [1.0.0] — Actividad 4.3: paquete profesional y video-demo
### Agregado
- Manual de instalación y operación (`docs/MANUAL.md`).
- Pruebas automáticas con `pytest` (`tests/`): preprocesamiento, validación de entrada, posprocesamiento,
  huella del modelo, paridad con la Actividad 3.3 (Keras y TFLite) y servicio web.
- Verificador de instalación (`src/verificar_instalacion.py`, también `docker run IMAGEN verificar`).
- Paquete completo reproducible (`python herramientas/empaquetar.py --completo`): incluye dataset y entrenamiento.
- Video-demo del sistema en funcionamiento.

## [0.2] — Actividad 4.2: empaquetado y despliegue (`v0.2-act4.2`)
### Agregado
- Conversión a TensorFlow Lite (float32, float16, int8 de pesos e int8 completo) con ficha por artefacto.
- Motor de inferencia intercambiable (Keras / LiteRT) y perfil `config/config_edge.yaml`.
- Servicio web FastAPI con página para dibujar o subir figuras.
- `Dockerfile` (completo) y `Dockerfile.edge` (sin TensorFlow), `docker-compose.yml`.
- Benchmark de 14 escenarios de despliegue y evidencia de ejecución.
- Paquete de entrega mínima con manifiesto SHA-256.
### Corregido
- Desempate determinista cuando dos clases tienen la misma probabilidad (modelos int8).
- Scripts con fin de línea CRLF al clonar en Windows; certificado corporativo opcional para `pip`.

## [0.1] — Actividad 4.1: pipeline operativo (`v0.1-act4.1`)
### Agregado
- Pipeline por fases (entrada, preprocesamiento, modelo, posprocesamiento, salida, registro).
- Modos imagen, lote y tiempo real simulado; logs y bitácora de auditoría.
- Modelo de la Actividad 3.3 versionado con ficha técnica y huella SHA-256.
- Datos de prueba y notebook de análisis.

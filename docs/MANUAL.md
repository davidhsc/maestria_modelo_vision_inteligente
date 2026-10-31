# Manual de instalación y operación

**Sistema:** Clasificador de figuras geométricas (círculo, cuadrado y triángulo) · **Versión:** 1.0.0
**Proyecto:** Maestría en Inteligencia Artificial · Visión Inteligente · Unidad IV
**Autor:** David Díaz de la Cruz · **Repositorio:** https://github.com/davidhsc/maestria_modelo_vision_inteligente

## 1. Descripción del sistema

El sistema recibe imágenes y responde qué figura geométrica contiene: **círculo**, **cuadrado** o **triángulo**, con su nivel de confianza. Usa una red neuronal convolucional (CNN) entrenada con un dataset propio de 1,200 imágenes (Actividad 3.3), con 99.44 % de exactitud en el conjunto de prueba.

Se puede operar de tres formas:

| Forma de uso | Para qué sirve | Cómo se inicia |
|---|---|---|
| Servicio web | Una página para dibujar o subir figuras, y una API para otros sistemas | `docker run -p 8000:8000 vision-figuras:1.0-edge` |
| Línea de comandos | Procesar una imagen, una carpeta completa o un video | `python src/pipeline_operativo.py --modo ...` |
| Contenedor por lotes | Lo mismo que la línea de comandos, sin instalar Python | `docker run ... vision-figuras:1.0-edge --modo lote ...` |

Cada imagen recorre las mismas fases:

```
entrada -> validación -> preprocesamiento (gris, 64x64, /255) -> modelo -> interpretación (clase, confianza, umbral) -> salida (pantalla, JSON, CSV, imagen anotada)
                                                  registro de cada ejecución (logs y bitácora)
```

El sistema se distribuye con dos **perfiles**, que producen los mismos resultados:

| Perfil | Modelo y motor | Recomendado para |
|---|---|---|
| Completo | `modelo_figuras_v1.keras` con TensorFlow | Servidores o PC; reentrenamiento y depuración |
| Edge | `modelo_figuras_v1_int8_pesos.tflite` con LiteRT | Dispositivos pequeños (gateway, cámara inteligente, Raspberry Pi) y uso general |

## 2. Requisitos

| | Con Docker (recomendado) | Instalación local con Python |
|---|---|---|
| Sistema operativo | Windows 10/11, macOS o Linux con Docker Desktop o Docker Engine | Windows, macOS o Linux |
| Software | Docker 24 o superior; Git (opcional) | Python 3.10 o superior (probado con 3.12 y 3.13); Git (opcional) |
| Memoria RAM libre | Edge: 128 MB · Completo: 512 MB | Edge: 200 MB · Completo: 1 GB |
| Disco | Edge: 0.7 GB · Completo: 2.5 GB | Edge: 0.4 GB · Completo: 1.5 GB |
| Red | Solo para descargar e instalar | Solo para descargar e instalar |

Verificado en Linux (Ubuntu 24.04) y en Windows 11 con Docker Desktop y Git Bash.

## 3. Instalación

### 3.1 Obtener el sistema

Con Git (permite cambiar entre versiones; ver sección 7.3):

```bash
git clone https://github.com/davidhsc/maestria_modelo_vision_inteligente.git
cd maestria_modelo_vision_inteligente
```

Sin Git: descargar y descomprimir el paquete `vision-figuras-v1.0.0.zip` (solo operación) o `vision-figuras-v1.0.0-completo.zip` (incluye dataset, entrenamiento, notebooks y pruebas).

### 3.2 Opción A: con Docker (recomendada)

Construir la imagen edge (unos 40 segundos) o la completa (unos 3 minutos):

```bash
docker build -f Dockerfile.edge -t vision-figuras:1.0-edge .
docker build -t vision-figuras:1.0-completo .
```

Comprobar la instalación:

```bash
docker run --rm vision-figuras:1.0-edge verificar
```

La respuesta debe terminar con `LISTO: 7 de 7 comprobaciones correctas.`

> **Red corporativa con antivirus o proxy que inspecciona HTTPS.** Si la construcción falla con `CERTIFICATE_VERIFY_FAILED`, ver la sección 8.

### 3.3 Opción B: instalación local con Python

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows (Git Bash): source .venv/Scripts/activate
                                          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-api.txt          # perfil completo
# o bien, perfil edge (sin TensorFlow, mucho más ligero):
pip install -r requirements-edge.txt -r requirements-api.txt
```

Comprobar la instalación:

```bash
python src/verificar_instalacion.py                                   # perfil completo
python src/verificar_instalacion.py --config config/config_edge.yaml  # perfil edge
```

### 3.4 Instalación para desarrollo (opcional)

Para ejecutar los notebooks, las pruebas automáticas y los scripts de `herramientas/`:

```bash
pip install -r requirements-notebook.txt -r requirements-api.txt pytest httpx
pytest -q                                  # 24 pruebas; deben pasar todas
```

## 4. Operación

### 4.1 Servicio web

Iniciar el servicio:

```bash
docker run -d -p 8000:8000 --name figuras vision-figuras:1.0-edge
# sin Docker:
PIPELINE_CONFIG=config/config_edge.yaml uvicorn --app-dir src api:app --host 0.0.0.0 --port 8000
```

Abrir **http://localhost:8000** en el navegador:

1. Dibujar una figura en el recuadro con el mouse o el dedo, o pulsar **Subir imagen**.
2. Pulsar **Clasificar dibujo**.
3. Leer el resultado: la clase, una etiqueta verde (**confianza suficiente**) o naranja (**baja confianza: revisar**) y la probabilidad de cada clase.

Rutas de la API, para integrar el sistema con otras aplicaciones:

| Método y ruta | Descripción | Respuesta |
|---|---|---|
| `GET /health` | Estado del servicio (Docker lo consulta cada 30 s) | `{"estado": "ok", "motor": "tflite", ...}` |
| `GET /info` | Versión, modelo, clases, métricas y umbral | JSON |
| `POST /predict` | Clasifica una imagen enviada en el campo `imagen` (máximo 5 MB) | JSON con `clase`, `confianza`, `estado`, `top` y tiempos |
| `GET /docs` | Documentación interactiva de la API | Página web |

Ejemplo:

```bash
curl -F imagen=@data/ejemplos/nuevas/cuadrado_1.png http://localhost:8000/predict
```

```json
{"archivo": "cuadrado_1.png", "clase": "cuadrado", "confianza": 0.9974, "estado": "ok",
 "top": [{"clase": "cuadrado", "prob": 0.9974}, {"clase": "triangulo", "prob": 0.0015}, {"clase": "circulo", "prob": 0.0011}],
 "inferencia_ms": 0.43, "total_ms": 0.69}
```

Errores: `400` si el archivo está vacío, no es una imagen o es demasiado pequeño; `413` si supera 5 MB.

Detener el servicio: `docker stop figuras && docker rm figuras`.

Para levantar los dos perfiles a la vez (completo en el puerto 8000 y edge en el 8001): `docker compose up -d`.

### 4.2 Línea de comandos

| Modo | Comando | Resultado |
|---|---|---|
| Una imagen | `python src/pipeline_operativo.py --modo imagen --entrada data/ejemplos/nuevas/triangulo_1.png` | Clase y confianza en pantalla, imagen anotada |
| Carpeta | `python src/pipeline_operativo.py --modo lote --entrada data/conjunto_prueba` | CSV con una fila por imagen; exactitud si hay una subcarpeta por clase |
| Video o cámara | `python src/pipeline_operativo.py --modo tiempo-real --entrada data/video/figuras.mp4 --fps 15` | Video anotado y métricas de latencia; `--entrada 0` usa la cámara |

Con Docker, los mismos argumentos van después del nombre de la imagen, y los datos y resultados se montan como carpetas:

```bash
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/outputs:/app/outputs" \
    vision-figuras:1.0-edge --modo lote --entrada data/conjunto_prueba
```

En Windows (PowerShell) se usa `${PWD}` en lugar de `$PWD`.

Opciones disponibles:

| Opción | Modo | Descripción |
|---|---|---|
| `--config RUTA` | todos | Perfil de configuración (por defecto `config/config.yaml`) |
| `--umbral 0.8` | todos | Confianza mínima para aceptar una predicción |
| `--salida CARPETA` | todos | Carpeta de resultados (por defecto `outputs/`) |
| `--lote 32` | lote | Imágenes procesadas juntas |
| `--guardar-imagenes` | lote | Guarda cada imagen con su resultado dibujado |
| `--fps 15` | tiempo real | Ritmo de llegada de los cuadros |
| `--sin-espera` | tiempo real | Procesa a la máxima velocidad posible |
| `--etiquetas CSV` | tiempo real | Archivo `cuadro,clase_real` para medir la exactitud |
| `--mostrar` | tiempo real | Muestra el video en una ventana (requiere pantalla) |

Código de salida: `0` si todo salió bien; `1` si hubo un error de entrada o de modelo. Los archivos inválidos dentro de una carpeta **no** detienen el lote: se registran en `errores.csv`.

### 4.3 Cómo interpretar los resultados

| Campo | Significado |
|---|---|
| `clase` | Figura con la probabilidad más alta |
| `confianza` | Probabilidad de esa clase (0 a 1) |
| `estado` | `ok` si la confianza supera el umbral; `baja_confianza` si no: **revisar manualmente** |
| `top` | Las tres clases ordenadas por probabilidad |
| `clase_suavizada` | Solo en video: votación de los últimos 5 cuadros; `incierto` si la mayoría tuvo baja confianza |

Cada ejecución crea la carpeta `outputs/<fecha-hora>_<modo>/` con:

* `resumen.json`: totales, exactitud (si aplica), latencias y versión del modelo.
* `resultados.csv`: una fila por imagen o cuadro, con la probabilidad de cada clase.
* `errores.csv`: archivos rechazados y el motivo (modo lote).
* Imagen o video anotado: verde = aceptado, naranja = revisar.

### 4.4 Configuración

Los parámetros están en `config/config.yaml` (perfil completo) y `config/config_edge.yaml` (perfil edge). No es necesario modificar el código.

| Parámetro | Valor | Descripción |
|---|---|---|
| `modelo.ruta` | `models/...` | Archivo del modelo (`.keras` o `.tflite`) |
| `modelo.verificar_hash` | `true` | Comprobar que el modelo no fue alterado |
| `modelo.hilos` | `null` / `1` | Hilos de CPU (solo `.tflite`) |
| `entrada.extensiones_validas` | png, jpg, bmp, tif | Formatos aceptados |
| `entrada.tamano_minimo` | `16` | Lado mínimo en píxeles |
| `inferencia.tamano_lote` | `32` / `8` | Imágenes por lote |
| `postprocesamiento.umbral_confianza` | `0.80` | Por debajo se marca "baja confianza" |
| `tiempo_real.fps_objetivo` | `15` | Cuadros por segundo esperados |
| `tiempo_real.ventana_suavizado` | `5` | Cuadros que votan; `1` desactiva el suavizado |

El perfil se elige con `--config` o con la variable de entorno `PIPELINE_CONFIG`. En Docker, para usar una configuración propia sin reconstruir la imagen:

```bash
docker run -v "$PWD/mi_config.yaml:/app/config/mi_config.yaml" -e PIPELINE_CONFIG=config/mi_config.yaml ...
```

**Elección del umbral:** subirlo reduce los errores aceptados, pero deja más imágenes para revisión manual. Con 0.80, en el video de prueba se aceptan 5.6 % de errores y se revisa el 6.1 % de los cuadros. Con 0.96, se aceptan 2.5 % de errores y se revisa el 16 %.

### 4.5 Registros y auditoría

* `outputs/logs/<run_id>.log`: detalle de cada ejecución (verificación del modelo, archivos rechazados, tiempos).
* `outputs/registro_ejecuciones.csv`: una línea por ejecución (modo, entrada, versión del modelo, imágenes procesadas, errores, exactitud y duración).
* `docker logs figuras`: registro del servicio web, con cada petición atendida.

## 5. Despliegue en dispositivos edge

La imagen edge está pensada para equipos con pocos recursos. Medido con los límites de Docker:

| Métrica | Completo | Edge |
|---|---|---|
| Exactitud | 99.44 % | 99.44 % |
| Tamaño de la imagen | ≈ 2.3 GB | ≈ 0.6 GB |
| Memoria RAM mínima | 512 MB | 128 MB |
| Tiempo por imagen (mediana) | ≈ 12 ms | ≈ 0.2 ms |
| De `docker run` a la primera predicción | ≈ 4.4 s | ≈ 0.6 s |

Para simular un dispositivo antes de instalarlo, se limitan los núcleos y la memoria del contenedor:

```bash
docker run -d -p 8000:8000 --cpus=0.25 --memory=128m --memory-swap=128m vision-figuras:1.0-edge
```

Para medir el rendimiento en el equipo destino:

```bash
docker run --rm -v "$PWD/data:/app/data:ro" vision-figuras:1.0-edge benchmark --etiqueta mi_equipo
python herramientas/benchmark_despliegue.py        # todos los escenarios (requiere ambas imágenes)
```

En dispositivos ARM (Raspberry Pi, Jetson), la imagen se construye en el propio dispositivo con el mismo `Dockerfile.edge`, o con `docker buildx build --platform linux/arm64 ...`.

## 6. Limitaciones conocidas

| Situación | Comportamiento | Recomendación |
|---|---|---|
| Figura cortada por el borde de la imagen | Exactitud de 63 %, a veces con alta confianza | Encuadrar la figura completa |
| Figura que no es círculo, cuadrado ni triángulo (estrella, pentágono) | Siempre elige una de las 3 clases, a veces con ~100 % de confianza | El sistema no detecta "otras" figuras |
| Contorno punteado o discontinuo | Un círculo punteado se clasifica como cuadrado o triángulo (marcado "revisar") | Usar trazos continuos |
| Imagen sin figura | Responde "triángulo" con ~45 % de confianza, siempre por debajo del umbral | Respetar la etiqueta "revisar" |
| Varias figuras en una imagen | Responde una sola clase | Una figura por imagen |
| Video a más de ~70 FPS | Se descartan cuadros; el límite es leer y escribir el video, no el modelo | Usar 15 a 30 FPS |

Estas limitaciones vienen del dataset de entrenamiento. Se corrigen reentrenando con figuras parciales, punteadas y una clase "otro" (sección 7.2).

## 7. Mantenimiento

### 7.1 Pruebas automáticas

```bash
pytest -q
```

Las 24 pruebas cubren el preprocesamiento (idéntico al del entrenamiento), la validación de archivos, la interpretación de resultados, la huella del modelo, la **paridad** con la Actividad 3.3 (99.44 % con Keras y con TFLite) y el servicio web. Deben ejecutarse antes de publicar cualquier cambio.

### 7.2 Actualizar el modelo

1. Regenerar o ampliar el dataset: `python entrenamiento/generar_dataset.py --salida entrenamiento/dataset_figuras`.
2. Reentrenar con el notebook `entrenamiento/Act3.3_Construccion_Modelo_Vision.ipynb`.
3. Versionar el modelo y su ficha técnica: `python herramientas/preparar_modelo.py` (cambiar `VERSION` en el script).
4. Generar las variantes TFLite: `python herramientas/convertir_tflite.py`.
5. Regenerar los datos de prueba: `python herramientas/generar_datos_prueba.py`.
6. Ejecutar `pytest -q`, reconstruir las imágenes Docker y medir con `herramientas/benchmark_despliegue.py`.

### 7.3 Versiones

Cada entrega tiene una etiqueta de git: `v0.1-act4.1`, `v0.2-act4.2` y `v1.0.0`. Para ver una versión anterior: `git checkout v0.1-act4.1`; para volver: `git checkout main`. El detalle está en el README y en `CHANGELOG.md`.

### 7.4 Generar los paquetes de entrega

```bash
python herramientas/empaquetar.py              # dist/vision-figuras-v1.0.0.zip (operación)
python herramientas/empaquetar.py --completo   # dist/vision-figuras-v1.0.0-completo.zip (todo el proyecto)
```

Cada paquete incluye `MANIFIESTO.json` con la huella SHA-256 de cada archivo.

## 8. Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `exec /usr/local/bin/entrypoint.sh: no such file or directory` | El repositorio se clonó en Windows con fin de línea CRLF (copias anteriores a la versión 0.2) | Actualizar (`git pull`) y reconstruir la imagen; la versión 1.0.0 lo corrige al construir |
| `pip` falla con `CERTIFICATE_VERIFY_FAILED` al construir | Un antivirus o proxy corporativo inspecciona HTTPS | Exportar los certificados de Windows con `python herramientas/exportar_certificados_windows.py` y construir con `docker build --secret id=ca_corporativa,src=certs/ca-windows.pem ...` |
| El contenedor se detiene con código `137` | Memoria insuficiente (*OOM*) | Dar más memoria (`--memory`) o usar la imagen edge |
| `port is already allocated` | El puerto 8000 ya está en uso | Usar otro puerto: `-p 8080:8000` |
| `La huella SHA-256 del modelo NO coincide` | El archivo del modelo cambió o no corresponde a la ficha técnica | Restaurar el archivo original o regenerar la ficha con `herramientas/preparar_modelo.py` |
| `Permission denied` al escribir en `outputs/` (Linux) | El contenedor no corre como root (usuario 1000) | `mkdir -p outputs && chmod 777 outputs`, o agregar `--user "$(id -u)"` |
| Aviso `TensorFlow GPU support is not available on native Windows` | TensorFlow no usa GPU en Windows nativo | Ninguna: el sistema usa CPU y es suficiente |
| Muchas predicciones con "baja confianza" | Las imágenes no se parecen al dataset (fondo oscuro, figuras cortadas, varias figuras) | Revisar la sección 6; ajustar el encuadre o reentrenar |
| `verificar` reporta una librería faltante | Instalación incompleta | Repetir `pip install -r ...` del perfil correspondiente |

## Anexo: estructura del proyecto

```
├── src/                      Pipeline (un módulo por fase), servicio web, benchmark y verificador
├── config/                   Perfiles de configuración: completo y edge
├── models/                   Modelo .keras, variantes .tflite, clases y ficha técnica
├── data/                     Conjunto de prueba, ejemplos y video de prueba
├── entrenamiento/            Generador del dataset y entrenamiento (Actividad 3.3)
├── notebooks/                Análisis de las actividades 4.1 y 4.2
├── tests/                    Pruebas automáticas (pytest)
├── herramientas/             Preparación del modelo, conversión, datos, benchmark y empaquetado
├── benchmarks/               Resultados de rendimiento y evidencia de ejecución
├── docker/                   Punto de entrada de los contenedores
├── docs/                     Este manual y el video-demo
├── Dockerfile, Dockerfile.edge, docker-compose.yml
├── requirements*.txt         Dependencias por perfil
├── README.md, LEEME_DESPLIEGUE.md, CHANGELOG.md
```

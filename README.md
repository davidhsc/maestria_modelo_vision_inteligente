# Sistema de visión inteligente: clasificador de figuras geométricas

**Maestría en Inteligencia Artificial · Visión Inteligente (ITV / TecNM) · Unidad IV: Despliegue de un modelo de visión**
Autor: David Díaz de la Cruz

Proyecto integrador de la Unidad IV. Toma la CNN entrenada en la **Actividad 3.3** (clasifica **círculo, cuadrado y
triángulo**, 99.44 % de exactitud en prueba) y la lleva de un notebook de laboratorio a un sistema operativo y desplegable.

| Actividad | Entregable | Estado |
|---|---|---|
| 4.1 Pipeline operativo | `src/` + `notebooks/Practica_Tema4_1_Pipeline_Operativo.ipynb` | ✅ |
| 4.2 Empaquetado y despliegue | `Dockerfile`, `Dockerfile.edge`, API, benchmark, `notebooks/Practica_Tema4_2_Empaquetado_Despliegue.ipynb` | ✅ |
| 4.3 Paquete profesional y video-demo | [Manual de instalación y operación](docs/MANUAL.md), pruebas automáticas, verificador, paquete completo, video-demo | ✅ |

### Video-demo

**[docs/video_demo_v1.0.0.mp4](docs/video_demo_v1.0.0.mp4)** (1 min 55 s, sin audio, con textos explicativos): el problema y el
modelo, el pipeline, el procesamiento por lote, el tiempo real, los contenedores Docker (también en Windows), el servicio
web, la prueba del círculo punteado y la comparación entre el despliegue completo y el edge.

---

## Versiones del proyecto

Cada entrega queda marcada con una **etiqueta (tag)** de git, fija en su commit aunque el proyecto siga avanzando:

| Etiqueta | Actividad | Qué contiene | Estado |
|---|---|---|---|
| `v0.1-act4.1` | 4.1 Pipeline operativo | Pipeline por fases (imagen, lote, tiempo real), datos de prueba, notebook 4.1 | ✅ |
| `v0.2-act4.2` | 4.2 Empaquetado y despliegue | + Modelos TFLite, API web, Dockerfiles completo y edge, benchmark, paquete de entrega | ✅ |
| `v1.0.0` | 4.3 Paquete profesional y video-demo | + Manual, pruebas automáticas, verificador de instalación, paquete completo, video-demo | ✅ |

### Cómo moverse entre versiones

```bash
git fetch --tags                        # traer las etiquetas del repositorio remoto
git tag -n1                             # listar las versiones con su descripción

git checkout v0.1-act4.1                # ver el proyecto tal como se entregó en la 4.1
git checkout v0.2-act4.2                # ... en la 4.2
git checkout main                       # volver a la versión más reciente
```

* `git checkout <etiqueta>` deja el repositorio en modo **solo lectura** (*detached HEAD*): sirve para revisar y
  ejecutar esa versión, pero no para hacer commits.
* Para **trabajar** a partir de una versión anterior, crea una rama: `git switch -c revision-4.1 v0.1-act4.1`.
* Para comparar dos versiones: `git diff v0.1-act4.1 v0.2-act4.2 --stat` (archivos que cambiaron).
* Sin git: en GitHub, el selector de ramas → pestaña **Tags** → cada versión se descarga como `.zip`.

> Al cambiar de versión, los archivos que no están en git (`outputs/`, `dist/`, `entrenamiento/dataset_figuras/`)
> no se modifican. Si una versión anterior se comporta distinto, regenera lo necesario con los scripts de
> `herramientas/` de **esa** versión, y vuelve a construir las imágenes Docker (`docker build ...`), porque una
> imagen construida con otra versión conserva el código con el que se construyó.

### Crear la etiqueta de una entrega

```bash
git tag -a v1.0.0 -m "Actividad 4.3 - Paquete profesional y video-demo"
git push origin --tags
```

---

## Inicio rápido

> Guía completa: **[docs/MANUAL.md](docs/MANUAL.md)** (instalación, operación, configuración, edge y solución de problemas).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/verificar_instalacion.py       # comprueba que todo quedó bien instalado

# Una imagen
python src/pipeline_operativo.py --modo imagen --entrada data/ejemplos/nuevas/triangulo_1.png

# Una carpeta completa (en lotes). Si tiene una subcarpeta por clase, calcula la exactitud.
python src/pipeline_operativo.py --modo lote --entrada data/conjunto_prueba

# Video cuadro por cuadro, simulando una cámara de 15 FPS (o --entrada 0 para una cámara real)
python src/pipeline_operativo.py --modo tiempo-real --entrada data/video/figuras.mp4 \
       --etiquetas data/video/figuras_etiquetas.csv --fps 15
```

Cada ejecución deja sus resultados en `outputs/<run_id>/` (`resultados.csv`, `resumen.json`, imágenes o video
anotados), su log en `outputs/logs/` y una línea en `outputs/registro_ejecuciones.csv`.

### Opciones del script

| Opción | Modo | Descripción |
|---|---|---|
| `--modo {imagen,lote,tiempo-real}` | todos | Modo de ejecución (obligatorio) |
| `--entrada` | todos | Imagen, carpeta, video o número de cámara |
| `--config` | todos | Otro archivo de configuración (por defecto `config/config.yaml`) |
| `--umbral 0.8` | todos | Confianza mínima para aceptar una predicción |
| `--salida` | todos | Carpeta de resultados (por defecto `outputs/`) |
| `--lote 32` | lote | Imágenes por lote |
| `--guardar-imagenes` | lote | Guarda cada imagen anotada |
| `--fps 15` | tiempo-real | Ritmo de llegada simulado de los cuadros |
| `--sin-espera` | tiempo-real | Procesa a máxima velocidad (mide la capacidad) |
| `--etiquetas` | tiempo-real | CSV `cuadro,clase_real` para medir exactitud |
| `--mostrar` | tiempo-real | Abre una ventana con el video (requiere pantalla) |

Código de salida: `0` = correcto; `1` = error de entrada o de modelo (mensaje en *stderr*).

### Con Docker (Actividad 4.2)

```bash
docker build -f Dockerfile.edge -t vision-figuras:1.0-edge .        # imagen ligera (TFLite)
docker run -d -p 8000:8000 vision-figuras:1.0-edge                  # servicio web -> http://localhost:8000
docker run --rm -v "$PWD/data:/app/data:ro" vision-figuras:1.0-edge --modo lote --entrada data/conjunto_prueba
```

Detalles, imagen completa, `docker compose` y simulación de dispositivos edge: **[LEEME_DESPLIEGUE.md](LEEME_DESPLIEGUE.md)**.

### Servicio web

`uvicorn --app-dir src api:app --port 8000` (o el contenedor). Rutas: `GET /` (página para dibujar o subir una
figura), `GET /health`, `GET /info`, `POST /predict` (campo `imagen`).

---

## Arquitectura del pipeline

```
 imagen / carpeta / video
        │
 [Fase 3] entrada.py           leer + validar (formato, corrupto, vacío, tamaño, canales, 16 bits)
        │
 [Fase 4] preprocesamiento.py  gris → 64×64 → /255 → (N, 64, 64, 1)   ← idéntico a la Actividad 3.3
        │
 [Fase 5] modelo.py            SHA-256 + forma de entrada + prueba de integridad (al iniciar)
 [Fase 6] modelo.py            forward-pass (training=False) medido en ms
        │
 [Fase 7] postprocesamiento.py clase + confianza + umbral; votación temporal en video
        │
 [Fase 8] salida.py            consola · JSON · CSV · imagen/video anotado
 [Fase 9] registro.py          log por ejecución + bitácora de auditoría
```

## Estructura

```
├── config/
│   ├── config.yaml              Perfil completo (Keras): rutas, umbral, tamaños, FPS
│   └── config_edge.yaml         Perfil edge (TFLite int8, 1 hilo)
├── models/
│   ├── modelo_figuras_v1.keras  Modelo de la Actividad 3.3, versionado
│   ├── modelo_figuras_v1_*.tflite  Variantes TFLite: float32, float16, int8_pesos, int8_completo
│   ├── clases.txt               Una clase por línea (orden de las salidas de la red)
│   └── model_card.json          Ficha técnica: entrada, métricas, versiones, SHA-256 de cada artefacto
├── src/                         Pipeline operativo (un módulo por fase), api.py y benchmark.py
├── Dockerfile / Dockerfile.edge Imágenes completa y edge · docker/entrypoint.sh · docker-compose.yml
├── benchmarks/                  Resultados de rendimiento (JSON) y evidencia de ejecución (logs)
├── tests/                       Pruebas automáticas (pytest -q)
├── docs/MANUAL.md               Manual de instalación y operación
├── data/
│   ├── conjunto_prueba/         Las 180 imágenes de prueba de la 3.3 (para verificar paridad)
│   ├── ejemplos/                Casos nuevos, variaciones, fuera de dominio, sin clase e inválidos
│   └── video/                   Video sintético de 30 s + etiquetas por cuadro
├── notebooks/                   Práctica 4.1 ejecutada (análisis de resultados)
├── entrenamiento/               Actividad 3.3: generador del dataset y notebook de entrenamiento
├── herramientas/                Scripts de preparación (modelo y datos de prueba)
├── requirements.txt             Dependencias del pipeline completo (versión fija)
├── requirements-edge.txt        Dependencias mínimas edge (sin TensorFlow)
├── requirements-api.txt         Servicio web
└── requirements-notebook.txt    Desarrollo: notebooks y herramientas
```

## Reproducir todo desde cero

```bash
pip install -r requirements-notebook.txt
cd entrenamiento && python generar_dataset.py && cd ..          # dataset propio (1,200 imágenes, semilla 42)
jupyter nbconvert --to notebook --execute --inplace entrenamiento/Act3.3_Construccion_Modelo_Vision.ipynb
python herramientas/preparar_modelo.py                          # Fase 1: versiona el modelo y crea su ficha
python herramientas/generar_datos_prueba.py                     # datos de prueba y video
jupyter nbconvert --to notebook --execute --inplace notebooks/Practica_Tema4_1_Pipeline_Operativo.ipynb
python herramientas/convertir_tflite.py                         # 4.2: variantes TFLite
docker compose build                                            # 4.2: imágenes completa y edge
python herramientas/benchmark_despliegue.py                     # 4.2: escenarios de rendimiento
python herramientas/empaquetar.py                               # 4.2: paquete de entrega (dist/)
jupyter nbconvert --to notebook --execute --inplace notebooks/Practica_Tema4_2_Empaquetado_Despliegue.ipynb
```

> Si se reentrena el modelo, `preparar_modelo.py` actualiza la huella SHA-256 de la ficha técnica. Si el archivo del
> modelo cambia sin actualizar la ficha, el pipeline se niega a cargarlo.

## Resultados principales (Actividad 4.1)

| Prueba | Resultado |
|---|---|
| Paridad con la Actividad 3.3 (180 imágenes) | 99.44 % — idéntico; probabilidades iguales hasta el redondeo |
| Throughput de inferencia (CPU, 2 núcleos) | ~100 img/s con lote 1 → ~3,000 img/s con lote 180 |
| Latencia por cuadro en tiempo real | p50 ≈ 14 ms, p95 ≈ 20 ms (presupuesto a 15 FPS: 66.7 ms) |
| Capacidad máxima en tiempo real | ≈ 60–80 FPS |
| Video (figuras completas) | 100 % de exactitud |
| Video (figura entrando por el borde) | 63 %: limitación identificada |
| Archivos inválidos | rechazados con motivo, sin detener el lote |

## Resultados principales (Actividad 4.2)

| | Completo (Keras + TensorFlow) | Edge (TFLite int8 + LiteRT) |
|---|---|---|
| Exactitud | 99.44 % | 99.44 % |
| Tamaño del modelo | 3.4 MB | 0.29 MB |
| Imagen Docker | ≈ 2.3 GB | ≈ 0.6 GB |
| Latencia por imagen (p50) | ≈ 12 ms | ≈ 0.2 ms |
| RAM mínima del contenedor | 512 MB | 128 MB |
| `docker run` → primera predicción | ≈ 4.4 s | ≈ 0.6 s |

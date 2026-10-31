"""
benchmark.py  —  Medición de rendimiento del pipeline (Actividad 4.2)
======================================================================
Mide, con la configuración indicada (modelo + motor):

  * carga_s            tiempo para importar el motor y cargar/validar el modelo (arranque)
  * exactitud          sobre una carpeta con una subcarpeta por clase
  * latencia (lote 1)  ms por imagen: solo inferencia y de punta a punta (preprocesar + inferir)
  * throughput         imágenes por segundo procesando en lotes
  * memoria_pico_mb    memoria máxima que usó el proceso (RSS)
  * recursos           CPUs y memoria que el contenedor permite usar (límites de Docker)

Uso:
    python src/benchmark.py --config config/config_edge.yaml --etiqueta edge --salida benchmarks/edge.json
    docker run --rm -v "$PWD/data:/app/data" IMAGEN benchmark --etiqueta contenedor
"""

import time
_INICIO_PROCESO = time.perf_counter()        # antes de cualquier import pesado

import argparse      # noqa: E402
import json          # noqa: E402
import logging       # noqa: E402
import os            # noqa: E402
import platform      # noqa: E402
import resource      # noqa: E402
import sys           # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np   # noqa: E402

from configuracion import cargar_config, ruta_absoluta   # noqa: E402
from entrada import leer_imagen                          # noqa: E402
from modelo import cargar_modelo, leer_clases            # noqa: E402
from preprocesamiento import crear_lote, preprocesar     # noqa: E402


def leer_limite(ruta, divisor=1):
    try:
        valor = Path(ruta).read_text().split()[0]
        return None if valor in ("max", "-1") else int(valor) / divisor
    except (OSError, ValueError, IndexError):
        return None


def recursos_del_contenedor():
    """Lee los límites de CPU y memoria que Docker impone (cgroups v1 o v2)."""
    cpus = None
    cpu_max = leer_limite_texto("/sys/fs/cgroup/cpu.max")                      # cgroups v2
    if cpu_max and cpu_max[0] != "max":
        cpus = int(cpu_max[0]) / int(cpu_max[1])
    else:                                                                       # cgroups v1
        cuota = leer_limite("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        periodo = leer_limite("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if cuota and periodo and cuota > 0:
            cpus = cuota / periodo
    memoria = (leer_limite("/sys/fs/cgroup/memory.max")
               or leer_limite("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if memoria and memoria > 1e15:                                              # "sin límite" en v1
        memoria = None
    return {"cpus_visibles": os.cpu_count(), "limite_cpus": cpus,
            "limite_memoria_mb": round(memoria / 2**20) if memoria else None}


def leer_limite_texto(ruta):
    try:
        return Path(ruta).read_text().split()
    except OSError:
        return None


def memoria_proceso():
    """Memoria del proceso en MB, separada en dos partes (Linux, /proc/self/status):
       anonima  -> datos propios del programa (tensores, arreglos): NO se puede liberar.
       archivos -> librerías (.so) cargadas desde disco: el sistema puede liberarla y volver
                   a leerla si falta memoria, por eso no cuenta igual para un límite de RAM."""
    campos = {}
    try:
        for linea in Path("/proc/self/status").read_text().splitlines():
            clave, _, valor = linea.partition(":")
            if clave in ("RssAnon", "RssFile", "VmHWM"):
                campos[clave] = round(int(valor.split()[0]) / 1024, 1)
    except OSError:
        pass
    return {"memoria_anonima_mb": campos.get("RssAnon"), "memoria_librerias_mb": campos.get("RssFile")}


def percentiles(ms):
    ms = np.asarray(ms)
    return {"p50_ms": round(float(np.percentile(ms, 50)), 3), "p95_ms": round(float(np.percentile(ms, 95)), 3),
            "media_ms": round(float(ms.mean()), 3)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Medición de rendimiento del pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--datos", default="data/conjunto_prueba", help="carpeta con una subcarpeta por clase")
    ap.add_argument("--repeticiones", type=int, default=300, help="mediciones de latencia (lote 1)")
    ap.add_argument("--etiqueta", default="sin_nombre", help="nombre del escenario")
    ap.add_argument("--salida", default=None, help="archivo JSON de resultados")
    ap.add_argument("--modelo", default=None, help="otro archivo de modelo (sustituye al del config)")
    ap.add_argument("--hilos", type=int, default=None, help="hilos de CPU para .tflite")
    args = ap.parse_args(argv)

    cfg = cargar_config(args.config or os.environ.get("PIPELINE_CONFIG"))
    if args.modelo:
        cfg["modelo"]["ruta"] = args.modelo
    if args.hilos:
        cfg["modelo"]["hilos"] = args.hilos
    p, m = cfg["preprocesamiento"], cfg["modelo"]
    clases = leer_clases(ruta_absoluta(m["clases"]))

    # ---- 1) Arranque: importar el motor + cargar y validar el modelo ----
    t0 = time.perf_counter()
    modelo, _ = cargar_modelo(ruta_absoluta(m["ruta"]), ruta_absoluta(m["tarjeta"]),
                              (p["alto"], p["ancho"], p["canales"]), m.get("verificar_hash", True),
                              log=logging.getLogger("benchmark"), hilos=m.get("hilos"))
    listo = time.perf_counter()
    carga_s = listo - t0

    # ---- 2) Leer imágenes (decodificar) y preprocesar ----
    carpeta = ruta_absoluta(args.datos)
    archivos = [(f, i) for i, c in enumerate(clases) for f in sorted((carpeta / c).glob("*.png"))]
    crudas = [leer_imagen(f, cfg["entrada"]["extensiones_validas"]) for f, _ in archivos]
    y = np.array([i for _, i in archivos])
    t = time.perf_counter()
    X = crear_lote([preprocesar(img, p["alto"], p["ancho"], p["divisor"]) for img in crudas])
    preproceso_ms = (time.perf_counter() - t) * 1000 / len(crudas)

    # ---- 3) Exactitud ----
    exactitud = float((modelo(X).argmax(1) == y).mean())

    # ---- 4) Latencia con lote de 1 (como en tiempo real) ----
    inferencia, punta_a_punta = [], []
    for k in range(args.repeticiones):
        img = crudas[k % len(crudas)]
        t = time.perf_counter()
        x = crear_lote([preprocesar(img, p["alto"], p["ancho"], p["divisor"])])
        t_inf = time.perf_counter()
        modelo(x)
        fin = time.perf_counter()
        inferencia.append((fin - t_inf) * 1000)
        punta_a_punta.append((fin - t) * 1000)

    # ---- 5) Throughput en lotes (mediana de 5 repeticiones) ----
    throughput = {}
    for tam in sorted({cfg["inferencia"]["tamano_lote"], 32}):
        modelo(X[:tam])                                   # calentamiento para ese tamaño
        tiempos = []
        for _ in range(5):
            t = time.perf_counter()
            for i in range(0, len(X), tam):
                modelo(X[i:i + tam])
            tiempos.append(time.perf_counter() - t)
        throughput[f"lote_{tam}"] = round(len(X) / float(np.median(tiempos)), 1)

    modelo_ruta = ruta_absoluta(m["ruta"])
    resultado = {
        "escenario": args.etiqueta, "motor": modelo.motor, "modelo": modelo_ruta.name,
        "modelo_kb": round(modelo_ruta.stat().st_size / 1024, 1),
        "hilos": m.get("hilos"), "python": platform.python_version(),
        **recursos_del_contenedor(),
        "proceso_hasta_modelo_listo_s": round(listo - _INICIO_PROCESO, 3),   # imports + carga del modelo
        "carga_s": round(carga_s, 3),
        "imagenes": len(X), "exactitud": round(exactitud, 4),
        "preproceso_ms_por_imagen": round(preproceso_ms, 3),
        "latencia_inferencia": percentiles(inferencia),
        "latencia_punta_a_punta": percentiles(punta_a_punta),
        "fps_max_estimado": round(1000 / float(np.percentile(punta_a_punta, 50)), 1),
        "throughput_img_s": throughput,
        "memoria_pico_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        **memoria_proceso(),
    }
    texto = json.dumps(resultado, ensure_ascii=False, indent=2)
    print(texto)
    if args.salida:
        Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
        Path(args.salida).write_text(texto, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

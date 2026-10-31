"""
benchmark_despliegue.py  —  Simulación de despliegue y medición de rendimiento (Actividad 4.2)
================================================================================================
Ejecuta src/benchmark.py en varios ESCENARIOS y guarda un JSON por escenario en
benchmarks/resultados/. También guarda la salida de cada comando en benchmarks/evidencia/.

Escenarios:
  A) Sin contenedor (en el equipo directamente): Keras y cada variante TFLite.
  B) Contenedor completo (TensorFlow): sin límites y con límites de un equipo modesto.
  C) Contenedor edge (LiteRT + int8): sin límites y con límites de dispositivos pequeños.
  D) Arranque en frío: cuánto tarda `docker run` desde cero hasta dar una predicción,
     y cuánto tarda el servicio web en responder /health.

Los límites de recursos se simulan con Docker:  --cpus (núcleos) y --memory (RAM).
Si el proceso supera la memoria permitida, Docker lo detiene (código 137, "OOM killed"):
eso también es un resultado.

Requisitos: haber construido las imágenes
    docker build -t vision-figuras:1.0-completo .
    docker build -f Dockerfile.edge -t vision-figuras:1.0-edge .
Uso (desde la raíz del proyecto):
    python herramientas/benchmark_despliegue.py
"""

import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "benchmarks" / "resultados"
EVIDENCIA = RAIZ / "benchmarks" / "evidencia"
COMPLETO, EDGE = "vision-figuras:1.0-completo", "vision-figuras:1.0-edge"
DATOS = f"{RAIZ / 'data'}:/app/data:ro"            # los datos se montan en modo solo lectura


def ejecutar(comando, nombre_log):
    """Ejecuta un comando, guarda su salida como evidencia y devuelve (código, salida, segundos)."""
    inicio = time.perf_counter()
    proc = subprocess.run(comando, capture_output=True, text=True)
    segundos = time.perf_counter() - inicio
    texto = f"$ {' '.join(comando)}\n\n{proc.stdout}{proc.stderr}\n[código de salida: {proc.returncode}] " \
            f"[duración: {segundos:.2f} s]\n"
    (EVIDENCIA / nombre_log).write_text(texto, encoding="utf-8")
    return proc.returncode, proc.stdout, segundos


def guardar(nombre, datos):
    (RESULTADOS / f"{nombre}.json").write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    estado = datos.get("estado", "ok")
    resumen = (f"p50 {datos['latencia_punta_a_punta']['p50_ms']} ms | exactitud {datos['exactitud']:.2%} | "
               f"memoria propia {datos.get('memoria_anonima_mb')} MB + librerías {datos.get('memoria_librerias_mb')} MB") \
        if estado == "ok" else estado
    print(f"  {nombre:<34} {resumen}")


def escenario_host(nombre, modelo=None, hilos=None):
    cmd = [sys.executable, "src/benchmark.py", "--etiqueta", nombre]
    if modelo:
        cmd += ["--modelo", modelo, "--config", "config/config_edge.yaml"]
    if hilos:
        cmd += ["--hilos", str(hilos)]
    codigo, salida, _ = ejecutar(cmd, f"bench_{nombre}.log")
    guardar(nombre, {"entorno": "host", **json.loads(salida)} if codigo == 0 else
            {"escenario": nombre, "estado": f"falló (código {codigo})"})


def escenario_docker(nombre, imagen, cpus=None, memoria=None, extra=()):
    cmd = ["docker", "run", "--rm", "-v", DATOS]
    if cpus:
        cmd += [f"--cpus={cpus}"]
    if memoria:
        cmd += [f"--memory={memoria}", f"--memory-swap={memoria}"]   # sin swap: límite real
    cmd += [imagen, "benchmark", "--etiqueta", nombre, *extra]
    codigo, salida, _ = ejecutar(cmd, f"bench_{nombre}.log")
    if codigo == 0:
        datos = {"entorno": "docker", "imagen": imagen, "cpus_docker": cpus, "memoria_docker": memoria,
                 **json.loads(salida[salida.index("{"):])}
    else:
        motivo = "memoria insuficiente: Docker detuvo el proceso (OOM, código 137)" if codigo == 137 \
            else f"falló (código {codigo})"
        datos = {"escenario": nombre, "entorno": "docker", "imagen": imagen, "cpus_docker": cpus,
                 "memoria_docker": memoria, "estado": motivo}
    guardar(nombre, datos)


def arranque_en_frio(nombre, imagen, repeticiones=3):
    """Tiempo total de `docker run` hasta obtener la predicción de UNA imagen (modo CLI)."""
    tiempos = []
    for k in range(repeticiones):
        codigo, _, seg = ejecutar(["docker", "run", "--rm", "-v", DATOS, imagen, "--modo", "imagen",
                                   "--entrada", "data/ejemplos/nuevas/triangulo_1.png"], f"arranque_{nombre}_{k + 1}.log")
        if codigo == 0:
            tiempos.append(seg)
    return round(statistics.median(tiempos), 2) if tiempos else None


def arranque_servicio(nombre, imagen, puerto):
    """Levanta el servicio web y mide cuánto tarda en responder /health (listo para atender)."""
    contenedor = f"bench-{nombre}"
    subprocess.run(["docker", "rm", "-f", contenedor], capture_output=True)
    inicio = time.perf_counter()
    subprocess.run(["docker", "run", "-d", "--name", contenedor, "-p", f"{puerto}:8000", imagen], capture_output=True)
    listo = None
    while time.perf_counter() - inicio < 120:
        try:
            urllib.request.urlopen(f"http://localhost:{puerto}/health", timeout=1)
            listo = time.perf_counter() - inicio
            break
        except OSError:
            time.sleep(0.1)
    # Una petición real de predicción para dejar evidencia de la API funcionando
    respuesta = subprocess.run(["curl", "-s", "-F", f"imagen=@{RAIZ / 'data/ejemplos/nuevas/cuadrado_1.png'}",
                                f"http://localhost:{puerto}/predict"], capture_output=True, text=True).stdout
    logs = subprocess.run(["docker", "logs", contenedor], capture_output=True, text=True)
    (EVIDENCIA / f"servicio_{nombre}.log").write_text(
        f"$ docker run -d -p {puerto}:8000 {imagen}\n\n{logs.stdout}{logs.stderr}\n"
        f"$ curl -F imagen=@cuadrado_1.png http://localhost:{puerto}/predict\n{respuesta}\n", encoding="utf-8")
    subprocess.run(["docker", "rm", "-f", contenedor], capture_output=True)
    return round(listo, 2) if listo else None


def tamano_imagen_mb(imagen):
    salida = subprocess.run(["docker", "image", "inspect", imagen, "--format", "{{.Size}}"],
                            capture_output=True, text=True).stdout.strip()
    return round(int(salida) / 2**20) if salida.isdigit() else None


def main():
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    EVIDENCIA.mkdir(parents=True, exist_ok=True)

    print("A) Sin contenedor")
    escenario_host("host_keras")
    for variante in ["float32", "float16", "int8_pesos", "int8_completo"]:
        escenario_host(f"host_tflite_{variante}", f"models/modelo_figuras_v1_{variante}.tflite")

    print("B) Contenedor completo (TensorFlow)")
    escenario_docker("docker_completo", COMPLETO)
    escenario_docker("docker_completo_1cpu_1gb", COMPLETO, cpus=1, memoria="1g")
    escenario_docker("docker_completo_1cpu_512mb", COMPLETO, cpus=1, memoria="512m")
    escenario_docker("docker_completo_1cpu_256mb", COMPLETO, cpus=1, memoria="256m")
    escenario_docker("docker_completo_1cpu_128mb", COMPLETO, cpus=1, memoria="128m")

    print("C) Contenedor edge (LiteRT)")
    escenario_docker("docker_edge", EDGE)
    escenario_docker("docker_edge_0.5cpu_256mb", EDGE, cpus=0.5, memoria="256m")
    escenario_docker("docker_edge_0.25cpu_128mb", EDGE, cpus=0.25, memoria="128m")
    escenario_docker("docker_edge_0.25cpu_64mb", EDGE, cpus=0.25, memoria="64m")

    print("D) Arranque en frío y servicio web")
    arranque = {}
    for nombre, imagen, puerto in [("completo", COMPLETO, 8100), ("edge", EDGE, 8101)]:
        arranque[nombre] = {"imagen": imagen, "tamano_imagen_mb": tamano_imagen_mb(imagen),
                            "docker_run_hasta_prediccion_s": arranque_en_frio(nombre, imagen),
                            "servicio_hasta_health_s": arranque_servicio(nombre, imagen, puerto)}
        print(f"  {nombre:<34} {arranque[nombre]}")
    (RESULTADOS / "arranque.json").write_text(json.dumps(arranque, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

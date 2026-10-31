"""
empaquetar.py  —  Paquete de entrega profesional (Actividad 4.2)
=================================================================
Genera dist/vision-figuras-v<versión>.zip con la ESTRUCTURA DE ENTREGA MÍNIMA:
lo necesario para que otra persona instale y ejecute el sistema, y nada más.

    vision-figuras-v1.0.0/
    ├── README.md                 instrucciones de instalación y uso
    ├── LEEME_DESPLIEGUE.md       despliegue con Docker (completo y edge)
    ├── requirements*.txt         dependencias con versión fija
    ├── Dockerfile, Dockerfile.edge, docker-compose.yml, .dockerignore, docker/
    ├── config/                   perfiles de configuración (completo y edge)
    ├── models/                   modelo .keras + variantes .tflite + ficha técnica
    ├── src/                      pipeline, API y benchmark
    ├── data/ejemplos/            imágenes para probar
    └── MANIFIESTO.json           lista de archivos con su huella SHA-256

Uso (desde la raíz del proyecto):
    python herramientas/empaquetar.py
"""

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
INCLUIR = ["README.md", "LEEME_DESPLIEGUE.md", "requirements.txt", "requirements-api.txt",
           "requirements-edge.txt", "Dockerfile", "Dockerfile.edge", "docker-compose.yml", ".dockerignore",
           "docker", "config", "models", "src", "data/ejemplos"]
IGNORAR = {"__pycache__", ".ipynb_checkpoints"}


def archivos_del_paquete():
    for elemento in INCLUIR:
        ruta = RAIZ / elemento
        if ruta.is_file():
            yield ruta
        elif ruta.is_dir():
            yield from (p for p in sorted(ruta.rglob("*"))
                        if p.is_file() and not IGNORAR.intersection(p.parts))


def main():
    version = json.loads((RAIZ / "models" / "model_card.json").read_text(encoding="utf-8"))["version"]
    nombre = f"vision-figuras-v{version}"
    destino = RAIZ / "dist" / f"{nombre}.zip"
    destino.parent.mkdir(exist_ok=True)

    manifiesto = {"paquete": nombre, "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "archivos": {}}
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for ruta in archivos_del_paquete():
            relativa = ruta.relative_to(RAIZ).as_posix()
            z.write(ruta, f"{nombre}/{relativa}")
            manifiesto["archivos"][relativa] = {"bytes": ruta.stat().st_size,
                                                "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest()}
        z.writestr(f"{nombre}/MANIFIESTO.json", json.dumps(manifiesto, ensure_ascii=False, indent=2))

    total = sum(a["bytes"] for a in manifiesto["archivos"].values())
    print(f"Paquete: {destino.relative_to(RAIZ)}")
    print(f"Archivos: {len(manifiesto['archivos'])} | contenido: {total / 2**20:.1f} MB | "
          f"zip: {destino.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()

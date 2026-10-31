"""
generar_datos_prueba.py  —  Datos para probar el pipeline operativo
====================================================================
Crea, dentro de data/:

  conjunto_prueba/<clase>/   Las MISMAS 180 imágenes de prueba de la Actividad 3.3
                             (se repite la división 70/15/15 con la misma semilla).
                             Sirve para comprobar que el pipeline da los mismos resultados.

  ejemplos/nuevas/           Figuras nuevas del mismo tipo que el dataset.
  ejemplos/variaciones/      Mismo tipo de figura, pero en otro formato: a color, grandes,
                             rectangulares, con transparencia, JPG comprimido, TIFF de 16 bits.
  ejemplos/fuera_de_dominio/ Figuras "a mano alzada" y con colores invertidos (fondo oscuro):
                             cosas que el modelo NUNCA vio al entrenar.
  ejemplos/sin_clase/        Figuras que no son ninguna de las 3 clases (pentágono, estrella...).
  ejemplos/invalidos/        Archivos que deben rechazarse: corrupto, vacío, texto, diminuto.

  video/figuras.mp4          Video sintético (simula una cámara) con figuras que aparecen,
  video/figuras_etiquetas.csv se mueven y giran, más la etiqueta real de cada cuadro.

Uso (desde la raíz del proyecto):
    python herramientas/generar_datos_prueba.py
"""

import csv
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "entrenamiento"))
from generar_dataset import CLASES, dibujar_figura, puntos_poligono  # noqa: E402

DATA = RAIZ / "data"
SEMILLA = 42


# ---------------------------------------------------------------------------- 1
def copiar_conjunto_prueba():
    """Repite EXACTAMENTE la división de la Actividad 3.3 para obtener sus 180 imágenes de prueba."""
    carpeta = RAIZ / "entrenamiento" / "dataset_figuras"
    if not carpeta.exists():
        raise SystemExit("Primero genera el dataset: python entrenamiento/generar_dataset.py "
                         "--salida entrenamiento/dataset_figuras")
    clases = sorted(p.name for p in carpeta.iterdir() if p.is_dir())
    rutas, y = [], []
    for indice, clase in enumerate(clases):                 # mismo orden que en la 3.3
        for archivo in sorted((carpeta / clase).glob("*.png")):
            rutas.append(archivo)
            y.append(indice)
    rutas, y = np.array(rutas), np.array(y)
    # Las dos divisiones con los mismos parámetros que la 3.3 (70 % / 15 % / 15 %)
    _, r_resto, _, y_resto = train_test_split(rutas, y, test_size=0.30, stratify=y, random_state=SEMILLA)
    _, r_prueba, _, _ = train_test_split(r_resto, y_resto, test_size=0.50, stratify=y_resto,
                                         random_state=SEMILLA)
    destino = DATA / "conjunto_prueba"
    shutil.rmtree(destino, ignore_errors=True)
    for ruta in r_prueba:
        (destino / ruta.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ruta, destino / ruta.parent.name / ruta.name)
    print(f"conjunto_prueba: {len(r_prueba)} imágenes (las mismas de la Actividad 3.3)")


# ---------------------------------------------------------------------------- 2
def dibujar_grande(clase, lado, fondo, tinta, radio, cx, cy, angulo, rellena, grosor, ruido=12):
    """Igual que dibujar_figura() pero a cualquier tamaño y en color (fondo y tinta en RGB)."""
    img = Image.new("RGB", (lado, lado), fondo)
    lapiz = ImageDraw.Draw(img)
    relleno = tinta if rellena else None
    if clase == "circulo":
        lapiz.ellipse([cx - radio, cy - radio, cx + radio, cy + radio], fill=relleno, outline=tinta, width=grosor)
    else:
        lados = 4 if clase == "cuadrado" else 3
        lapiz.polygon(puntos_poligono(cx, cy, radio, lados, angulo), fill=relleno, outline=tinta, width=grosor)
    arr = np.array(img).astype(np.float32) + np.random.normal(0, ruido, (lado, lado, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def figura_mano_alzada(clase, lado=400):
    """Simula un dibujo hecho a mano: trazo delgado y tembloroso, esquinas imperfectas, papel con sombra."""
    papel = np.full((lado, lado), 245, np.float32)
    papel -= np.linspace(0, 35, lado)[None, :]               # sombra suave de izquierda a derecha
    img = Image.fromarray(papel.astype(np.uint8)).convert("RGB")
    lapiz = ImageDraw.Draw(img)
    cx, cy = lado / 2 + random.uniform(-30, 30), lado / 2 + random.uniform(-30, 30)
    radio = random.uniform(110, 150)
    if clase == "circulo":
        vertices = [(cx + radio * math.cos(a) * random.uniform(0.95, 1.05),
                     cy + radio * math.sin(a) * random.uniform(0.92, 1.08))
                    for a in np.linspace(0, 2 * math.pi, 40)]
    else:
        lados = 4 if clase == "cuadrado" else 3
        vertices = [(x + random.uniform(-12, 12), y + random.uniform(-12, 12))
                    for x, y in puntos_poligono(cx, cy, radio, lados, random.uniform(0, 2 * math.pi))]
        vertices.append(vertices[0])
    # Cada lado se dibuja como una línea temblorosa (muchos segmentos con desvío aleatorio)
    for (x0, y0), (x1, y1) in zip(vertices[:-1], vertices[1:]):
        pasos = 12
        puntos = [(x0 + (x1 - x0) * t / pasos + random.uniform(-3, 3),
                   y0 + (y1 - y0) * t / pasos + random.uniform(-3, 3)) for t in range(pasos + 1)]
        lapiz.line(puntos, fill=(30, 30, 90), width=random.randint(4, 6), joint="curve")
    return img


def crear_ejemplos():
    random.seed(7)
    np.random.seed(7)
    base = DATA / "ejemplos"
    shutil.rmtree(base, ignore_errors=True)

    # a) Nuevas: mismo generador que el dataset (pero otra semilla -> imágenes nunca vistas)
    (base / "nuevas").mkdir(parents=True)
    for clase in CLASES:
        for k in range(1, 4):
            dibujar_figura(clase).save(base / "nuevas" / f"{clase}_{k}.png")

    # b) Variaciones de formato (mismo "tipo" de figura que el dataset)
    v = base / "variaciones"
    v.mkdir()
    dibujar_grande("triangulo", 320, (235, 228, 210), (150, 30, 30), 110, 160, 165, 0.4, True, 8) \
        .save(v / "triangulo_color_320px.png")
    dibujar_grande("circulo", 320, (210, 230, 250), (20, 60, 140), 100, 150, 170, 0, False, 12) \
        .save(v / "circulo_color_320px.jpg", quality=35)                      # JPG muy comprimido
    cuadrado = dibujar_grande("cuadrado", 320, (250, 250, 250), (60, 60, 60), 120, 160, 160, 0.3, True, 8)
    cuadrado.resize((480, 240)).save(v / "cuadrado_rectangular_480x240.png")  # se "aplasta" al reducir a 64x64
    # PNG con transparencia: figura opaca sobre fondo 100 % transparente
    rgba = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    ImageDraw.Draw(rgba).polygon(puntos_poligono(128, 128, 90, 4, 0.8), fill=(40, 40, 40, 255))
    rgba.save(v / "cuadrado_transparente.png")
    # TIFF de 16 bits (profundidad poco común en cámaras industriales)
    gris16 = (np.array(dibujar_figura("circulo")).astype(np.uint16) * 257)
    cv2.imwrite(str(v / "circulo_16bits.tif"), gris16)

    # c) Fuera de dominio: lo que el modelo nunca vio
    f = base / "fuera_de_dominio"
    f.mkdir()
    for clase in CLASES:
        for k in range(1, 3):
            figura_mano_alzada(clase).save(f / f"{clase}_mano_alzada_{k}.png")
        invertida = 255 - np.array(dibujar_figura(clase))                     # figura clara, fondo oscuro
        Image.fromarray(invertida).save(f / f"{clase}_invertida.png")

    # e) Sin clase: figuras que NO son círculo, cuadrado ni triángulo.
    #    Un clasificador de "clases cerradas" está obligado a elegir una de sus 3 clases.
    s = base / "sin_clase"
    s.mkdir()
    for nombre, lados in [("pentagono", 5), ("hexagono", 6)]:
        img = Image.new("L", (64, 64), 235)
        ImageDraw.Draw(img).polygon(puntos_poligono(32, 32, 22, lados, 0.3), fill=40)
        img.save(s / f"{nombre}.png")
    estrella = Image.new("L", (64, 64), 235)
    puntas = [(32 + (24 if k % 2 == 0 else 10) * math.cos(-math.pi / 2 + k * math.pi / 5),
               32 + (24 if k % 2 == 0 else 10) * math.sin(-math.pi / 2 + k * math.pi / 5)) for k in range(10)]
    ImageDraw.Draw(estrella).polygon(puntas, fill=40)
    estrella.save(s / "estrella.png")
    dos = Image.new("L", (64, 64), 235)
    ImageDraw.Draw(dos).ellipse([4, 8, 28, 32], fill=40)
    ImageDraw.Draw(dos).polygon(puntos_poligono(46, 44, 14, 3, -math.pi / 2), fill=40)
    dos.save(s / "circulo_y_triangulo.png")
    Image.new("L", (64, 64), 235).save(s / "fondo_vacio.png")

    # d) Inválidos: deben rechazarse sin detener el pipeline
    i = base / "invalidos"
    i.mkdir()
    (i / "corrupta.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(np.random.randint(0, 256, 500, dtype=np.uint8)))
    (i / "vacia.jpg").write_bytes(b"")
    (i / "notas.txt").write_text("Esto no es una imagen.\n", encoding="utf-8")
    Image.new("L", (8, 8), 255).save(i / "diminuta_8x8.png")

    total = sum(1 for p in base.rglob("*") if p.is_file())
    print(f"ejemplos: {total} archivos (nuevas, variaciones, fuera_de_dominio, invalidos)")


# ---------------------------------------------------------------------------- 3
def crear_video(lado=240, fps=15, escenas=12, cuadros_escena=30, cuadros_vacios=8):
    """Video que simula una cámara fija: figuras que aparecen, se mueven, giran o entran por el borde,
    separadas por cuadros vacíos (sin figura). Guarda la etiqueta real de cada cuadro."""
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    escala = lado / 64                                     # mismas proporciones que el dataset
    destino = DATA / "video"
    destino.mkdir(parents=True, exist_ok=True)
    escritor = cv2.VideoWriter(str(destino / "figuras.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (lado, lado))
    etiquetas, n = [], 0
    comportamientos = ["estatica", "se_mueve", "gira", "entra_por_borde"]

    def escribir(img_pil, clase, escena, comportamiento):
        nonlocal n
        escritor.write(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))
        etiquetas.append({"cuadro": n, "clase_real": clase, "escena": escena, "comportamiento": comportamiento})
        n += 1

    for escena in range(escenas):
        fondo = random.randint(200, 255)
        # Cuadros vacíos: solo fondo con ruido (no hay figura que clasificar)
        for _ in range(cuadros_vacios):
            vacio = np.clip(np.full((lado, lado, 3), fondo, np.float32)
                            + np.random.normal(0, 12, (lado, lado, 1)), 0, 255).astype(np.uint8)
            escribir(Image.fromarray(vacio), "ninguna", escena, "vacio")

        clase = CLASES[escena % 3]
        comp = comportamientos[(escena // 3) % 4]
        radio = random.randint(14, 24) * escala
        tono = random.randint(0, 110)
        rellena, grosor = random.random() < 0.5, round(random.randint(2, 4) * escala)
        angulo = random.uniform(0, 2 * math.pi)
        cx, cy = lado / 2 + random.uniform(-20, 20), lado / 2 + random.uniform(-20, 20)
        if comp == "se_mueve":
            x0, x1 = radio + 5, lado - radio - 5
        elif comp == "entra_por_borde":
            x0, x1 = -radio * 0.6, lado / 2           # empieza 80 % fuera del cuadro y entra
        for t in range(cuadros_escena):
            avance = t / (cuadros_escena - 1)
            x = x0 + (x1 - x0) * avance if comp in ("se_mueve", "entra_por_borde") else cx
            a = angulo + (avance * math.pi if comp == "gira" else 0)
            img = dibujar_grande(clase, lado, (fondo,) * 3, (tono,) * 3, radio, x, cy, a, rellena, grosor)
            escribir(img, clase, escena, comp)
    escritor.release()

    with open(destino / "figuras_etiquetas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cuadro", "clase_real", "escena", "comportamiento"])
        w.writeheader()
        w.writerows(etiquetas)
    print(f"video: {n} cuadros de {lado}x{lado} a {fps} FPS ({n / fps:.1f} s) + etiquetas por cuadro")


if __name__ == "__main__":
    copiar_conjunto_prueba()
    crear_ejemplos()
    crear_video()

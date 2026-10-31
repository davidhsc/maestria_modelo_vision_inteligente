"""
pipeline_operativo.py  —  Script principal: orquesta TODAS las fases
=====================================================================

    Entrada -> Preprocesamiento -> Inferencia -> Posprocesamiento -> Salida
    (Fase 3)   (Fase 4)            (Fases 5-6)   (Fase 7)            (Fases 8-9)

Tres modos de ejecución:

  imagen       Una sola imagen.
               python src/pipeline_operativo.py --modo imagen --entrada data/ejemplos/nuevas/cuadrado_1.png

  lote         Todas las imágenes de una carpeta, procesadas en lotes.
               Si la carpeta tiene una subcarpeta por clase, también calcula la exactitud.
               python src/pipeline_operativo.py --modo lote --entrada data/conjunto_prueba

  tiempo-real  Cuadro por cuadro desde un video (o una cámara: --entrada 0),
               respetando el ritmo de llegada de los cuadros (FPS) como lo haría una cámara.
               python src/pipeline_operativo.py --modo tiempo-real --entrada data/video/figuras.mp4

Escribe en outputs/<run_id>/: resultados.csv, resumen.json y las imágenes/video anotados.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

from configuracion import cargar_config, ruta_absoluta
from entrada import ErrorEntrada, abrir_fuente_video, leer_imagen, listar_imagenes
from modelo import ErrorModelo, cargar_modelo, dispositivo_disponible, inferir, leer_clases
from postprocesamiento import INCIERTO, SuavizadorTemporal, interpretar
from preprocesamiento import crear_lote, preprocesar
from registro import configurar_log, nuevo_run_id, registrar_ejecucion
from salida import (anotar, crear_escritor_video, guardar_csv, guardar_imagen,
                    guardar_json, resumen_latencias)


class PipelineOperativo:
    """Agrupa el modelo ya cargado + la configuración, y ofrece un método por modo."""

    def __init__(self, ruta_config=None, modo="interactivo", umbral=None, carpeta_salida=None):
        self.cfg = cargar_config(ruta_config)
        if umbral is not None:                                       # el CLI puede sobrescribir el config
            self.cfg["postprocesamiento"]["umbral_confianza"] = umbral
        self.salida_base = ruta_absoluta(carpeta_salida or self.cfg["salida"]["carpeta"])

        self.run_id = nuevo_run_id(modo)
        self.carpeta_run = self.salida_base / self.run_id
        self.log = configurar_log(self.run_id, self.salida_base)
        self.log.info(f"=== Ejecución {self.run_id} ===")

        # ----- Fase 5: carga y validación del modelo -----
        p, m = self.cfg["preprocesamiento"], self.cfg["modelo"]
        self.forma = (p["alto"], p["ancho"], p["canales"])
        self.clases = leer_clases(ruta_absoluta(m["clases"]))
        self.modelo, self.tarjeta = cargar_modelo(
            ruta_absoluta(m["ruta"]), ruta_absoluta(m["tarjeta"]), self.forma,
            verificar_hash=m.get("verificar_hash", True), log=self.log)
        if self.tarjeta.get("clases") and self.tarjeta["clases"] != self.clases:
            raise ErrorModelo("clases.txt no coincide con las clases de model_card.json")

    # ------------------------------------------------------------------ utilidades
    def _preprocesar(self, img):
        p = self.cfg["preprocesamiento"]
        return preprocesar(img, p["alto"], p["ancho"], p["divisor"])

    def _interpretar(self, probabilidades):
        pp = self.cfg["postprocesamiento"]
        return interpretar(probabilidades, self.clases, pp["umbral_confianza"], pp["top_k"])

    def _leer(self, ruta):
        e = self.cfg["entrada"]
        return leer_imagen(ruta, e["extensiones_validas"], e["tamano_minimo"])

    def _cerrar(self, modo, entrada, resumen):
        """Guarda resumen.json y agrega la línea de auditoría."""
        resumen = {"run_id": self.run_id, "modo": modo, "entrada": str(entrada),
                   "modelo": self.tarjeta.get("archivo"), "version_modelo": self.tarjeta.get("version"),
                   "dispositivo": dispositivo_disponible(),
                   "umbral_confianza": self.cfg["postprocesamiento"]["umbral_confianza"], **resumen}
        guardar_json(resumen, self.carpeta_run / "resumen.json")
        lat = resumen.get("latencia_por_imagen", {})
        registrar_ejecucion(self.salida_base, {
            "run_id": self.run_id, "modo": modo, "entrada": str(entrada),
            "version_modelo": resumen["version_modelo"], "procesadas": resumen.get("procesadas", 0),
            "errores": resumen.get("errores", 0), "exactitud": resumen.get("exactitud", ""),
            "latencia_p50_ms": lat.get("p50_ms", ""), "duracion_s": resumen.get("duracion_s", "")})
        self.log.info(f"Resultados en: {self.carpeta_run}")
        return resumen

    # ------------------------------------------------------------------ modo IMAGEN
    def predecir_imagen(self, ruta, guardar=True):
        """Pipeline completo para UNA imagen. Devuelve el diccionario de resultado."""
        inicio = time.perf_counter()
        img = self._leer(ruta)                                          # Fase 3
        x = crear_lote([self._preprocesar(img)])                        # Fase 4  -> (1, 64, 64, 1)
        probabilidades, ms = inferir(self.modelo, x)                    # Fase 6
        resultado = {"archivo": Path(ruta).name, **self._interpretar(probabilidades[0]),
                     "inferencia_ms": round(ms, 2),
                     "total_ms": round((time.perf_counter() - inicio) * 1000, 2)}   # Fase 7
        self.log.info(f"{resultado['archivo']}: {resultado['clase']} ({resultado['confianza']:.1%}) "
                      f"[{resultado['estado']}] en {resultado['total_ms']:.1f} ms")
        if guardar:                                                     # Fase 8
            guardar_imagen(anotar(img, resultado), self.carpeta_run / f"{Path(ruta).stem}_anotada.png")
            guardar_json(resultado, self.carpeta_run / "resultado.json")
            self._cerrar("imagen", ruta, {"procesadas": 1, "errores": 0,
                                          "latencia_por_imagen": {"p50_ms": resultado["total_ms"]},
                                          "resultado": resultado})
        return resultado

    # ------------------------------------------------------------------ modo LOTE
    def ejecutar_lote(self, carpeta, tamano_lote=None, guardar_imagenes=False):
        """Procesa todas las imágenes de una carpeta en lotes. Las inválidas se registran y se omiten."""
        tamano_lote = tamano_lote or self.cfg["inferencia"]["tamano_lote"]
        archivos = listar_imagenes(carpeta)
        self.log.info(f"Modo lote: {len(archivos)} archivos en '{carpeta}', lotes de {tamano_lote}")
        inicio = time.perf_counter()
        filas, errores, tiempos_lote = [], [], []

        def procesar(pendientes):
            """Ejecuta UN lote: pendientes = lista de (ruta, imagen_cruda)."""
            x = crear_lote([self._preprocesar(img) for _, img in pendientes])
            probabilidades, ms = inferir(self.modelo, x)
            tiempos_lote.append((len(pendientes), ms))
            for (ruta, img), prob in zip(pendientes, probabilidades):
                r = self._interpretar(prob)
                # Si la carpeta padre se llama como una clase, la usamos como "verdad"
                real = ruta.parent.name if ruta.parent.name in self.clases else ""
                filas.append({"archivo": str(ruta.relative_to(carpeta)), "clase": r["clase"],
                              "confianza": r["confianza"], "estado": r["estado"], "clase_real": real,
                              "correcto": (r["clase"] == real) if real else "",
                              **{f"p_{c}": round(float(p), 4) for c, p in zip(self.clases, prob)}})
                if guardar_imagenes:
                    guardar_imagen(anotar(img, r), self.carpeta_run / "imagenes" / f"{ruta.stem}.png")

        pendientes = []
        for ruta in archivos:
            try:
                pendientes.append((ruta, self._leer(ruta)))              # Fase 3 (con validación)
            except ErrorEntrada as e:
                self.log.warning(f"Se omite '{ruta.name}': {e}")
                errores.append({"archivo": str(ruta.relative_to(carpeta)), "error": str(e)})
                continue
            if len(pendientes) == tamano_lote:                           # lote lleno -> procesar
                procesar(pendientes)
                pendientes = []
        if pendientes:                                                   # último lote incompleto
            procesar(pendientes)

        duracion = time.perf_counter() - inicio
        guardar_csv(filas, self.carpeta_run / "resultados.csv")
        guardar_csv(errores, self.carpeta_run / "errores.csv")

        evaluadas = [f for f in filas if f["clase_real"]]
        ms_por_imagen = [ms / n for n, ms in tiempos_lote for _ in range(n)]
        resumen = {
            "archivos": len(archivos), "procesadas": len(filas), "errores": len(errores),
            "baja_confianza": sum(f["estado"] == "baja_confianza" for f in filas),
            "conteo_por_clase": {c: sum(f["clase"] == c for f in filas) for c in self.clases},
            "tamano_lote": tamano_lote, "lotes": len(tiempos_lote),
            "latencia_por_imagen": resumen_latencias(ms_por_imagen),
            "throughput_img_s": round(len(filas) / duracion, 1) if duracion else 0,
            "duracion_s": round(duracion, 2),
        }
        if evaluadas:
            resumen["exactitud"] = round(sum(f["correcto"] for f in evaluadas) / len(evaluadas), 4)
            resumen["evaluadas"] = len(evaluadas)
        self.log.info(f"Procesadas {resumen['procesadas']} | errores {resumen['errores']} | "
                      f"baja confianza {resumen['baja_confianza']} | {resumen['throughput_img_s']} img/s"
                      + (f" | exactitud {resumen['exactitud']:.2%}" if evaluadas else ""))
        return self._cerrar("lote", carpeta, resumen)

    # ------------------------------------------------------------------ modo TIEMPO REAL
    def ejecutar_tiempo_real(self, fuente, fps=None, sin_espera=False, etiquetas=None,
                             mostrar=False, guardar_video=True):
        """Procesa un flujo de video cuadro por cuadro, simulando la llegada en tiempo real.

        Con una cámara real, los cuadros llegan cada 1/fps segundos aunque el programa
        no esté listo. Si procesar un cuadro tarda más que eso, los cuadros que llegaron
        mientras tanto se PIERDEN. Aquí se simula exactamente eso con un reloj.
        sin_espera=True procesa todo lo más rápido posible (mide la capacidad máxima).
        """
        tr = self.cfg["tiempo_real"]
        fps = fps or tr["fps_objetivo"]
        presupuesto_ms = 1000 / fps                       # tiempo disponible por cuadro
        cuadros, fps_origen = abrir_fuente_video(fuente)
        verdad = self._leer_etiquetas_video(etiquetas)
        suavizador = SuavizadorTemporal(tr["ventana_suavizado"])
        self.log.info(f"Modo tiempo real: fuente '{fuente}' ({fps_origen:.0f} FPS de origen), "
                      f"objetivo {fps} FPS = {presupuesto_ms:.1f} ms por cuadro"
                      + (" | SIN espera (máxima velocidad)" if sin_espera else ""))

        filas, escritor, descartados = [], None, 0
        inicio = time.perf_counter()
        for i, cuadro in enumerate(cuadros):
            # --- Simulación del reloj de la cámara ---
            if not sin_espera:
                llegada = inicio + i / fps                # momento en que "llega" este cuadro
                ahora = time.perf_counter()
                if ahora < llegada:
                    time.sleep(llegada - ahora)           # aún no llega: esperamos
                elif ahora - llegada > 1 / fps:
                    descartados += 1                      # ya llegó el siguiente: este se perdió
                    continue

            t0 = time.perf_counter()
            x = crear_lote([self._preprocesar(cuadro)])                   # Fase 4
            probabilidades, ms_inf = inferir(self.modelo, x)              # Fase 6
            r = self._interpretar(probabilidades[0])                      # Fase 7
            estable = suavizador.actualizar(r["clase"], confiable=(r["estado"] == "ok"))
            ms_total = (time.perf_counter() - t0) * 1000

            real = verdad.get(i, "")
            filas.append({"cuadro": i, "clase": r["clase"], "confianza": r["confianza"],
                          "estado": r["estado"], "clase_suavizada": estable, "clase_real": real,
                          "inferencia_ms": round(ms_inf, 2), "total_ms": round(ms_total, 2),
                          "fuera_de_presupuesto": ms_total > presupuesto_ms})

            if guardar_video or mostrar:                                  # Fase 8
                fps_actual = len(filas) / max(time.perf_counter() - inicio, 1e-6)
                visible = {**r, "clase": estable}
                if estable == INCIERTO:                                   # nada claro en escena
                    visible = {**r, "clase": "sin figura clara", "estado": "baja_confianza"}
                anotado = anotar(cuadro, visible,
                                 extra=f"cuadro {i}  {ms_total:.1f} ms  {fps_actual:.1f} FPS")
                if guardar_video:
                    if escritor is None:
                        alto, ancho = anotado.shape[:2]
                        escritor = crear_escritor_video(self.carpeta_run / "video_anotado.mp4", fps, (ancho, alto))
                    escritor.write(anotado)
                if mostrar:
                    cv2.imshow("Pipeline en tiempo real (q para salir)", anotado)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        if escritor is not None:
            escritor.release()
        if mostrar:
            cv2.destroyAllWindows()

        duracion = time.perf_counter() - inicio
        guardar_csv(filas, self.carpeta_run / "resultados.csv")
        resumen = {
            "fps_objetivo": fps, "sin_espera": sin_espera,
            "cuadros_recibidos": len(filas) + descartados, "procesadas": len(filas),
            "cuadros_descartados": descartados, "errores": 0,
            "fps_efectivo": round(len(filas) / duracion, 1) if duracion else 0,
            "latencia_por_imagen": resumen_latencias([f["total_ms"] for f in filas]),
            "fuera_de_presupuesto": sum(f["fuera_de_presupuesto"] for f in filas),
            "duracion_s": round(duracion, 2),
        }
        evaluables = [f for f in filas if f["clase_real"] in self.clases]
        if evaluables:
            resumen["exactitud"] = round(sum(f["clase"] == f["clase_real"] for f in evaluables) / len(evaluables), 4)
            resumen["exactitud_suavizada"] = round(
                sum(f["clase_suavizada"] == f["clase_real"] for f in evaluables) / len(evaluables), 4)
            resumen["evaluadas"] = len(evaluables)
        vacios = [f for f in filas if f["clase_real"] == "ninguna"]       # cuadros sin figura
        if vacios:
            resumen["vacios_marcados_inciertos"] = round(
                sum(f["clase_suavizada"] == INCIERTO for f in vacios) / len(vacios), 4)
        self.log.info(f"Cuadros procesados {resumen['procesadas']} / descartados {descartados} | "
                      f"{resumen['fps_efectivo']} FPS efectivos | p95 {resumen['latencia_por_imagen'].get('p95_ms')} ms"
                      + (f" | exactitud {resumen['exactitud']:.2%} (suavizada {resumen['exactitud_suavizada']:.2%})"
                         if evaluables else ""))
        return self._cerrar("tiempo-real", fuente, resumen)

    @staticmethod
    def _leer_etiquetas_video(ruta):
        """CSV opcional con columnas 'cuadro,clase_real' para evaluar el video."""
        if not ruta:
            return {}
        with open(ruta, encoding="utf-8") as f:
            return {int(fila["cuadro"]): fila["clase_real"] for fila in csv.DictReader(f)}


# ====================================================================== CLI
def crear_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline operativo del clasificador de figuras (círculo, cuadrado, triángulo).")
    parser.add_argument("--modo", choices=["imagen", "lote", "tiempo-real"], required=True)
    parser.add_argument("--entrada", required=True,
                        help="imagen, carpeta, video o número de cámara (0) según el modo")
    parser.add_argument("--config", default=None, help="ruta al config.yaml (opcional)")
    parser.add_argument("--umbral", type=float, default=None, help="umbral de confianza (0 a 1)")
    parser.add_argument("--salida", default=None, help="carpeta de resultados (por defecto outputs/)")
    parser.add_argument("--lote", type=int, default=None, help="[lote] imágenes por lote")
    parser.add_argument("--guardar-imagenes", action="store_true", help="[lote] guardar imágenes anotadas")
    parser.add_argument("--fps", type=float, default=None, help="[tiempo-real] FPS de llegada simulados")
    parser.add_argument("--sin-espera", action="store_true", help="[tiempo-real] procesar a máxima velocidad")
    parser.add_argument("--etiquetas", default=None, help="[tiempo-real] CSV cuadro,clase_real para evaluar")
    parser.add_argument("--mostrar", action="store_true", help="[tiempo-real] abrir ventana (requiere pantalla)")
    return parser


def main(argv=None):
    args = crear_parser().parse_args(argv)
    try:
        pipe = PipelineOperativo(args.config, args.modo, args.umbral, args.salida)
        if args.modo == "imagen":
            r = pipe.predecir_imagen(args.entrada)
            print(f"\n{'=' * 50}\nRESULTADO: {r['clase']}  ({r['confianza']:.2%})  [{r['estado']}]")
            print("  ".join(f"{t['clase']}={t['prob']:.2%}" for t in r["top"]) + f"\n{'=' * 50}")
        elif args.modo == "lote":
            pipe.ejecutar_lote(Path(args.entrada), args.lote, args.guardar_imagenes)
        else:
            pipe.ejecutar_tiempo_real(args.entrada, args.fps, args.sin_espera, args.etiquetas, args.mostrar)
        return 0
    except (ErrorEntrada, ErrorModelo, FileNotFoundError) as e:
        # Errores esperados: mensaje claro y código de salida distinto de 0 (útil para scripts y Docker)
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

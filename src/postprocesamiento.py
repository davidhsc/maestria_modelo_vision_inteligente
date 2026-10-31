"""
postprocesamiento.py  —  Fase 7: posprocesamiento
==================================================
La red solo entrega números, por ejemplo [0.02, 0.95, 0.03].
Aquí se convierten en una respuesta que una persona o un sistema pueda usar:

    {"clase": "cuadrado", "confianza": 0.95, "estado": "ok", "top": [...]}

Además se aplica un UMBRAL DE CONFIANZA: si la red no está suficientemente
segura, la predicción se marca como "baja_confianza" para que alguien la revise.
"""

from collections import Counter, deque

import numpy as np


def interpretar(probabilidades, clases, umbral=0.80, top_k=3):
    """Convierte el vector de probabilidades de UNA imagen en un diccionario legible."""
    # Índices de mayor a menor probabilidad. kind="stable" hace que, si hay EMPATE
    # (posible con modelos cuantizados a 8 bits), gane siempre la primera clase, igual que argmax.
    orden = np.argsort(-np.asarray(probabilidades), kind="stable")
    indice = int(orden[0])
    confianza = float(probabilidades[indice])
    return {
        "clase": clases[indice],
        "confianza": round(confianza, 4),
        "estado": "ok" if confianza >= umbral else "baja_confianza",
        "top": [{"clase": clases[i], "prob": round(float(probabilidades[i]), 4)} for i in orden[:top_k]],
    }


INCIERTO = "incierto"   # etiqueta cuando no hay suficientes votos confiables


class SuavizadorTemporal:
    """Suavizado para video: votación entre los últimos N cuadros.

    En video, la predicción puede "parpadear" (cambiar de clase un cuadro y
    regresar al siguiente) por ruido o por el movimiento. Votar entre los
    últimos N cuadros da una etiqueta más estable, a cambio de reaccionar
    unos cuadros más tarde cuando la escena cambia de verdad.

    solo_confiables=True: los cuadros de BAJA confianza no votan.
    Así, cuando no hay figura (la red duda), la salida es "incierto" en lugar
    de arrastrar una clase equivocada a los cuadros siguientes.
    """

    def __init__(self, ventana=5, solo_confiables=True):
        self.ultimas = deque(maxlen=max(1, ventana))   # guarda solo las N más recientes
        self.solo_confiables = solo_confiables

    def actualizar(self, clase, confiable=True):
        self.ultimas.append(clase if (confiable or not self.solo_confiables) else None)
        votos = Counter(c for c in self.ultimas if c is not None)
        if not votos or sum(votos.values()) <= len(self.ultimas) // 2:
            return INCIERTO                              # mayoría de cuadros dudosos
        return votos.most_common(1)[0][0]                # la clase más votada

"""
stats.py — Números de la pantalla de métricas.

Mide el ACUERDO entre el sistema y el médico y el uso propio; no mide al
modelo (eso está en MODEL_METADATA, medido contra el test anotado).
"""

from collections import Counter
from datetime import timedelta
from typing import Optional

import numpy as np

from config.tiempo import a_local, ahora_local

# Por debajo de este n no se publica porcentaje, sólo el recuento crudo.
N_MINIMO = 10

ETIQUETAS_URGENCIA = {
    "HIGH": "Prioritario",
    "MEDIUM": "Revisar",
    "LOW_BORDERLINE": "Límite",
    "LOW": "Sin hallazgos",
}


def _tasa(parte: int, total: int) -> dict:
    """Proporción con su denominador y la marca de si alcanza N_MINIMO."""
    return {
        "parte": parte,
        "total": total,
        "tasa": (parte / total) if total else 0.0,
        "suficiente": total >= N_MINIMO,
    }


# Nombre público: todo porcentaje de la app (también el panel de admin) sale de acá.
tasa = _tasa


# ── Tablero clásico ──────────────────────────────────────────────────────────


def calculate_agreement_rate(feedbacks: list[dict]) -> dict:
    total = len(feedbacks)
    agreed = sum(1 for f in feedbacks if f.get("agreed"))
    return {
        "total": total,
        "agreed": agreed,
        "disagreed": total - agreed,
        "rate": agreed / total if total else 0.0,
    }


def analyses_per_day(analyses: list[dict], days: int = 30) -> dict:
    """Análisis por día en los últimos `days` días, agrupados por día LOCAL (no UTC)."""
    hoy = ahora_local().date()
    inicio = hoy - timedelta(days=days - 1)

    por_dia = Counter()
    for a in analyses:
        local = a_local(a.get("timestamp"))
        if local is not None and inicio <= local.date() <= hoy:
            por_dia[local.date()] += 1

    dias = [inicio + timedelta(days=i) for i in range(days)]
    return {
        "labels": [d.strftime("%Y-%m-%d") for d in dias],
        "values": [por_dia[d] for d in dias],
    }


def confidence_distribution(analyses: list[dict], bins: int = 10) -> dict:
    """Histograma del score máximo por análisis, en escala 0-1."""
    confidences = [a.get("max_detection_confidence", 0) for a in analyses]
    if not confidences:
        return {"labels": [], "values": []}

    counts, bordes = np.histogram(confidences, bins=bins, range=(0, 1))
    coma = lambda x: f"{x:.1f}".replace(".", ",")  # noqa: E731
    labels = [f"{coma(bordes[i])}–{coma(bordes[i + 1])}" for i in range(len(counts))]
    return {"labels": labels, "values": counts.tolist()}


def dashboard_summary(analyses: list[dict], feedbacks: list[dict]) -> dict:
    total = len(analyses)
    agreement = calculate_agreement_rate(feedbacks)
    suma_conf = sum(a.get("max_detection_confidence", 0) for a in analyses)
    return {
        "total_analyses": total,
        "total_detections": sum(1 for a in analyses if a.get("is_abnormal", False)),
        "avg_confidence": round(suma_conf / total if total else 0.0, 3),
        "agreement_rate": round(agreement["rate"], 3),
        "agreement_details": agreement,
    }


def urgency_distribution(analyses: list[dict]) -> dict:
    """Análisis por urgencia, siempre en el orden HIGH, MEDIUM, LOW."""
    orden = ["HIGH", "MEDIUM", "LOW"]
    conteo = Counter(a.get("urgency") or "LOW" for a in analyses)
    return {
        "labels": [ETIQUETAS_URGENCIA[u] for u in orden],
        "values": [conteo[u] for u in orden],
    }


# ── Métricas de práctica ─────────────────────────────────────────────────────


def franjas_de_seguridad(umbral_dibujo: float, corte_aviso: float) -> list[dict]:
    """Franjas del score: sin marca / bajo el corte / sobre el corte / alta."""
    medio = corte_aviso + (1.0 - corte_aviso) / 2
    return [
        {"clave": "sin_marca", "desde": 0.0, "hasta": umbral_dibujo,
         "etiqueta": "Sin región señalada"},
        {"clave": "bajo_aviso", "desde": umbral_dibujo, "hasta": corte_aviso,
         "etiqueta": "Bajo el umbral de clasificación"},
        {"clave": "sobre_aviso", "desde": corte_aviso, "hasta": medio,
         "etiqueta": "Sobre el umbral de clasificación"},
        {"clave": "alta", "desde": medio, "hasta": 1.01,
         "etiqueta": "Puntaje alto"},
    ]


def _acuerdo_en(grupo: list[dict]) -> dict:
    return _tasa(sum(1 for r in grupo if r["agreed"]), len(grupo))


def concordancia(registros: list[dict], umbral_dibujo: float, corte_aviso: float) -> dict:
    """Acuerdo sistema-médico: general, cuadrante 2x2, por triage y por franja de score.

    Cada registro trae `is_abnormal`, `urgency`, `max_detection_confidence` y
    `agreed` (None si nadie opinó). La celda `limpio_no` (no marcó y el médico
    discrepa) es el posible falso negativo.
    """
    revisados = [r for r in registros if r.get("agreed") is not None]
    n = len(revisados)
    marcados = [r for r in revisados if r.get("is_abnormal")]
    limpios = [r for r in revisados if not r.get("is_abnormal")]

    marco_ok = sum(1 for r in marcados if r["agreed"])
    marco_no = len(marcados) - marco_ok
    limpio_ok = sum(1 for r in limpios if r["agreed"])
    limpio_no = len(limpios) - limpio_ok

    por_triage = []
    for clave, etiqueta in ETIQUETAS_URGENCIA.items():
        grupo = [r for r in revisados if (r.get("urgency") or "LOW") == clave]
        if grupo:
            por_triage.append({"clave": clave, "etiqueta": etiqueta, **_acuerdo_en(grupo)})

    por_franja = []
    for f in franjas_de_seguridad(umbral_dibujo, corte_aviso):
        grupo = [r for r in revisados
                 if f["desde"] <= (r.get("max_detection_confidence") or 0.0) < f["hasta"]]
        if grupo:
            por_franja.append({
                "clave": f["clave"],
                "etiqueta": f["etiqueta"],
                "desde": f["desde"],
                "hasta": min(f["hasta"], 1.0),
                **_acuerdo_en(grupo),
            })

    return {
        "n": n,
        "total": len(registros),
        "cobertura": _tasa(n, len(registros)),  # qué parte de los análisis se revisó
        "general": _tasa(marco_ok + limpio_ok, n),
        "cuadrante": {
            "marco_ok": marco_ok,
            "marco_no": marco_no,
            "limpio_ok": limpio_ok,
            "limpio_no": limpio_no,
        },
        "cuando_marco": _tasa(marco_ok, len(marcados)),
        "cuando_no_marco": _tasa(limpio_ok, len(limpios)),
        "por_triage": por_triage,
        "por_franja": por_franja,
    }


def _percentil(valores: list[float], p: float) -> Optional[float]:
    """Percentil con interpolación lineal."""
    if not valores:
        return None
    orden = sorted(valores)
    if len(orden) == 1:
        return orden[0]
    pos =(len(orden) - 1) * p
    bajo = int(pos)
    alto = min(bajo + 1, len(orden) - 1)
    return orden[bajo] + (orden[alto] - orden[bajo]) * (pos - bajo)


def desempeno_sistema(registros: list[dict]) -> dict:
    """Tiempo de inferencia: mediana y p95 (un arranque en frío no corre la mediana)."""
    tiempos = [r["inference_time_ms"] for r in registros if r.get("inference_time_ms")]
    return {
        "n": len(tiempos),
        "mediana_ms": _percentil(tiempos, 0.50),
        "p95_ms": _percentil(tiempos, 0.95),
        "max_ms": max(tiempos) if tiempos else None,
    }


def practica(registros: list[dict], dias: int = 30) -> dict:
    """Uso propio. `por_dia_activo` divide por días CON actividad, no por días corridos."""
    total = len(registros)
    fechas = [a_local(r.get("timestamp")) for r in registros]
    fechas = [f.date() for f in fechas if f is not None]

    hoy = ahora_local().date()
    desde = hoy - timedelta(days=dias - 1)
    dias_activos = len(set(fechas))
    zonas = [r.get("n_zonas") or 0 for r in registros]

    return {
        "total": total,
        "ventana_dias": dias,
        "en_ventana": sum(1 for f in fechas if desde <= f <= hoy),
        "primero": min(fechas) if fechas else None,
        "ultimo": max(fechas) if fechas else None,
        "dias_activos": dias_activos,
        "por_dia_activo": (total / dias_activos) if dias_activos else 0.0,
        "marcados": _tasa(sum(1 for r in registros if r.get("is_abnormal")), total),
        "zonas_total": sum(zonas),
        "zonas_por_estudio": (sum(zonas) / total) if total else 0.0,
        "zonas_max": max(zonas) if zonas else 0,
    }

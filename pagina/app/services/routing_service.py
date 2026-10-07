"""
routing_service.py — Elección del modelo y nivel de urgencia.

Valida la región que eligió el médico y traduce el puntaje del detector a un
nivel de prioridad. Los umbrales son operativos: el puntaje NO es una probabilidad.
"""

from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from config.settings import (
    ABNORMAL_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    DEFAULT_REGION,
    REGION_AVAILABLE,
    URGENCY_HIGH_THRESHOLD,
    URGENCY_MED_THRESHOLD,
    YOLO_MODELS,
)


def _coma(x: float) -> str:
    """Puntaje con coma decimal y dos decimales (escala 0-1, nunca en %)."""
    return f"{x:.2f}".replace(".", ",")


def route_image(manual_region: str | None) -> tuple[str, str]:
    """Devuelve (region, 'manual'). 400 si la región no existe, 503 si su modelo no está."""
    region = (manual_region or DEFAULT_REGION).strip()

    if region not in YOLO_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Región '{region}' no reconocida. Opciones válidas: {', '.join(YOLO_MODELS)}",
        )

    if not REGION_AVAILABLE.get(region, False):
        disponibles = ", ".join(r for r, ok in REGION_AVAILABLE.items() if ok) or "ninguna"
        raise HTTPException(
            status_code=503,
            detail=f"El modelo para '{region}' no está disponible. Regiones activas: {disponibles}",
        )

    pesos = Path(YOLO_MODELS[region])
    if not pesos.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Falta el archivo de pesos de '{region}': {pesos.name}. "
                "Verifique que el archivo de pesos esté en disco y que la ruta configurada en .env sea correcta."
            ),
        )

    return region, "manual"


def calculate_urgency(max_detection_confidence: float) -> str:
    """'HIGH', 'MEDIUM' o 'LOW' según el score máximo del detector."""
    if max_detection_confidence >= URGENCY_HIGH_THRESHOLD:
        return "HIGH"
    if max_detection_confidence >= URGENCY_MED_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def urgency_detail(max_detection_confidence: float, is_abnormal: bool) -> dict:
    """Rótulo y clase CSS del nivel de urgencia. Separa el caso límite del negativo limpio.

    El puntaje se informa en escala 0-1 con coma decimal: no es una probabilidad.
    """
    conf = max_detection_confidence
    nivel = calculate_urgency(conf)

    if nivel == "HIGH":
        return {
            "nivel": "HIGH",
            "titulo": "PRIORITARIO — Puntaje alto del detector; confirmar con lectura médica",
            "clase": "urgency-high",
            "detalle": (
                f"Puntaje máximo del detector {_coma(conf)}, igual o superior al "
                f"umbral de prioridad ({_coma(URGENCY_HIGH_THRESHOLD)})."
            ),
        }

    if nivel == "MEDIUM":
        return {
            "nivel": "MEDIUM",
            "titulo": "REVISAR — Confirmar con lectura médica",
            "clase": "urgency-medium",
            "detalle": (
                f"Puntaje máximo del detector {_coma(conf)}, igual o superior al "
                f"umbral de clasificación ({_coma(ABNORMAL_THRESHOLD)})."
            ),
        }

    # Bajo el umbral, pero con una caja dibujada: caso límite.
    if not is_abnormal and conf >= CONFIDENCE_THRESHOLD:
        return {
            "nivel": "LOW_BORDERLINE",
            "titulo": "LÍMITE — Sin hallazgos sobre el umbral de clasificación; región señalada",
            "clase": "urgency-borderline",
            "detalle": (
                f"El puntaje máximo del detector ({_coma(conf)}) es inferior al "
                f"umbral de clasificación ({_coma(ABNORMAL_THRESHOLD)}) e igual o superior "
                f"al umbral de visualización ({_coma(CONFIDENCE_THRESHOLD)}). "
                "Se recomienda la correlación clínica."
            ),
        }

    if conf > 0:
        detalle = (
            f"Puntaje máximo del detector {_coma(conf)}, inferior al umbral de "
            f"clasificación ({_coma(ABNORMAL_THRESHOLD)}). Un resultado sin hallazgos "
            "no descarta fractura."
        )
    else:
        detalle = (
            "El detector no señaló ninguna región. Un resultado sin hallazgos "
            "no descarta fractura."
        )
    return {
        "nivel": "LOW",
        "titulo": "SIN HALLAZGOS — Considerar el contexto clínico",
        "clase": "urgency-low",
        "detalle": detalle,
    }


# Orden de gravedad de los niveles de `urgency_detail`. Uno desconocido cuenta como el más bajo.
_GRAVEDAD = {"HIGH": 3, "MEDIUM": 2, "LOW_BORDERLINE": 1, "LOW": 0}


def imagen_mas_urgente(resultados: list[dict]) -> Optional[int]:
    """El `index` de la imagen más urgente de un estudio (en la que abre el visor).

    Desempata por score más alto y después por índice más bajo, para que
    el mismo estudio abra siempre en la misma imagen.
    """
    if not resultados:
        return None

    def clave(r: dict) -> tuple:
        nivel = (r.get("urgency") or {}).get("nivel") or "LOW"
        return (
            _GRAVEDAD.get(nivel, 0),
            r.get("max_detection_confidence") or 0.0,
            -(r.get("index") or 0),
        )

    return max(resultados, key=clave)["index"]

"""
settings.py — Configuración de TraumaVision AI, leída del archivo .env.
"""

import os
import secrets
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Rutas del proyecto ---
BASE_DIR = Path(__file__).resolve().parent.parent  # Carpeta raíz del proyecto
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(BASE_DIR / "app" / "uploads")))

# --- Constantes de la aplicación ---
APP_NAME = "TraumaVision AI"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Sistema de Soporte a la Decisión Clínica para Detección de Fracturas"
MAX_UPLOAD_SIZE_MB = 20      # Tamaño máximo de imagen suelta
MAX_ZIP_SIZE_MB = 200        # Tamaño máximo del ZIP de un estudio
MAX_ZIP_UNCOMPRESSED_MB = 600  # Tope del contenido descomprimido (anti zip-bomb)
MAX_ZIP_ENTRIES = 400        # Tope de archivos dentro del ZIP

# Modo demo: crea usuarios de prueba y muestra sus claves. Apagado salvo que se pida.
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

# --- Seguridad ---
_DEFAULT_SECRET = "dev-key-cambiar-en-produccion"
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY or SECRET_KEY == _DEFAULT_SECRET:
    # Nunca una clave conocida: se genera una efímera (las sesiones se pierden al reiniciar).
    SECRET_KEY = secrets.token_hex(32)
    warnings.warn(
        "SECRET_KEY no definida en .env — se generó una clave efímera. "
        "Las sesiones se invalidan al reiniciar. Defina SECRET_KEY en .env "
        "con: python -c \"import secrets; print(secrets.token_hex(32))\"",
        RuntimeWarning,
        stacklevel=2,
    )

SESSION_COOKIE_NAME = "traumavision_session"
SESSION_MAX_AGE_SECONDS = int(os.getenv("SESSION_MAX_AGE_SECONDS", str(8 * 3600)))
# Cookie sólo por HTTPS: true en cualquier despliegue con TLS.
SESSION_HTTPS_ONLY = os.getenv("SESSION_HTTPS_ONLY", "false").lower() in ("1", "true", "yes")

# Orígenes permitidos para CORS. Nunca "*": con cookies el navegador lo rechaza.
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if o.strip()
]

# --- Base de datos ---
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'traumavision.db'}")

# ── Preprocesamiento ──────────────────────────────────────────────────────────
# CLAHE igual que en el entrenamiento (datos/grazpedwri/armar_dataset.py).
APPLY_CLAHE_AT_INFERENCE = os.getenv("APPLY_CLAHE_AT_INFERENCE", "true").lower() in ("1", "true", "yes")
CLAHE_CLIP_LIMIT = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
CLAHE_TILE_SIZE = int(os.getenv("CLAHE_TILE_SIZE", "8"))

# ── Umbrales de detección ─────────────────────────────────────────────────────
# 0,15 dibuja la caja; 0,22 clasifica el estudio como anormal.
# 0,22 se eligió sobre VALIDACIÓN con v1r: el más alto con sensibilidad >= 96 %.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.15"))
ABNORMAL_THRESHOLD = float(os.getenv("ABNORMAL_THRESHOLD", "0.22"))

# ── Urgencia clínica ──────────────────────────────────────────────────────────
# Cortes operativos sobre el score de YOLO, que NO es una probabilidad.
# El corte medio es el mismo que el de anormalidad.
URGENCY_HIGH_THRESHOLD = float(os.getenv("URGENCY_HIGH_THRESHOLD", "0.85"))
URGENCY_MED_THRESHOLD = float(os.getenv("URGENCY_MED_THRESHOLD", str(ABNORMAL_THRESHOLD)))

# ── Modelos disponibles ───────────────────────────────────────────────────────
# Un único modelo en producción: v1r, muñeca pediátrica (GRAZPEDWRI-DX).
YOLO_MODELS = {
    "muneca_pediatrica": os.getenv(
        "YOLO_MODEL_MUNECA",
        str(BASE_DIR.parent / "modelo/v1r/weights/best.pt"),
    ),
}

DEFAULT_REGION = "muneca_pediatrica"
YOLO_WEIGHTS_PATH = YOLO_MODELS[DEFAULT_REGION]

# ── Modelo vigente ────────────────────────────────────────────────────────────
# Nombre de la corrida: modelo/<corrida>/weights/best.pt -> "<corrida>".
MODELO_VIGENTE = Path(YOLO_WEIGHTS_PATH).parent.parent.name

# "yolov8m_v1r" es el nombre viejo de los mismos pesos de "v1r".
_ALIAS_DEL_MODELO = {"v1r": {"yolov8m_v1r"}}
VERSIONES_VIGENTES = frozenset({MODELO_VIGENTE, *_ALIAS_DEL_MODELO.get(MODELO_VIGENTE, ())})


def es_modelo_vigente(model_version) -> bool:
    """¿El análisis se hizo con el modelo vigente? Sólo esos cuentan en las métricas."""
    return model_version in VERSIONES_VIGENTES


# El informe de texto de un modelo anterior no se muestra (queda en la base y en el CSV).
NOTA_INFORME_ANTERIOR = (
    "El informe de texto original de este análisis lo generó un modelo anterior "
    "y no se muestra. Se conserva sin cambios en la base y en la exportación CSV."
)


def texto_del_informe(model_version, report_text) -> str:
    """El informe de texto que se puede mostrar de un análisis."""
    if es_modelo_vigente(model_version):
        return report_text or ""
    return NOTA_INFORME_ANTERIOR


# ── Métricas del modelo ───────────────────────────────────────────────────────
# Cada número sale de modelo/resultados/metricas*.json (no escribirlos a mano);
# tests/test_pagina.py comprueba que coincidan.
MODEL_METADATA: dict = {
    "muneca_pediatrica": {
        "label": "Muñeca — Pediátrica (0–19 años)",
        "description": "GRAZPEDWRI-DX · 20.327 imágenes · YOLOv8m · 60 épocas",
        "icon": "🦴",
        "available": True,
        # caja (metricas.json)
        "mAP50": 0.9436,
        "mAP50_95": 0.5528,
        "precision": 0.92,
        "recall": 0.8875,
        "test_n": 3038,
        "test_n_pacientes": 914,
        "test_n_instancias": 2722,
        "metrics_date": "2026-09-30",
        # nivel imagen, a τ = ABNORMAL_THRESHOLD (metricas.json)
        "sensibilidad": 0.9674,
        "especificidad": 0.9308,
        "vpp": 0.9655,
        # AUC-ROC y AP no dependen del umbral
        "auc_roc": 0.9868,
        "ap": 0.9932,
        "test_TP": 1960,
        "test_FN": 66,
        "test_FP": 70,
        "test_TN": 942,
        "test_n_positivos": 2026,
        "test_n_negativos": 1012,
        # test externo PediURF (metricas_externo.json), por caso y por imagen
        "test_externo": {
            "pesos": "modelo/v1r/weights/best.pt",
            "fecha": "2026-09-30",
            "umbral": 0.22,
            "n_casos": 1000,
            "n_imagenes": 1790,
            "por_caso": {
                "TP": 493,
                "FN": 7,
                "FP": 84,
                "TN": 416,
                "sensibilidad": 0.986,
                "especificidad": 0.832,
            },
            "por_imagen": {
                "TP": 968,
                "FN": 26,
                "FP": 93,
                "TN": 703,
                "sensibilidad": 0.9738,
                "especificidad": 0.8832,
            },
        },
        "supported_regions": ["Muñeca pediátrica"],
        "low_recall_regions": [],
    },
}

REGION_LABELS = {k: v["label"] for k, v in MODEL_METADATA.items()}
REGION_AVAILABLE = {k: v["available"] for k, v in MODEL_METADATA.items()}


# --- Disclaimer legal (se muestra en toda la app) ---
LEGAL_DISCLAIMER = (
    "AVISO LEGAL: este sistema es un prototipo académico de Sistema de Soporte a la "
    "Decisión Clínica (SSDC), destinado a demostración y no apto para estudios de "
    "pacientes identificables. NO realiza diagnósticos médicos ni reemplaza el juicio "
    "clínico del profesional. Los resultados generados son sugerencias computacionales "
    "que deben ser interpretadas, validadas y aprobadas por un médico matriculado."
)


# --- Salvedad de dominio (PDF), derivada de MODEL_METADATA ---
# Valor "no se pasó el argumento": distinto de None, que significa
# "el análisis no guardó su región / su modelo".
REGION_NO_INDICADA = object()


def domain_disclaimer(region=REGION_NO_INDICADA, model_version=REGION_NO_INDICADA) -> str:
    """Alcance y limitaciones del modelo; sin región conocida no cita ninguna métrica."""
    clave = DEFAULT_REGION if region is REGION_NO_INDICADA else region
    meta = MODEL_METADATA.get(clave) if clave else None

    if not meta:  # modelo retirado o sin región registrada
        return (
            "ALCANCE Y LIMITACIONES: este resultado se encuentra FUERA DEL DOMINIO VALIDADO. "
            "Se generó con un modelo retirado del sistema o sin región registrada, "
            "cuyas métricas no son reproducibles. El resultado no debe utilizarse para ordenar "
            "la revisión ni para fundamentar ninguna conducta clínica."
        )

    sens = meta["sensibilidad"]  # por imagen, a τ = ABNORMAL_THRESHOLD
    faltan = round((1 - sens) * 100)
    pct = lambda x: ("%.1f" % (x * 100)).replace(".", ",") + " %"  # noqa: E731

    # Modelo anterior: se aclara que las métricas son del vigente, no de este análisis.
    if model_version is not REGION_NO_INDICADA and not es_modelo_vigente(model_version):
        return (
            "ALCANCE Y LIMITACIONES: análisis realizado con un modelo anterior "
            f"({model_version or 'sin registrar'}), que ya no está en uso. "
            "No se informa prioridad clínica y el resultado no debe utilizarse para "
            "ordenar la revisión. Las métricas publicadas del sistema corresponden "
            f"al modelo vigente ({MODELO_VIGENTE}) y no a este resultado: sensibilidad "
            f"por imagen {pct(sens)}, medida el {meta['metrics_date']}. "
            "UN INFORME SIN HALLAZGOS NO DESCARTA FRACTURA. "
            "El puntaje del detector que muestra el sistema no es una probabilidad "
            "de fractura y no debe interpretarse como tal."
        )

    validacion = (
        "Validación retrospectiva: interna, separada por paciente, y externa "
        "(PediURF, recortes de muñeca). Sin estudio prospectivo."
        if meta.get("test_externo")
        else "Validación interna, separada por paciente: sin estudio prospectivo."
    )
    return (
        "ALCANCE Y LIMITACIONES: el modelo está validado ÚNICAMENTE sobre "
        f"{meta['label']}, entrenado y evaluado sobre {meta['description']}. "
        "Fuera de ese dominio este informe carece de validez. "
        f"Sensibilidad por imagen {pct(sens)}, medida el {meta['metrics_date']} sobre "
        f"un conjunto de prueba de {meta['test_n']} imágenes separado por paciente: "
        f"alrededor de {faltan} de cada 100 radiografías con fractura no son señaladas, de modo "
        "que UN INFORME SIN HALLAZGOS NO DESCARTA FRACTURA. "
        "El puntaje del detector que muestra el sistema no es una probabilidad "
        "de fractura y no debe interpretarse como tal. "
        + validacion
    )

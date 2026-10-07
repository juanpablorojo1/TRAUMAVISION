"""
predict.py — Detección de fracturas con YOLOv8 (muñeca pediátrica).

Aplica a la placa el mismo CLAHE que se usó en el entrenamiento, corre el
modelo, dibuja las cajas y arma el informe de texto.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from config.settings import (
    ABNORMAL_THRESHOLD,
    APPLY_CLAHE_AT_INFERENCE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    CONFIDENCE_THRESHOLD,
    DEFAULT_REGION,
    MODEL_METADATA,
    YOLO_MODELS,
)

ROJO = (0, 0, 220)     # BGR: caja sobre el umbral de anormalidad
AMBAR = (0, 170, 240)  # BGR: caja de baja confianza (sólo referencia)


@dataclass
class DetectionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass
class PredictionResult:
    """Resultado de una inferencia. El score es el de YOLO: NO es una probabilidad."""

    is_abnormal: bool
    max_detection_confidence: float
    boxes: List[DetectionBox]  # todas las cajas desde CONFIDENCE_THRESHOLD
    annotated_image: np.ndarray
    inference_time_ms: float
    image_size: tuple
    model_version: str = ""
    clahe_applied: bool = False

    @property
    def significant_boxes(self) -> List[DetectionBox]:
        return [b for b in self.boxes if b.confidence >= ABNORMAL_THRESHOLD]

    @property
    def low_confidence_boxes(self) -> List[DetectionBox]:
        return [b for b in self.boxes if b.confidence < ABNORMAL_THRESHOLD]


def _coma(x: float, decimales: int = 2) -> str:
    """Número con coma decimal (el score va en escala 0-1, nunca en %)."""
    return f"{x:.{decimales}f}".replace(".", ",")


def apply_clahe_rgb(pil_image: Image.Image) -> Image.Image:
    """CLAHE igual que en el entrenamiento: gris -> CLAHE -> 3 canales."""
    gris = np.array(pil_image.convert("L"))
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE),
    )
    realzada = clahe.apply(gris)
    return Image.fromarray(cv2.cvtColor(realzada, cv2.COLOR_GRAY2RGB))


def _a_pil(image) -> Image.Image:
    """Acepta ruta, array de OpenCV (gris o BGR) o imagen PIL."""
    if isinstance(image, (str, Path)):
        # Misma lectura que la carga web (16 bits se reescala, no se recorta).
        from src.preprocessing.transforms import load_image

        placa = load_image(Path(image).read_bytes())
        if placa is None:
            raise ValueError(f"No se pudo leer la imagen: {image}")
        return placa.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image.convert("RGB")


def _rectangulo_punteado(img, pt1, pt2, color, grosor, paso=8):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, paso * 2):
        cv2.line(img, (x, y1), (min(x + paso, x2), y1), color, grosor)
        cv2.line(img, (x, y2), (min(x + paso, x2), y2), color, grosor)
    for y in range(y1, y2, paso * 2):
        cv2.line(img, (x1, y), (x1, min(y + paso, y2)), color, grosor)
        cv2.line(img, (x2, y), (x2, min(y + paso, y2)), color, grosor)


def _ubicacion_en_imagen(b: DetectionBox, ancho: int, alto: int) -> str:
    """Sector de la imagen donde cae la caja (no anatómico: el modelo no sabe de huesos)."""
    cx = (b.x1 + b.x2) / 2 / max(ancho, 1)
    cy = (b.y1 + b.y2) / 2 / max(alto, 1)
    vertical = "superior" if cy < 1 / 3 else ("central" if cy < 2 / 3 else "inferior")
    horizontal = "izquierdo" if cx < 1 / 3 else ("central" if cx < 2 / 3 else "derecho")
    if vertical == "central" and horizontal == "central":
        return "sector central de la imagen"
    return f"sector {vertical} {horizontal} de la imagen"


def _tamano_relativo(b: DetectionBox, ancho: int, alto: int) -> str:
    area = ((b.x2 - b.x1) * (b.y2 - b.y1)) / max(ancho * alto, 1)
    if area < 0.01:
        return "focal"
    if area < 0.05:
        return "de pequeño tamaño"
    return "extenso"


class FractureDetector:
    """Detector YOLO; `FractureDetector.get(region)` lo carga una vez y lo reutiliza."""

    _cache: dict = {}

    @classmethod
    def get(cls, region: Optional[str] = None) -> "FractureDetector":
        region = region or DEFAULT_REGION
        if region not in YOLO_MODELS:
            raise ValueError(
                f"Región '{region}' no configurada. "
                f"Disponibles: {', '.join(YOLO_MODELS)}"
            )
        if region not in cls._cache:
            cls._cache[region] = cls(YOLO_MODELS[region], region)
        return cls._cache[region]

    def __init__(self, weights_path: str, region: str):
        from ultralytics import YOLO

        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {path}\n"
                "Verifique YOLO_MODEL_MUNECA en .env y que el archivo de pesos esté en disco."
            )

        self.model = YOLO(str(path))
        self.confidence_threshold = CONFIDENCE_THRESHOLD  # desde acá se dibuja la caja
        self.abnormal_threshold = ABNORMAL_THRESHOLD      # desde acá el estudio es anormal
        self.region = region
        self.model_version = path.parent.parent.name  # modelo/<corrida>/weights/best.pt
        # Las rutas llaman a predict desde varios hilos: una inferencia a la vez.
        self._candado = threading.Lock()
        print(
            f"[TraumaVision] Modelo cargado: {path.name} "
            f"({self.region} · {self.model_version} · "
            f"CLAHE={'sí' if APPLY_CLAHE_AT_INFERENCE else 'no'})"
        )

    def predict(self, image) -> PredictionResult:
        inicio = time.time()
        placa = _a_pil(image)
        ancho, alto = placa.size

        # El modelo ve la placa con CLAHE; al médico se le muestra la original.
        entrada = apply_clahe_rgb(placa) if APPLY_CLAHE_AT_INFERENCE else placa
        with self._candado:
            resultados = self.model.predict(
                source=entrada,
                conf=self.confidence_threshold,
                imgsz=640,
                verbose=False,
            )

        boxes = []
        for box in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append(DetectionBox(x1, y1, x2, y2, float(box.conf)))
        boxes.sort(key=lambda b: b.confidence, reverse=True)
        max_conf = max((b.confidence for b in boxes), default=0.0)

        img_bgr = cv2.cvtColor(np.array(placa), cv2.COLOR_RGB2BGR)
        return PredictionResult(
            is_abnormal=max_conf >= self.abnormal_threshold,
            max_detection_confidence=max_conf,
            boxes=boxes,
            annotated_image=self._draw_boxes(img_bgr, boxes),
            inference_time_ms=(time.time() - inicio) * 1000,
            image_size=(ancho, alto),
            model_version=self.model_version,
            clahe_applied=APPLY_CLAHE_AT_INFERENCE,
        )

    def _draw_boxes(self, img: np.ndarray, boxes: List[DetectionBox]) -> np.ndarray:
        """Rojo sólido = hallazgo; ámbar punteado = bajo el umbral de clasificación."""
        out = img.copy()
        for b in boxes:
            if b.confidence >= self.abnormal_threshold:
                color, etiqueta = ROJO, f"Hallazgo {_coma(b.confidence)}"
                cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), color, 2)
            else:
                color, etiqueta = AMBAR, f"Bajo umbral {_coma(b.confidence)}"
                _rectangulo_punteado(out, (b.x1, b.y1), (b.x2, b.y2), color, 1)

            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_texto = max(b.y1, th + 6)
            cv2.rectangle(out, (b.x1, y_texto - th - 6), (b.x1 + tw + 6, y_texto), color, -1)
            cv2.putText(
                out, etiqueta, (b.x1 + 3, y_texto - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return out

    def generate_report_text(self, result: PredictionResult) -> str:
        """Informe de texto: TÉCNICA / HALLAZGOS / IMPRESIÓN / LIMITACIONES."""
        lineas = (
            self._tecnica(result)
            + self._hallazgos(result)
            + self._impresion(result)
            + self._limitaciones()
        )
        return "\n".join(lineas)

    def _tecnica(self, result: PredictionResult) -> list:
        return [
            "INFORME DE ANÁLISIS AUTOMATIZADO — TraumaVision AI",
            "=" * 49,  # ASCII: la Helvetica del PDF no tiene caracteres de dibujo de caja
            "",
            "TÉCNICA",
            f"Radiografía analizada con detector de fracturas YOLOv8m "
            f"({result.model_version}), preprocesamiento "
            f"{'CLAHE' if result.clahe_applied else 'sin realce'}, "
            f"en {result.inference_time_ms:.0f} ms.",
            f"Umbral de visualización: {_coma(self.confidence_threshold)} · "
            f"umbral de clasificación: {_coma(self.abnormal_threshold)} "
            "(puntaje del detector, escala 0-1).",
            "",
        ]

    def _hallazgos(self, result: PredictionResult) -> list:
        ancho, alto = result.image_size
        lineas = ["HALLAZGOS"]
        if not result.boxes:
            lineas.append(
                "No se identifican imágenes compatibles con trazo de fractura "
                "con puntaje igual o superior al umbral de visualización."
            )
            return lineas

        for i, b in enumerate(result.significant_boxes, 1):
            lineas.append(
                f"{i}. Imagen compatible con trazo de fractura, "
                f"{_tamano_relativo(b, ancho, alto)}, en el "
                f"{_ubicacion_en_imagen(b, ancho, alto)} "
                f"(puntaje del detector: {_coma(b.confidence)})."
            )
        bajas = result.low_confidence_boxes
        if bajas:
            lineas.append("")
            lineas.append(
                f"Se señalan además {len(bajas)} región(es) con puntaje inferior al "
                "umbral de clasificación (caja delimitadora punteada), como referencia "
                "para la correlación clínica; no constituyen hallazgos:"
            )
            for b in bajas:
                lineas.append(
                    f"   • {_ubicacion_en_imagen(b, ancho, alto)} "
                    f"({_coma(b.confidence)})"
                )
        return lineas

    def _impresion(self, result: PredictionResult) -> list:
        if result.is_abnormal:
            veredicto = (
                f"Estudio CON HALLAZGOS: {len(result.significant_boxes)} imagen(es) "
                "compatible(s) con fractura con puntaje igual o superior al umbral "
                "de clasificación."
            )
        else:
            veredicto = "Estudio SIN HALLAZGOS que alcancen el umbral de clasificación."
        return [
            "",
            "IMPRESIÓN",
            veredicto,
            "Nota: el puntaje informado es la salida interna del detector y no "
            "constituye una probabilidad de fractura.",
        ]

    def _limitaciones(self) -> list:
        # La sensibilidad sale de la metadata de ESTA región; si no la tiene, no se cita.
        sensibilidad = (MODEL_METADATA.get(self.region) or {}).get("sensibilidad")
        if sensibilidad is None:
            medida = "."
        else:
            medida = f" (sensibilidad medida: {_coma(sensibilidad * 100, 1)} % por imagen)."
        return [
            "",
            "LIMITACIONES",
            "El sistema detecta y localiza imágenes compatibles con fractura en "
            "muñeca pediátrica; no tipifica el trazo ni identifica el hueso "
            f"comprometido. Un resultado sin hallazgos no descarta patología{medida}",
            "",
            "AVISO LEGAL: informe generado automáticamente por un prototipo académico "
            "de soporte a la decisión clínica. No constituye un diagnóstico; debe ser "
            "interpretado, validado y firmado por un médico matriculado.",
        ]

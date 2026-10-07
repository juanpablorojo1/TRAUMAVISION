"""
transforms.py — Lectura de imágenes (PNG, JPEG, BMP, TIFF y DICOM) y de estudios en ZIP.

El CLAHE de la inferencia está en `src/detection/predict.py`.
"""

import io
import re
import zipfile
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pydicom
from PIL import Image
try:  # pydicom >= 3
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:  # pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

from config.settings import MAX_ZIP_ENTRIES, MAX_ZIP_UNCOMPRESSED_MB


def sanitize_filename(nombre: str) -> str:
    """Nombre base del archivo, sin ruta y con los números largos (IDs, fechas) tapados."""
    base = nombre.replace("\\", "/").rsplit("/", 1)[-1]
    base = re.sub(r"\.(dcm|dicom)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\d{4,}", "…", base)
    base = re.sub(r"[^\w\s.\-…]", "", base, flags=re.UNICODE).strip()
    return base[:60] or "imagen"


# ── PNG, JPEG, BMP, TIFF ─────────────────────────────────────────────────────


def load_image(contents: bytes) -> Optional[Image.Image]:
    """Imagen suelta -> PIL en gris de 8 bits, leída igual que en el entrenamiento.

    `datos/grazpedwri/armar_dataset.py` usa `cv2.imread(..., IMREAD_GRAYSCALE)`,
    que pasa 16 -> 8 bits reescalando. Pillow, en cambio, recorta todo lo que
    pasa de 255 y una placa de 16 bits queda casi toda blanca.
    Devuelve None si el archivo no es una imagen que OpenCV pueda leer.
    """
    gris = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gris is None:
        return None
    return Image.fromarray(gris)


# ── DICOM ────────────────────────────────────────────────────────────────────


def load_dicom(dicom_bytes: bytes) -> Image.Image:
    """DICOM -> imagen PIL RGB de 8 bits. No lee los datos del paciente."""
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
    arr = ds.pixel_array

    # 1. Modality LUT: valores crudos -> escala física del equipo.
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        arr = arr.astype(np.float32) * slope + intercept

    # 2. VOI LUT: la ventana que grabó el equipo (si falla, queda el estirado de abajo).
    if hasattr(ds, "WindowCenter") or hasattr(ds, "VOILUTSequence"):
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass

    arr = np.asarray(arr, dtype=np.float32)

    # Multiframe (n, alto, ancho): se toma el primer cuadro.
    if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] not in (1, 3):
        arr = arr[0]

    # 3. Estirar a 0–255.
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)

    # 4. MONOCHROME1 viene invertida (blanco = mínimo).
    if str(getattr(ds, "PhotometricInterpretation", "")).strip().upper() == "MONOCHROME1":
        arr = 255.0 - arr

    arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(arr)


# ── ZIP ──────────────────────────────────────────────────────────────────────


@dataclass
class DicomEntry:
    filename: str  # ya sanitizado
    image: Image.Image
    instance_number: Optional[int] = None


def _es_entrada_segura(info: zipfile.ZipInfo) -> bool:
    """Descarta carpetas, basura de macOS, ocultos y rutas peligrosas (zip slip)."""
    nombre = info.filename
    return (
        not info.is_dir()
        and not nombre.startswith("__MACOSX")
        and not nombre.rsplit("/", 1)[-1].startswith(".")
        and not nombre.startswith(("/", "\\"))
        and ".." not in nombre.replace("\\", "/").split("/")
    )


def _parece_dicom(nombre: str) -> bool:
    """Extensión .dcm, .dicom o ninguna (muchos DICOM clínicos no tienen)."""
    base = nombre.rsplit("/", 1)[-1]
    ext = nombre.rsplit(".", 1)[-1].lower() if "." in base else ""
    return ext in ("dcm", "dicom", "")


def _numero_de_instancia(datos: bytes) -> Optional[int]:
    cabecera = pydicom.dcmread(io.BytesIO(datos), stop_before_pixels=True)
    try:
        return int(cabecera.InstanceNumber)
    except (AttributeError, ValueError, TypeError):
        return None


def extract_dicoms_from_zip(zip_bytes: bytes) -> list[DicomEntry]:
    """DICOM de un ZIP, en memoria y ordenados por InstanceNumber (o nombre).

    Con topes de cantidad y de tamaño descomprimido contra ZIP maliciosos.
    Levanta ValueError si no hay ningún DICOM válido.
    """
    entries: list[DicomEntry] = []
    errores: list[str] = []
    tope_bytes = MAX_ZIP_UNCOMPRESSED_MB * 1024 * 1024

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        infos = [i for i in zf.infolist() if _es_entrada_segura(i)]

        if len(infos) > MAX_ZIP_ENTRIES:
            raise ValueError(
                f"El ZIP contiene {len(infos)} archivos y el máximo admitido es "
                f"{MAX_ZIP_ENTRIES}."
            )

        # Tamaño declarado en la cabecera, antes de descomprimir nada.
        total_declarado = sum(i.file_size for i in infos)
        if total_declarado > tope_bytes:
            raise ValueError(
                f"El contenido descomprimido ({total_declarado / 1024 / 1024:.0f} MB) "
                f"supera el máximo de {MAX_ZIP_UNCOMPRESSED_MB} MB."
            )

        leidos = 0
        for info in infos:
            nombre = info.filename
            if not _parece_dicom(nombre):
                continue

            datos = zf.read(nombre)
            leidos += len(datos)
            if leidos > tope_bytes:
                raise ValueError(
                    f"El ZIP excede el máximo descomprimido de {MAX_ZIP_UNCOMPRESSED_MB} MB."
                )

            try:
                entries.append(
                    DicomEntry(
                        filename=sanitize_filename(nombre),
                        instance_number=_numero_de_instancia(datos),
                        image=load_dicom(datos),
                    )
                )
            except Exception as exc:
                errores.append(f"{sanitize_filename(nombre)}: {exc}")

    if not entries:
        detalle = "; ".join(errores[:5]) if errores else "El ZIP no contiene archivos DICOM válidos."
        raise ValueError(f"No se encontraron imágenes DICOM en el archivo: {detalle}")

    entries.sort(key=lambda e: (e.instance_number is None, e.instance_number or 0, e.filename))
    return entries

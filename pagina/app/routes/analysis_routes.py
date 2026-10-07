"""
analysis_routes.py — Análisis de radiografías.

  GET  /analysis/upload              formulario de carga
  POST /analysis/upload              analiza una imagen
  POST /analysis/upload-study        analiza un ZIP de DICOM
  GET  /analysis/history             historial del usuario
  GET  /analysis/{id}/results        resultado guardado
  GET  /analysis/{id}/pdf            informe en PDF
  GET  /analysis/study/{id}          resultado de un estudio
  GET  /analysis/imagen/{archivo}    sirve una imagen, previa verificación

Todas exigen sesión y sólo devuelven datos del usuario logueado.
"""

import json
import re
import uuid
from io import BytesIO

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from PIL import Image, ImageDraw
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.database import crud
from app.database.db import get_db
from app.database.models import Analysis, User
from app.dependencies.auth import require_user
from app.dependencies.csrf import get_csrf_token, verify_csrf
from app.plantillas import crear_templates
from app.services.routing_service import (
    calculate_urgency,
    imagen_mas_urgente,
    route_image,
    urgency_detail,
)
from config.settings import (
    ABNORMAL_THRESHOLD,
    APP_NAME,
    CONFIDENCE_THRESHOLD,
    DEFAULT_REGION,
    LEGAL_DISCLAIMER,
    MAX_UPLOAD_SIZE_MB,
    MAX_ZIP_SIZE_MB,
    MODEL_METADATA,
    MODELO_VIGENTE,
    REGION_AVAILABLE,
    REGION_LABELS,
    UPLOADS_DIR,
    es_modelo_vigente,
    texto_del_informe,
)

router = APIRouter()

templates = crear_templates()

# Tope de filas del historial; la pantalla avisa si el total real es mayor.
HISTORIAL_MAX_FILAS = 500

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "application/dicom"}
_TIPOS_ZIP = ("application/zip", "application/x-zip-compressed")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.(png|jpg|jpeg)$")

# Color de las cajas en el PDF de un análisis de un modelo anterior (violeta KIBBO 200).
_CAJA_NEUTRA = "#CFC4F1"


# ─── Ayudas ──────────────────────────────────────────────────────────────────

def _base_context(request: Request, user: User) -> dict:
    """Contexto común de todas las plantillas de análisis."""
    meta = MODEL_METADATA.get(DEFAULT_REGION)
    return {
        "app_name": APP_NAME,
        "disclaimer": LEGAL_DISCLAIMER,
        "current_user": user,
        "model_metadata": MODEL_METADATA,
        "region_labels": REGION_LABELS,
        "region_available": REGION_AVAILABLE,
        "max_size_mb": MAX_UPLOAD_SIZE_MB,
        "max_zip_mb": MAX_ZIP_SIZE_MB,
        "abnormal_threshold": ABNORMAL_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_sensitivity": (meta or {}).get("sensibilidad", 0.0),
        "dominio_label": (meta or {}).get("label", DEFAULT_REGION),
        "meta": meta,
        "modelo_vigente": MODELO_VIGENTE,
        "csrf_token": get_csrf_token(request),
    }


def _upload_error(request: Request, user: User, mensaje: str, status_code: int = 200):
    """Vuelve a mostrar el formulario de carga con un mensaje de error."""
    return templates.TemplateResponse(
        request, "upload.html",
        {**_base_context(request, user), "error": mensaje},
        status_code=status_code,
    )


def _texto_del_informe(analysis) -> str:
    return texto_del_informe(analysis.model_version, analysis.report_text)


def _modelo_anterior(region, model_version) -> bool:
    """Análisis de muñeca hecho con un modelo que ya no está en uso."""
    return region == DEFAULT_REGION and not es_modelo_vigente(model_version)


def _etiqueta_region(region) -> str:
    return MODEL_METADATA.get(region, {}).get("label", region or "—")


def _boxes_json(boxes) -> str:
    """Las cajas guardadas, en JSON para el visor, de mayor a menor confianza."""
    ordenadas = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    return json.dumps(
        [
            {"i": i, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "c": round(b.confidence, 4)}
            for i, b in enumerate(ordenadas)
        ],
        separators=(",", ":"),
    )


def _cajas_para_guardar(result) -> list[dict]:
    return [
        {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "confidence": b.confidence}
        for b in result.boxes
    ]


def _guardar_anotada(result, nombre: str) -> None:
    """Guarda la imagen con las cajas dibujadas (el modelo la devuelve en BGR)."""
    anotada = Image.fromarray(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB))
    anotada.save(str(UPLOADS_DIR / nombre))


def _cargar_detector(region_key: str):
    from src.detection.predict import FractureDetector

    return FractureDetector.get(region_key)


async def _archivo_del_formulario(request: Request) -> tuple[StarletteUploadFile | None, str]:
    """Lee `file` y `region` del formulario.

    Se lee acá y no en la firma de la ruta para que la sesión y el CSRF se
    verifiquen antes de recibir el archivo.
    """
    form = await request.form()
    archivo = form.get("file")
    if not isinstance(archivo, StarletteUploadFile):
        archivo = None
    region = form.get("region")
    region = region if isinstance(region, str) and region else DEFAULT_REGION
    return archivo, region


# ─── Formulario ──────────────────────────────────────────────────────────────

@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, user: User = Depends(require_user)):
    return templates.TemplateResponse(request, "upload.html", _base_context(request, user))


# ─── Análisis de una imagen ──────────────────────────────────────────────────

@router.post("/upload")
async def analyze_image(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
    _: None = Depends(verify_csrf),
):
    """Analiza una radiografía y redirige a su resultado."""
    file, region = await _archivo_del_formulario(request)
    if file is None:
        return _upload_error(request, user, "No se recibió ningún archivo. Seleccione una radiografía para analizar.")

    is_dicom = (file.filename or "").lower().endswith(".dcm") or file.content_type == "application/dicom"
    if file.content_type not in _ALLOWED_IMAGE_TYPES and not is_dicom:
        return _upload_error(request, user, "Tipo de archivo no permitido. Use JPEG, PNG, BMP, TIFF o DICOM (.dcm).")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return _upload_error(request, user, f"La imagen excede el tamaño máximo de {MAX_UPLOAD_SIZE_MB} MB.")
    if not contents:
        return _upload_error(request, user, "El archivo está vacío.")

    try:
        region_key, routing_method = route_image(region)
    except HTTPException as exc:
        return _upload_error(request, user, exc.detail, status_code=exc.status_code)

    # Un error al decodificar es del archivo, no del modelo.
    try:
        if is_dicom:
            from src.preprocessing.transforms import load_dicom
            image = load_dicom(contents)
        else:
            # Como en el entrenamiento: OpenCV en gris (16 bits se reescala, no se recorta).
            from src.preprocessing.transforms import load_image
            image = load_image(contents)
    except Exception as exc:
        return _upload_error(request, user, f"No se pudo procesar el archivo: {exc}")
    if image is None:
        return _upload_error(request, user, "No se pudo leer la imagen: el archivo no es una imagen válida o está dañado.")

    try:
        detector = _cargar_detector(region_key)
    except (FileNotFoundError, ValueError) as exc:
        return _upload_error(request, user, f"El modelo de IA no está disponible: {exc}", status_code=503)

    uid = uuid.uuid4().hex[:12]
    original_filename = f"{uid}_original.png"
    annotated_filename = f"{uid}_annotated.png"

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(str(UPLOADS_DIR / original_filename))

    # La inferencia corre en otro hilo para no congelar el servidor mientras tanto.
    result = await run_in_threadpool(detector.predict, image)
    _guardar_anotada(result, annotated_filename)

    analysis = crud.create_analysis(
        db=db,
        user_id=user.id,
        original_image_path=original_filename,
        annotated_image_path=annotated_filename,
        report_text=detector.generate_report_text(result),
        max_detection_confidence=result.max_detection_confidence,
        is_abnormal=result.is_abnormal,
        inference_time_ms=result.inference_time_ms,
        anatomical_region=region_key,
        model_version=result.model_version,
        routing_method=routing_method,
        urgency=calculate_urgency(result.max_detection_confidence),
    )
    if result.boxes:
        crud.create_detection_boxes(db, analysis.id, _cajas_para_guardar(result))

    return RedirectResponse(url=f"/analysis/{analysis.id}/results", status_code=303)


# ─── Historial, imágenes y resultado ─────────────────────────────────────────

@router.get("/history", response_class=HTMLResponse)
async def history_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    analyses = crud.get_analyses_by_user(db, user_id=user.id, limit=HISTORIAL_MAX_FILAS)
    return templates.TemplateResponse(
        request, "history.html",
        {
            **_base_context(request, user),
            "analyses": analyses,
            "total_registros": crud.count_analyses_by_user(db, user_id=user.id),
            "default_region": DEFAULT_REGION,
            "vigentes": {a.id for a in analyses if es_modelo_vigente(a.model_version)},
        },
    )


@router.get("/imagen/{archivo}")
async def serve_image(
    archivo: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Sirve una radiografía sólo si pertenece a un análisis del usuario."""
    if not _SAFE_FILENAME_RE.match(archivo):
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido.")

    es_del_usuario = (
        db.query(Analysis)
        .filter(
            Analysis.user_id == user.id,
            (Analysis.original_image_path == archivo)
            | (Analysis.annotated_image_path == archivo),
        )
        .first()
    )
    if es_del_usuario is None:
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")

    ruta = (UPLOADS_DIR / archivo).resolve()
    if not str(ruta).startswith(str(UPLOADS_DIR.resolve())) or not ruta.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")

    return FileResponse(
        str(ruta),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.get("/{analysis_id}/results", response_class=HTMLResponse)
async def view_analysis(
    analysis_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    analysis = crud.get_analysis_for_user(db, analysis_id, user.id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Análisis no encontrado.")

    boxes = crud.get_boxes_by_analysis(db, analysis.id)

    return templates.TemplateResponse(
        request, "results.html",
        {
            **_base_context(request, user),
            "analysis": analysis,
            "analysis_id": analysis.id,
            "annotated_image": analysis.annotated_image_path,
            "original_image": analysis.original_image_path,
            "report_text": _texto_del_informe(analysis),
            "informe_oculto": not es_modelo_vigente(analysis.model_version),
            "is_abnormal": analysis.is_abnormal,
            "inference_time": f"{analysis.inference_time_ms or 0:.0f}",
            "boxes": boxes,
            "significant_boxes": [b for b in boxes if b.confidence >= ABNORMAL_THRESHOLD],
            "low_confidence_boxes": [b for b in boxes if b.confidence < ABNORMAL_THRESHOLD],
            "boxes_json": _boxes_json(boxes),
            "urgency": urgency_detail(
                analysis.max_detection_confidence or 0.0,
                bool(analysis.is_abnormal),
            ),
            "anatomical_region": _etiqueta_region(analysis.anatomical_region),
            "region_key": analysis.anatomical_region,
            "default_region": DEFAULT_REGION,
            "model_version": analysis.model_version,
            "modelo_anterior": _modelo_anterior(analysis.anatomical_region, analysis.model_version),
            "feedback": crud.get_feedback_by_analysis(db, analysis.id),
        },
    )


# ─── PDF ─────────────────────────────────────────────────────────────────────

def _anotada_neutra(original: Image.Image, cajas) -> Image.Image:
    """La placa original con las cajas guardadas, todas en el color neutro."""
    lienzo = original.convert("RGB")
    trazo = max(2, round(max(lienzo.size) / 500))
    dibujo = ImageDraw.Draw(lienzo)
    for b in cajas:
        dibujo.rectangle([b.x1, b.y1, b.x2, b.y2], outline=_CAJA_NEUTRA, width=trazo)
    return lienzo


def _imagen_anotada(analysis) -> Image.Image:
    """La placa con las cajas. Si el modelo ya no es el vigente, las cajas van en color neutro."""
    if not es_modelo_vigente(analysis.model_version):
        original = Image.open(str(UPLOADS_DIR / analysis.original_image_path))
        return _anotada_neutra(original, analysis.detection_boxes)
    return Image.open(str(UPLOADS_DIR / analysis.annotated_image_path))


def _pdf_del_analisis(analysis, user: User) -> bytes:
    """El informe PDF del análisis."""
    from src.reports.generator import generate_pdf_report

    return generate_pdf_report(
        annotated_image=_imagen_anotada(analysis),
        report_text=_texto_del_informe(analysis),
        analysis_id=str(analysis.id),
        doctor_name=user.name,
        region=analysis.anatomical_region,
        created_at=analysis.created_at,
        model_version=analysis.model_version,
    )


@router.get("/{analysis_id}/pdf")
async def download_pdf(
    analysis_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    analysis = crud.get_analysis_for_user(db, analysis_id, user.id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Análisis no encontrado.")

    return StreamingResponse(
        BytesIO(_pdf_del_analisis(analysis, user)),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=traumavision_informe_{analysis.id}.pdf"},
    )


# ─── Estudios de varias imágenes ─────────────────────────────────────────────

@router.post("/upload-study")
async def analyze_study(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
    _: None = Depends(verify_csrf),
):
    """Analiza un ZIP con varios DICOM y redirige al resultado del estudio."""
    file, region = await _archivo_del_formulario(request)
    if file is None:
        return _upload_error(request, user, "No se recibió ningún archivo. Cargue un archivo ZIP con los DICOM del estudio.")

    is_zip = (file.filename or "").lower().endswith(".zip") or file.content_type in _TIPOS_ZIP
    if not is_zip:
        return _upload_error(request, user, "Para estudios de varias imágenes, cargue un archivo ZIP con los DICOM.")

    contents = await file.read()
    if len(contents) > MAX_ZIP_SIZE_MB * 1024 * 1024:
        return _upload_error(request, user, f"El archivo ZIP excede el límite de {MAX_ZIP_SIZE_MB} MB.")

    try:
        region_key, routing_method = route_image(region)
    except HTTPException as exc:
        return _upload_error(request, user, exc.detail, status_code=exc.status_code)

    try:
        from src.preprocessing.transforms import extract_dicoms_from_zip
        dicom_entries = extract_dicoms_from_zip(contents)
    except ValueError as exc:
        return _upload_error(request, user, str(exc))
    except Exception as exc:
        return _upload_error(request, user, f"No se pudo leer el ZIP: {exc}")

    try:
        detector = _cargar_detector(region_key)
    except (FileNotFoundError, ValueError) as exc:
        return _upload_error(request, user, f"El modelo de IA no está disponible: {exc}", status_code=503)

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    study_token = uuid.uuid4().hex[:8]

    study = crud.create_study(
        db=db,
        user_id=user.id,
        original_filename=file.filename or "estudio.zip",
        total_images=len(dicom_entries),
        images_with_findings=0,
        anatomical_region=region_key,
        model_version=detector.model_version,
    )

    con_hallazgos = 0
    for idx, entry in enumerate(dicom_entries):
        result = await run_in_threadpool(detector.predict, entry.image)

        token = f"{study_token}{idx:04d}"
        orig_name = f"{token}_original.png"
        annot_name = f"{token}_annotated.png"
        entry.image.convert("RGB").save(str(UPLOADS_DIR / orig_name))
        _guardar_anotada(result, annot_name)

        if result.is_abnormal:
            con_hallazgos += 1
        veredicto = (
            "Con hallazgos sobre el umbral de clasificación" if result.is_abnormal
            else "Sin hallazgos sobre el umbral de clasificación"
        )

        analysis = crud.create_analysis(
            db=db,
            user_id=user.id,
            study_id=study.id,
            original_image_path=orig_name,
            annotated_image_path=annot_name,
            report_text=f"Imagen {idx + 1} — {entry.filename} — {veredicto}",
            max_detection_confidence=result.max_detection_confidence,
            is_abnormal=result.is_abnormal,
            inference_time_ms=result.inference_time_ms,
            anatomical_region=region_key,
            model_version=result.model_version,
            routing_method=routing_method,
            urgency=calculate_urgency(result.max_detection_confidence),
        )
        if result.boxes:
            crud.create_detection_boxes(db, analysis.id, _cajas_para_guardar(result))

    study.images_with_findings = con_hallazgos
    db.commit()

    # Redirigir (y no mostrar la página acá) evita que F5 reenvíe el ZIP.
    return RedirectResponse(url=f"/analysis/study/{study.id}", status_code=303)


def _fila_de_imagen(numero: int, a: Analysis) -> dict:
    """Los datos de una imagen del estudio para la plantilla (`numero` empieza en 1)."""
    cajas = sorted(a.detection_boxes, key=lambda b: b.confidence, reverse=True)
    return {
        "index": numero,
        "analysis_id": a.id,
        "source_filename": a.original_image_path,
        "original_image": a.original_image_path,
        "annotated_image": a.annotated_image_path,
        "max_detection_confidence": a.max_detection_confidence or 0.0,
        "is_abnormal": a.is_abnormal,
        "significant_count": a.findings_above_abnormal,
        "low_confidence_count": len(a.detection_boxes) - a.findings_above_abnormal,
        "inference_time_ms": a.inference_time_ms or 0.0,
        "urgency": urgency_detail(a.max_detection_confidence or 0.0, bool(a.is_abnormal)),
        "boxes": cajas,
        "boxes_json": _boxes_json(cajas),
    }


@router.get("/study/{study_id}", response_class=HTMLResponse)
async def view_study(
    study_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    study = crud.get_study_for_user(db, study_id, user.id)
    if not study:
        raise HTTPException(status_code=404, detail="Estudio no encontrado.")

    resultados = [
        _fila_de_imagen(i + 1, a)
        for i, a in enumerate(crud.get_analyses_by_study(db, study_id))
    ]

    # Fuera del dominio validado no hay urgencia que ordene: el visor abre en la primera imagen.
    fuera_de_dominio = (
        study.anatomical_region != DEFAULT_REGION
        or _modelo_anterior(study.anatomical_region, study.model_version)
    )
    if fuera_de_dominio and resultados:
        inicial = resultados[0]["index"]
    else:
        inicial = imagen_mas_urgente(resultados)
    imagen_inicial = next((r for r in resultados if r["index"] == inicial), None)

    return templates.TemplateResponse(
        request, "study_results.html",
        {
            **_base_context(request, user),
            "study_id": study.id,
            "initial_index": inicial,
            # El veredicto del estudio es el de la imagen en la que abre el visor.
            "urgencia_estudio": imagen_inicial["urgency"] if imagen_inicial else None,
            "original_zip": study.original_filename,
            "total_images": study.total_images,
            "images_with_findings": study.images_with_findings,
            "anatomical_region": _etiqueta_region(study.anatomical_region),
            "region_key": study.anatomical_region,
            "default_region": DEFAULT_REGION,
            "model_version": study.model_version,
            "modelo_anterior": _modelo_anterior(study.anatomical_region, study.model_version),
            "results": resultados,
        },
    )

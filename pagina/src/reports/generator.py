"""
generator.py — Informe PDF de un análisis.

Encabezado, datos del análisis, placa con los hallazgos, texto del informe,
alcance del modelo y aviso legal.
"""

import os
import tempfile
import unicodedata
from datetime import datetime
from io import BytesIO

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config.settings import (
    APP_NAME,
    LEGAL_DISCLAIMER,
    REGION_NO_INDICADA,
    domain_disclaimer,
)
from config.tiempo import ahora_local, fmt_local

# Colores de la marca KIBBO (los mismos hex que la capa --kb-* de style.css).
KB_TINTA = colors.HexColor("#1a1040")
KB_SECUNDARIO = colors.HexColor("#5C6074")
KB_VIOLETA = colors.HexColor("#461CCA")


# -- Texto seguro para Helvetica ----------------------------------------------
# Helvetica sólo dibuja cp1252 (todo el castellano entra); el resto saldría
# como cuadrados negros, así que se reemplaza por ASCII.

_EQUIVALENCIAS = {
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "⇒": "=>",
    "≥": ">=",
    "≤": "<=",
    "≠": "!=",
    "✓": "[ok]",
    "✔": "[ok]",
    "✅": "[ok]",
    "✗": "[x]",
    "✘": "[x]",
    "❌": "[x]",
    "⚠": "!",
}


def _dibujable(ch: str) -> bool:
    try:
        ch.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


def _equivalente_de_caja(ch: str) -> str | None:
    """ASCII para un carácter de dibujo de caja o de bloque, según su nombre Unicode."""
    nombre = unicodedata.name(ch, "")
    if nombre.startswith("BLOCK "):
        return "-"
    if not nombre.startswith("BOX DRAWINGS"):
        return None
    if "HORIZONTAL" in nombre:
        return "=" if "DOUBLE" in nombre else "-"
    if "VERTICAL" in nombre:
        return "|"
    return "+"


def _sin_acentos_raros(ch: str) -> str:
    """Descompone el carácter y se queda con lo dibujable (un emoji desaparece)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", ch)
        if _dibujable(c) and not unicodedata.combining(c)
    )


def glifos_seguros(texto: str) -> str:
    """`texto` con todo carácter que Helvetica no pueda dibujar reemplazado."""
    if not texto:
        return texto
    salida = []
    for ch in texto:
        if _dibujable(ch):
            salida.append(ch)
            continue
        reemplazo = _equivalente_de_caja(ch)
        if reemplazo is None:
            reemplazo = _EQUIVALENCIAS.get(ch)
        if reemplazo is None:
            reemplazo = _sin_acentos_raros(ch)
        salida.append(reemplazo)
    return "".join(salida)


# -- PDF ----------------------------------------------------------------------


def _estilos() -> dict:
    base = getSampleStyleSheet()
    base["Normal"].textColor = KB_TINTA  # sin negro puro en todo el informe
    normal = base["Normal"]
    return {
        "titulo": ParagraphStyle(
            "CustomTitle", parent=base["Title"],
            fontSize=18, spaceAfter=20, alignment=TA_CENTER, textColor=KB_VIOLETA,
        ),
        "subtitulo": ParagraphStyle(
            "Subtitle", parent=normal, fontSize=12, alignment=TA_CENTER,
        ),
        "cuerpo": ParagraphStyle(
            "Body", parent=normal, fontSize=10, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        # La salvedad de dominio va recuadrada y legible, no en el gris del pie.
        "alcance": ParagraphStyle(
            "Scope", parent=normal,
            fontSize=8, leading=10, alignment=TA_JUSTIFY,
            borderPadding=6, borderWidth=0.5, borderColor=KB_TINTA,
            spaceBefore=14, spaceAfter=6,
        ),
        "regla": ParagraphStyle(
            "Line", parent=normal, fontSize=6, textColor=KB_SECUNDARIO,
        ),
        "pie": ParagraphStyle(
            "Disclaimer", parent=normal,
            fontSize=7, textColor=KB_SECUNDARIO, alignment=TA_JUSTIFY, spaceBefore=20,
        ),
    }


def _parrafo(texto: str, estilo) -> Paragraph:
    """Todo párrafo del PDF pasa por acá para sanear los glifos."""
    return Paragraph(glifos_seguros(texto), estilo)


def _tabla_de_datos(fecha_analisis: str, analysis_id: str, doctor_name: str, patient_id: str) -> Table:
    filas = [
        ("Fecha del análisis:", fecha_analisis),
        ("ID de análisis:", analysis_id or "N/A"),
        ("Profesional:", doctor_name),
        ("Identificador del paciente:", patient_id),
    ]
    tabla = Table(
        [[glifos_seguros(a), glifos_seguros(b)] for a, b in filas],
        colWidths=[5 * cm, 10 * cm],
    )
    tabla.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tabla


def generate_pdf_report(
    annotated_image: Image.Image,
    report_text: str,
    doctor_name: str = "No especificado",
    patient_id: str = "Anónimo",
    analysis_id: str = "",
    region=REGION_NO_INDICADA,         # None = el análisis no guardó su región
    created_at: datetime | None = None,
    model_version=REGION_NO_INDICADA,  # omitido = informe del modelo vigente
) -> bytes:
    """Arma el PDF y devuelve sus bytes."""
    estilos = _estilos()
    impreso = ahora_local().strftime("%d/%m/%Y %H:%M:%S")
    fecha_analisis = fmt_local(created_at, "%d/%m/%Y %H:%M") or "sin registrar"

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        annotated_image.save(tmp.name)
        tmp_path = tmp.name

    elementos = [
        _parrafo(APP_NAME, estilos["titulo"]),
        _parrafo(
            "Prototipo académico de soporte a la decisión clínica — Detección de fracturas",
            estilos["subtitulo"],
        ),
        Spacer(1, 20),
        _tabla_de_datos(fecha_analisis, analysis_id, doctor_name, patient_id),
        Spacer(1, 20),
        _parrafo("<b>Radiografía con las cajas delimitadoras del detector:</b>", estilos["cuerpo"]),
        Spacer(1, 10),
        RLImage(tmp_path, width=14 * cm, height=14 * cm, kind="proportional"),
        Spacer(1, 15),
        _parrafo("<b>Informe del sistema:</b>", estilos["cuerpo"]),
        Spacer(1, 6),
    ]
    elementos += [
        _parrafo(linea, estilos["cuerpo"])
        for linea in report_text.split("\n")
        if linea.strip()
    ]
    elementos += [
        _parrafo(domain_disclaimer(region, model_version), estilos["alcance"]),
        Spacer(1, 30),
        _parrafo("-" * 80, estilos["regla"]),
        _parrafo(LEGAL_DISCLAIMER, estilos["pie"]),
        _parrafo(f"Generado por {APP_NAME} — emitido el {impreso}", estilos["pie"]),
    ]

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    doc.build(elementos)

    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    return buffer.getvalue()

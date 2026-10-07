"""
test_pagina.py — Lo esencial de la página, en un solo archivo.

1-2. Los números del modelo que muestra la página salen de los JSON que
     escriben modelo/test_interno.py y modelo/test_externo.py.
3-4. Login (en modo demo, con todas las cuentas) y rutas protegidas.
5-7. Subir una radiografía, aislamiento entre médicos y PDF.
     Una PNG de 16 bits se lee como en el entrenamiento; un archivo roto da error.
8-9. Métricas: sin margen de error y sólo con el modelo vigente.
     La página de alcance muestra los números de MODEL_METADATA.
10.  Los tests no escriben en app/uploads real.
11-12. El informe no se envía por correo; los rótulos de prioridad muestran
     el puntaje del detector en escala 0-1, nunca como porcentaje.
"""

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from tests.conftest import TEST_USER

RESULTADOS = Path(__file__).resolve().parents[2] / "modelo" / "resultados"
TOL = 5e-4
V2 = "yolov8m_v2_radiologico"


# ─── Ayudas ──────────────────────────────────────────────────────────────────

def png_bytes(lado: int = 128, valor: int = 0) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((lado, lado, 3), valor, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def png_16_bits(alto: int = 256, ancho: int = 256) -> bytes:
    """PNG de 16 bits en gris con un gradiente de 0 a 65535, como los de GRAZPEDWRI-DX."""
    gradiente = np.linspace(0, 65535, ancho).astype(np.uint16)
    ok, buf = cv2.imencode(".png", np.tile(gradiente, (alto, 1)))
    assert ok
    return buf.tobytes()


def crear_analisis(db_session, user_id: int, **kwargs):
    from app.database import crud

    datos = dict(
        user_id=user_id,
        original_image_path="a_original.png",
        annotated_image_path="a_annotated.png",
        report_text="Informe de prueba",
        max_detection_confidence=0.9,
        is_abnormal=True,
        inference_time_ms=12.0,
        anatomical_region="muneca_pediatrica",
        model_version="v1r",
        routing_method="manual",
        urgency="HIGH",
    )
    datos.update(kwargs)
    return crud.create_analysis(db_session, **datos)


def guardar_imagenes_de(analysis) -> None:
    """Escribe en disco las dos PNG que el generador de PDF va a abrir."""
    from config.settings import UPLOADS_DIR

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), 20).save(str(UPLOADS_DIR / analysis.original_image_path))
    Image.new("RGB", (48, 48), 90).save(str(UPLOADS_DIR / analysis.annotated_image_path))


# ─── 1-2. Números de la página == artefactos del modelo ─────────────────────

def test_metadata_igual_a_metricas_del_test_interno():
    from config.settings import ABNORMAL_THRESHOLD, DEFAULT_REGION, MODEL_METADATA, YOLO_WEIGHTS_PATH

    meta = MODEL_METADATA[DEFAULT_REGION]
    archivo = RESULTADOS / "metricas.json"
    assert archivo.exists(), "Falta modelo/resultados/metricas.json: la metadata no tiene respaldo"
    m = json.loads(archivo.read_text(encoding="utf-8"))

    # Caja
    for clave in ("mAP50", "mAP50_95", "precision", "recall"):
        assert meta[clave] == pytest.approx(m[clave], abs=TOL), clave
    assert meta["test_n"] == m["n_imagenes"]
    assert meta["test_n_instancias"] == m["n_instancias"]
    # Nivel imagen
    for clave in ("sensibilidad", "especificidad", "vpp", "auc_roc", "ap"):
        assert meta[clave] == pytest.approx(m[clave], abs=TOL), clave
    # Matriz de confusión
    assert (meta["test_TP"], meta["test_FN"], meta["test_FP"], meta["test_TN"]) == (
        m["TP"], m["FN"], m["FP"], m["TN"])
    assert meta["test_n_positivos"] == m["n_positivos"]
    assert meta["test_n_negativos"] == m["n_negativos"]
    # Umbral, pesos y fecha
    assert m["umbral"] == pytest.approx(ABNORMAL_THRESHOLD)
    assert Path(m["pesos"]).parts[-3:] == Path(YOLO_WEIGHTS_PATH).parts[-3:]
    assert m["fecha"] == meta["metrics_date"]


def test_metadata_igual_a_metricas_del_test_externo():
    from config.settings import ABNORMAL_THRESHOLD, DEFAULT_REGION, MODEL_METADATA, YOLO_WEIGHTS_PATH

    e = MODEL_METADATA[DEFAULT_REGION]["test_externo"]
    archivo = RESULTADOS / "metricas_externo.json"
    assert archivo.exists(), "Falta modelo/resultados/metricas_externo.json: la metadata externa no tiene respaldo"
    ext = json.loads(archivo.read_text(encoding="utf-8"))

    assert e["n_casos"] == ext["n_casos"]
    assert e["n_imagenes"] == ext["n_imagenes"]
    assert e["fecha"] == ext["fecha"]
    assert e["umbral"] == pytest.approx(ext["umbral"])
    assert ext["umbral"] == pytest.approx(ABNORMAL_THRESHOLD)
    assert Path(ext["pesos"]).parts[-3:] == Path(YOLO_WEIGHTS_PATH).parts[-3:]
    for nivel in ("por_caso", "por_imagen"):
        a, b = e[nivel], ext[nivel]
        assert (a["TP"], a["FN"], a["FP"], a["TN"]) == (b["TP"], b["FN"], b["FP"], b["TN"]), nivel
        for clave in ("sensibilidad", "especificidad"):
            assert a[clave] == pytest.approx(b[clave], abs=TOL), (nivel, clave)


# ─── 3-4. Sesión ─────────────────────────────────────────────────────────────

def test_login_correcto_e_incorrecto(client, users):
    r = client.post("/login", data={"email": TEST_USER["email"], "password": "mala"})
    assert r.status_code == 401
    assert "incorrectos" in r.text.lower()

    r = client.post("/login", data={"email": TEST_USER["email"], "password": TEST_USER["password"]})
    assert r.status_code == 303
    assert r.headers["location"] == "/analysis/upload"


def test_el_panel_demo_lista_la_cuenta_admin_solo_en_modo_demo(client, monkeypatch):
    from app.database.demo_seed import DEMO_USERS
    from app.routes import auth_routes

    admin = next(u for u in DEMO_USERS if u["is_admin"])

    monkeypatch.setattr(auth_routes, "DEMO_MODE", True)
    html = client.get("/login").text
    assert "Cuentas de demostración" in html
    assert all(u["email"] in html for u in DEMO_USERS)
    assert '<span class="nav-user-tag">admin</span>' in html
    assert html.count('class="demo-row"') == len(DEMO_USERS)

    monkeypatch.setattr(auth_routes, "DEMO_MODE", False)
    html = client.get("/login").text
    assert "Cuentas de demostración" not in html
    assert admin["email"] not in html and admin["password"] not in html


def test_sin_sesion_las_pantallas_redirigen_al_login(client):
    for ruta in ("/", "/analysis/upload", "/analysis/history", "/dashboard/", "/dashboard/api/stats"):
        r = client.get(ruta)
        assert r.status_code == 303, ruta
        assert r.headers["location"].startswith("/login"), ruta


# ─── 5-7. Análisis ───────────────────────────────────────────────────────────

def test_subir_una_radiografia_crea_un_analisis(auth_client, db_session, users):
    from app.database import crud

    r = auth_client.post(
        "/analysis/upload",
        files={"file": ("rx.png", png_bytes(128), "image/png")},
    )
    assert r.status_code == 303, r.text[:400]
    assert "/results" in r.headers["location"]

    analisis = crud.get_analyses_by_user(db_session, users["principal"])
    assert len(analisis) == 1
    assert analisis[0].model_version
    assert analisis[0].anatomical_region == "muneca_pediatrica"
    assert auth_client.get(r.headers["location"]).status_code == 200


def test_un_medico_no_ve_el_analisis_la_imagen_ni_el_pdf_de_otro(auth_client, db_session, users):
    ajeno = crear_analisis(db_session, users["otro"], annotated_image_path="secreto_annotated.png")
    guardar_imagenes_de(ajeno)

    assert auth_client.get(f"/analysis/{ajeno.id}/results").status_code == 404
    assert auth_client.get("/analysis/imagen/secreto_annotated.png").status_code == 404
    assert auth_client.get(f"/analysis/{ajeno.id}/pdf").status_code == 404


def test_el_pdf_de_un_analisis_propio_se_genera(auth_client, db_session, users):
    a = crear_analisis(db_session, users["principal"])
    guardar_imagenes_de(a)

    r = auth_client.get(f"/analysis/{a.id}/pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")


def test_una_png_de_16_bits_se_lee_como_en_el_entrenamiento(
    auth_client, db_session, users, monkeypatch
):
    """Se reescala a 8 bits (cv2.imdecode en gris), no se satura en 255 como con Pillow."""
    from app.database import crud
    from config.settings import UPLOADS_DIR
    from src.detection.predict import FractureDetector

    recibidas = []
    predict_real = FractureDetector.predict

    def predict_espia(self, image):
        recibidas.append(image)
        return predict_real(self, image)

    monkeypatch.setattr(FractureDetector, "predict", predict_espia)

    contenido = png_16_bits()
    r = auth_client.post(
        "/analysis/upload",
        files={"file": ("rx16.png", contenido, "image/png")},
    )
    assert r.status_code == 303, r.text[:400]

    esperada = cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_GRAYSCALE)
    al_detector = np.array(recibidas[-1].convert("L"))
    assert np.array_equal(al_detector, esperada)
    assert (al_detector == 255).mean() < 0.05  # con Pillow era casi el 100 %
    assert al_detector.min() < 10 and al_detector.max() > 245

    # La original guardada (sobre la que se dibujan las cajas) es la misma placa.
    analisis = crud.get_analyses_by_user(db_session, users["principal"])[0]
    guardada = np.array(Image.open(UPLOADS_DIR / analisis.original_image_path).convert("L"))
    assert np.array_equal(guardada, esperada)
    anotada = Image.open(UPLOADS_DIR / analisis.annotated_image_path)
    assert anotada.size == (esperada.shape[1], esperada.shape[0])


def test_un_archivo_que_no_es_imagen_da_el_error_de_archivo_invalido(auth_client):
    r = auth_client.post(
        "/analysis/upload",
        files={"file": ("rx.png", b"esto no es una imagen", "image/png")},
    )
    assert r.status_code == 200
    assert "el archivo no es una imagen válida o está dañado" in r.text


# ─── 8-9. Métricas ───────────────────────────────────────────────────────────

def test_metricas_responde_y_no_muestra_el_margen_de_error(auth_client):
    r = auth_client.get("/dashboard/")
    assert r.status_code == 200
    html = r.text
    assert "IC 95" not in html and "IC&nbsp;95" not in html
    assert "intervalo de confianza" not in html


def test_un_analisis_de_un_modelo_anterior_no_cuenta_ni_muestra_su_informe(
    auth_client, db_session, users
):
    from app.database import crud

    vigente = crear_analisis(db_session, users["principal"], model_version="v1r")
    crud.create_feedback(db_session, vigente.id, agreed=True)
    viejo = crear_analisis(
        db_session, users["principal"], model_version=V2,
        report_text="Probabilidad estimada de fractura: 95 % (score calibrado por Platt scaling).",
    )
    crud.create_feedback(db_session, viejo.id, agreed=False)

    datos = auth_client.get("/dashboard/api/stats").json()
    assert datos["total_analyses"] == 1
    assert datos["agreement_details"]["total"] == 1

    html = auth_client.get(f"/analysis/{viejo.id}/results").text
    assert "Platt" not in html and "Probabilidad estimada" not in html
    assert "lo generó un modelo anterior y no se muestra" in html


def test_la_pagina_de_alcance_muestra_los_numeros_del_modelo_vigente(client):
    from config.settings import (
        ABNORMAL_THRESHOLD, CONFIDENCE_THRESHOLD, DEFAULT_REGION, MODEL_METADATA,
        URGENCY_HIGH_THRESHOLD,
    )

    meta = MODEL_METADATA[DEFAULT_REGION]
    ext = meta["test_externo"]
    from app.plantillas import _decimal
    coma = _decimal
    pct = lambda x: coma(x * 100, 1) + "&nbsp;%"  # noqa: E731

    r = client.get("/aviso-legal")
    assert r.status_code == 200
    html = r.text
    esperados = [
        pct(meta["sensibilidad"]), pct(meta["especificidad"]), coma(meta["auc_roc"], 3),
        coma(meta["mAP50"], 3), coma(meta["recall"], 3),
        pct(ext["por_caso"]["sensibilidad"]), pct(ext["por_caso"]["especificidad"]),
        pct(ext["por_imagen"]["sensibilidad"]), pct(ext["por_imagen"]["especificidad"]),
        coma(ABNORMAL_THRESHOLD, 2), coma(CONFIDENCE_THRESHOLD, 2), coma(URGENCY_HIGH_THRESHOLD, 2),
        f"{meta['test_n_pacientes']} pacientes", meta["metrics_date"],
    ]
    for texto in esperados:
        assert texto in html, texto
    for viejo in ("multicéntric", "calibrad", "v2", "0,25", "0,50", "IC 95"):
        assert viejo not in html, viejo


# ─── 10. Carpeta de uploads ──────────────────────────────────────────────────

def test_los_tests_no_escriben_en_los_uploads_reales():
    from config.settings import BASE_DIR, UPLOADS_DIR

    assert UPLOADS_DIR.resolve() != (BASE_DIR / "app" / "uploads").resolve()


# ─── 11-12. Sin envío por correo y puntaje en escala 0-1 ────────────────────

def test_el_informe_no_se_envia_por_correo(auth_client, db_session, users):
    a = crear_analisis(db_session, users["principal"])

    r = auth_client.post(f"/analysis/{a.id}/email", data={"destino": "colega@hospital.com"})
    assert r.status_code in (404, 405)

    html = auth_client.get(f"/analysis/{a.id}/results").text
    assert "Descargar informe PDF" in html
    for rastro in ("Enviar por mail", "mail-dialogo", "/email", "SMTP"):
        assert rastro not in html, rastro


def test_los_rotulos_de_prioridad_muestran_el_puntaje_en_escala_0_a_1():
    from app.services.routing_service import urgency_detail
    from config.settings import ABNORMAL_THRESHOLD, CONFIDENCE_THRESHOLD, URGENCY_HIGH_THRESHOLD

    coma = lambda x: f"{x:.2f}".replace(".", ",")  # noqa: E731
    alto = (URGENCY_HIGH_THRESHOLD + 1) / 2
    medio = (ABNORMAL_THRESHOLD + URGENCY_HIGH_THRESHOLD) / 2
    limite = (CONFIDENCE_THRESHOLD + ABNORMAL_THRESHOLD) / 2
    bajo = CONFIDENCE_THRESHOLD / 2

    niveles = [urgency_detail(alto, True), urgency_detail(medio, True),
               urgency_detail(limite, False), urgency_detail(bajo, False), urgency_detail(0.0, False)]
    assert [n["nivel"] for n in niveles] == ["HIGH", "MEDIUM", "LOW_BORDERLINE", "LOW", "LOW"]
    assert niveles[0]["titulo"] == "PRIORITARIO — Puntaje alto del detector; confirmar con lectura médica"
    for n in niveles:
        assert "%" not in n["titulo"] + n["detalle"], n
    assert coma(alto) in niveles[0]["detalle"] and coma(URGENCY_HIGH_THRESHOLD) in niveles[0]["detalle"]
    assert coma(medio) in niveles[1]["detalle"] and coma(ABNORMAL_THRESHOLD) in niveles[1]["detalle"]
    assert coma(limite) in niveles[2]["detalle"] and coma(ABNORMAL_THRESHOLD) in niveles[2]["detalle"]
    assert coma(bajo) in niveles[3]["detalle"]

"""
auth_routes.py — Inicio y cierre de sesión.

  GET  /login    formulario (en modo demo lista las cuentas de prueba)
  POST /login    valida credenciales y abre la sesión
  POST /logout   cierra la sesión
"""

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import crud
from app.database.db import get_db
from app.database.demo_seed import demo_credentials
from app.dependencies.auth import get_current_user, login_user, logout_user
from app.dependencies.csrf import get_csrf_token, verify_csrf
from app.plantillas import crear_templates
from app.services.rate_limit import check_and_consume
from config.settings import APP_NAME, DEMO_MODE, LEGAL_DISCLAIMER

router = APIRouter()

templates = crear_templates()

# Tope contra la fuerza bruta de contraseñas.
_MAX_INTENTOS_POR_HORA = 40


def _pagina_login(request: Request, status_code: int = 200, **extra):
    contexto = {
        "app_name": APP_NAME,
        "disclaimer": LEGAL_DISCLAIMER,
        "demo_mode": DEMO_MODE,
        "demo_users": demo_credentials() if DEMO_MODE else [],
        "csrf_token": get_csrf_token(request),
        **extra,
    }
    return templates.TemplateResponse(request, "login.html", contexto, status_code=status_code)


def _destino_seguro(next_url: str | None) -> str:
    """Sólo rutas internas: evita que `next` redirija a otro sitio."""
    if not next_url or not next_url.startswith("/") or next_url.startswith("//"):
        return "/analysis/upload"
    return next_url


@router.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    next: str | None = None,
    db: Session = Depends(get_db),
):
    if get_current_user(request, db) is not None:
        return RedirectResponse(url=_destino_seguro(next), status_code=303)
    return _pagina_login(request, next=next or "")


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form(""),
    db: Session = Depends(get_db),
):
    ip = request.client.host if request.client else "desconocida"
    if not check_and_consume(f"login:{ip}", _MAX_INTENTOS_POR_HORA):
        return _pagina_login(
            request, status_code=429,
            next=next, error="Demasiados intentos fallidos. Espere unos minutos antes de volver a intentarlo.",
        )

    user = crud.authenticate_user(db, email, password)
    if user is None:
        # El mismo mensaje para email inexistente y contraseña mala: no revela qué cuentas existen.
        return _pagina_login(
            request, status_code=401,
            next=next, error="Correo electrónico o contraseña incorrectos.", email=email,
        )

    login_user(request, user)
    return RedirectResponse(url=_destino_seguro(next), status_code=303)


@router.post("/logout")
async def logout(request: Request, _: None = Depends(verify_csrf)):
    logout_user(request)
    return RedirectResponse(url="/login", status_code=303)

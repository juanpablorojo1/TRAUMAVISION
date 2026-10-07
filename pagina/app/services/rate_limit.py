"""
rate_limit.py — Límite de usos por hora, en memoria.

Frena la fuerza bruta en el login. Se reinicia con el proceso: alcanza
para un prototipo de un solo worker.
"""

import threading
import time
from collections import defaultdict, deque

VENTANA_SEGUNDOS = 3600

_usos_por_clave: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def check_and_consume(clave: str, limite: int) -> bool:
    """Registra un uso y devuelve True si todavía quedaba cupo en la última hora."""
    ahora = time.time()
    with _lock:
        usos = _usos_por_clave[clave]
        while usos and usos[0] < ahora - VENTANA_SEGUNDOS:
            usos.popleft()
        if len(usos) >= limite:
            return False
        usos.append(ahora)
        return True


def reset() -> None:
    """Borra todos los contadores (lo usan los tests)."""
    with _lock:
        _usos_por_clave.clear()

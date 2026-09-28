import hashlib
import hmac
import secrets
import time
from urllib.parse import urlsplit

from fastapi import Request

SESSION_SECONDS = 12 * 60 * 60
COOKIE_NAME = "pfl_session"


def cookie_value(secret):
    payload = f"{int(time.time()) + SESSION_SECONDS}.{secrets.token_hex(16)}"
    signature = hmac.new(secret.encode(), f"session:{payload}".encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def session_payload(request: Request, secret):
    value = request.cookies.get(COOKIE_NAME, "")
    parts = value.split(".")
    if len(parts) != 3:
        return None
    expiry, nonce, signature = parts
    if (
        not expiry.isdecimal()
        or len(nonce) != 32
        or any(char not in "0123456789abcdef" for char in nonce)
    ):
        return None
    now = int(time.time())
    if int(expiry) <= now or int(expiry) > now + SESSION_SECONDS:
        return None
    payload = f"{expiry}.{nonce}"
    expected = hmac.new(secret.encode(), f"session:{payload}".encode(), hashlib.sha256).hexdigest()
    return payload if hmac.compare_digest(signature, expected) else None


def csrf_token(secret, payload):
    return hmac.new(secret.encode(), f"csrf:{payload}".encode(), hashlib.sha256).hexdigest()


def service_authenticated(request: Request, secret):
    authorization = request.headers.get("authorization", "")
    if not secret or not authorization.startswith("Bearer "):
        return False
    supplied = authorization.removeprefix("Bearer ")
    return hmac.compare_digest(supplied, secret)


def origin_allowed(request: Request, app_origin):
    expected = urlsplit(app_origin)
    host = request.headers.get("host", "")
    if host.casefold() != expected.netloc.casefold():
        return False
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != app_origin.rstrip("/"):
        return False
    fetch_site = request.headers.get("sec-fetch-site")
    return fetch_site not in {"cross-site", "same-site"}

import hmac
import logging
import re

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from medical_assistant.settings import get_settings

from .codex import CodexProvider
from .common import ALLOWED_MODELS, MAX_PAYLOAD_BYTES, ProviderError, check_request


class InferenceBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    payload: dict
    role: str
    model: str
    request_id: str
    final_only: bool


settings = get_settings()
provider = CodexProvider(settings)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
logger = logging.getLogger(__name__)


def authorize(authorization: str) -> None:
    expected = f"Bearer {settings.bridge_token}"
    if not settings.bridge_token or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
async def health(authorization: str = Header(default="")):
    authorize(authorization)
    return await provider.readiness()


@app.post("/infer")
async def infer(request: Request, authorization: str = Header(default="")):
    authorize(authorization)
    body = None
    body_bytes = bytearray()
    async for chunk in request.stream():
        body_bytes.extend(chunk)
        if len(body_bytes) > MAX_PAYLOAD_BYTES + 1024:
            raise HTTPException(status_code=413, detail="request too large")
    try:
        body = InferenceBody.model_validate_json(body_bytes)
        check_request(body.payload, body.role, body.model, body.request_id)
        return await provider.generate(
            body.payload,
            role=body.role,
            model=body.model,
            request_id=body.request_id,
            final_only=body.final_only,
        )
    except ProviderError as exc:
        logger.warning(
            "codex_failure request_id=%s role=%s model=%s code=%s exit_code=%s diagnostic=%s",
            body.request_id
            if body and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", body.request_id)
            else "invalid",
            body.role if body and body.role in ALLOWED_MODELS else "invalid",
            body.model if body and body.model in {"gpt-6-sol", "gpt-6-luna"} else "invalid",
            exc.code,
            exc.exit_code,
            exc.diagnostic,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": exc.code,
                "message": "provider inference failed",
                "usage": {
                    key: value
                    for key, value in exc.usage.items()
                    if key
                    in {
                        "input_tokens",
                        "cached_input_tokens",
                        "cache_write_input_tokens",
                        "output_tokens",
                        "reasoning_output_tokens",
                    }
                    and type(value) is int
                    and value >= 0
                },
            },
        ) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="invalid inference request") from exc


@app.post("/cancel/{request_id}")
async def cancel(request_id: str, authorization: str = Header(default="")):
    authorize(authorization)
    if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id) is None:
        raise HTTPException(status_code=422, detail="invalid request id")
    await provider.cancel(request_id)
    return {"cancelled": True}


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8768, access_log=False)


if __name__ == "__main__":
    main()

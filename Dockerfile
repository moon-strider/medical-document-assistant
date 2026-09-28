FROM node:22.22.1-bookworm-slim@sha256:4f77a690f2f8946ab16fe1e791a3ac0667ae1c3575c3e4d0d4589e9ed5bfaf3d AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.12.10@sha256:2bb3ebca0a796a155094a27773d290c4b074572e6107f171d88d086682fd2500 AS uv

FROM python:3.13.11-slim-bookworm@sha256:20080e807bfc404f8450b185cf0fc95d553462673598549613735f70a5b4d5d0 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_CACHE_DIR=/root/.cache/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY backend/src ./backend/src
COPY migrations ./migrations
COPY prompts ./prompts
RUN --mount=type=cache,id=medical-document-assistant-uv,target=/root/.cache/uv,sharing=locked uv sync --locked --no-dev --no-editable \
    --no-install-package torch \
    --no-install-package cuda-bindings \
    --no-install-package cuda-pathfinder \
    --no-install-package cuda-toolkit \
    --no-install-package nvidia-cublas \
    --no-install-package nvidia-cuda-cupti \
    --no-install-package nvidia-cuda-nvrtc \
    --no-install-package nvidia-cuda-runtime \
    --no-install-package nvidia-cudnn-cu13 \
    --no-install-package nvidia-cufft \
    --no-install-package nvidia-cufile \
    --no-install-package nvidia-curand \
    --no-install-package nvidia-cusolver \
    --no-install-package nvidia-cusparse \
    --no-install-package nvidia-cusparselt-cu13 \
    --no-install-package nvidia-nccl-cu13 \
    --no-install-package nvidia-nvjitlink \
    --no-install-package nvidia-nvshmem-cu13 \
    --no-install-package nvidia-nvtx \
    --no-install-package triton
ARG TARGETARCH
RUN --mount=type=cache,id=medical-document-assistant-uv,target=/root/.cache/uv,sharing=locked case "$TARGETARCH" in \
      arm64) wheel='https://download-r2.pytorch.org/whl/cpu/torch-2.14.0%2Bcpu-cp313-cp313-manylinux_2_28_aarch64.whl#sha256=092d5c12938850dfbd90a654b3c8dac34c33e300f88eb19ee6f4ef93992c6347' ;; \
      amd64) wheel='https://download-r2.pytorch.org/whl/cpu/torch-2.14.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=160e1bc46aeded3111d2801f8ae10dc9a1b946843a7e126b4dbf5e19c5706e95' ;; \
      *) exit 1 ;; \
    esac && uv pip install --python /opt/venv/bin/python --no-deps "$wheel" && /opt/venv/bin/python -c "import torch; assert torch.__version__ == '2.14.0+cpu'"

FROM python:3.13.11-slim-bookworm@sha256:20080e807bfc404f8450b185cf0fc95d553462673598549613735f70a5b4d5d0
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/backend/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/models/hf XDG_CACHE_HOME=/models/cache HOME=/tmp
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app/pyproject.toml /app/uv.lock /app/
COPY --from=build /app/backend/src /app/backend/src
COPY --from=build /app/migrations /app/migrations
COPY --from=build /app/prompts /app/prompts
COPY --from=frontend /build/frontend/dist /app/frontend/dist
COPY deploy/worker_entry.py /app/deploy/worker_entry.py
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home --home-dir /tmp app && mkdir -p /data /models/hf /models/cache && chown -R app:app /data /models
USER 10001:10001
EXPOSE 8080
CMD ["python", "-m", "medical_assistant.server"]

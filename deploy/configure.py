import os
import secrets
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "tmp" / "work" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = runtime / ".env"
    if target.exists():
        print(f"Keeping existing configuration: {target}")
        return
    ports = {
        key: int(os.environ.get(key, default))
        for key, default in (
            ("PFL_HTTP_PORT", "8080"),
            ("PFL_POSTGRES_PORT", "5439"),
            ("PFL_LANGFUSE_PORT", "3045"),
            ("PFL_MINIO_PORT", "9095"),
        )
    }
    if not all(1 <= port <= 65535 for port in ports.values()):
        raise ValueError("Published ports must be between 1 and 65535")
    if len(set(ports.values())) != len(ports):
        raise ValueError("Published ports must be distinct")
    http_port = ports["PFL_HTTP_PORT"]
    postgres_port = ports["PFL_POSTGRES_PORT"]
    langfuse_port = ports["PFL_LANGFUSE_PORT"]
    values = {
        key: secrets.token_hex(24)
        for key in [
            "POSTGRES_PASSWORD", "PFL_DB_PASSWORD", "PFL_DB_READ_PASSWORD",
            "LANGFUSE_DB_PASSWORD", "LANGFUSE_SALT", "LANGFUSE_NEXTAUTH_SECRET",
            "CLICKHOUSE_PASSWORD", "REDIS_PASSWORD", "MINIO_PASSWORD",
            "LANGFUSE_USER_PASSWORD", "PFL_LAUNCH_TOKEN", "PFL_SERVICE_TOKEN",
            "PFL_BRIDGE_TOKEN",
        ]
    }
    values.update({
        "LANGFUSE_ENCRYPTION_KEY": secrets.token_hex(32),
        "PFL_LANGFUSE_PUBLIC_KEY": f"pk-lf-{secrets.token_hex(16)}",
        "PFL_LANGFUSE_SECRET_KEY": f"sk-lf-{secrets.token_hex(24)}",
        "PFL_LANGFUSE_BASE_URL": f"http://127.0.0.1:{langfuse_port}",
        "PFL_LANGFUSE_PUBLIC_URL": f"http://127.0.0.1:{langfuse_port}",
        **{key: str(port) for key, port in ports.items()},
        "PFL_APP_ORIGIN": f"http://127.0.0.1:{http_port}",
        "PFL_DATA_DIR": str(runtime / "app-data"),
        "PFL_CODEX_HOME": str(runtime / "codex-home"),
        "PFL_FRONTEND_DIR": str(root / "frontend" / "dist"),
        "PFL_PROVIDER": "codex",
    })
    values["PFL_DATABASE_URL"] = f"postgresql+psycopg://assistant:{values['PFL_DB_PASSWORD']}@127.0.0.1:{postgres_port}/assistant"
    values["PFL_READ_DATABASE_URL"] = f"postgresql+psycopg://assistant_readonly:{values['PFL_DB_READ_PASSWORD']}@127.0.0.1:{postgres_port}/assistant"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")
    print(f"Created private configuration: {target}")


if __name__ == "__main__":
    main()

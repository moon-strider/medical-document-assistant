import os

import uvicorn

from medical_assistant.api import app


def main():
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=os.environ.get("PFL_BIND_HOST", "0.0.0.0"),
            port=int(os.environ.get("PFL_BIND_PORT", "8080")),
            access_log=False,
            timeout_graceful_shutdown=30,
        )
    )
    app.state.request_shutdown = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

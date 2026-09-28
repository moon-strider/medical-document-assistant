import logging
import os
import signal
import socket
import threading
import uuid

from medical_assistant.settings import get_settings
from medical_assistant.storage import Store
from medical_assistant.worker import run_once


def main():
    logging.basicConfig(level=logging.INFO)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    settings = get_settings()
    store = Store(settings)
    store.migrate()
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    try:
        while not stopping.is_set():
            try:
                if not run_once(store, settings, worker_id):
                    stopping.wait(1)
            except Exception:
                logging.exception("Worker loop failed")
                stopping.wait(3)
    finally:
        store.engine.dispose()


if __name__ == "__main__":
    main()

from __future__ import annotations

import queue

try:
    from tkinterdnd2 import TkinterDnD

    _TK_ROOT_CLS = TkinterDnD.Tk
except ImportError:
    import tkinter as tk

    _TK_ROOT_CLS = tk.Tk

from app.jobs import Job, WorkerEvent
from app.ui import MainWindow
from app.worker import SENTINEL, TranscriptionWorker
from src import history_db


def main() -> int:
    db_conn = history_db.init_db()

    job_queue: "queue.Queue[Job | object]" = queue.Queue()
    event_queue: "queue.Queue[WorkerEvent]" = queue.Queue()

    worker = TranscriptionWorker(job_queue, event_queue, db_conn)
    worker.start()

    root = _TK_ROOT_CLS()
    MainWindow(root, job_queue, event_queue, db_conn)

    def on_close() -> None:
        job_queue.put(SENTINEL)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    db_conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

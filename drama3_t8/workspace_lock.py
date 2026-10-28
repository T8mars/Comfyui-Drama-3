"""Keep a second process from changing an active workspace's durable queue."""
from pathlib import Path
import os


class WorkspaceBusy(RuntimeError):
    pass


class WorkspaceLock:
    def __init__(self, path):
        self.file = Path(path).open("a+b")
        try:
            self.file.seek(0, 2)
            if self.file.tell() == 0:
                self.file.write(b"\0")
                self.file.flush()
            self.file.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise WorkspaceBusy("此工作台已在运行，请使用已打开的页面") from exc
        except BaseException:
            self.file.close()
            raise

    def close(self):
        if not self.file.closed:
            try:
                self.file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file, fcntl.LOCK_UN)
            finally:
                self.file.close()

# Authors: Xander Cai

import os
import sys
from pathlib import Path


class Logger(object):
    def __init__(self, path=None):
        self.console = sys.stdout
        self.path = path
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            if Path(path).exists():
                self.file = open(path, "a", encoding="utf-8")
            else:
                self.file = open(path, "w", encoding="utf-8")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

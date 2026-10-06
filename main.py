"""MusiCut 入口"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许从源码目录直接运行 main.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MusiCut")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

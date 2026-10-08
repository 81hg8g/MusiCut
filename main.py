"""MusiCut 入口"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许从源码目录直接运行 main.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from src import __version__
from src.gui.main_window import MainWindow
from src.utils.paths import resource_path

# Windows 任务栏图标分组标识；不设置会沿用默认图标
_APP_USER_MODEL_ID = "MusiCut.App"


def _set_app_user_model_id() -> None:
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


def _load_icon() -> QIcon:
    """优先用多尺寸 ICO，缺失时回退 PNG。"""
    for name in ("MusiCut.ico", "MusiCut.png"):
        path = resource_path("assets", name)
        if path.exists():
            return QIcon(str(path))
    return QIcon()


def main() -> int:
    _set_app_user_model_id()
    app = QApplication(sys.argv)
    app.setApplicationName("MusiCut")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(_load_icon())

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

"""资源路径解析（兼容源码运行与 PyInstaller 打包）"""
from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    """资源根目录：打包后为解包目录，源码运行时为项目根目录。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    """拼接资源路径，例如 resource_path('assets', 'MusiCut.ico')。"""
    return app_root().joinpath(*parts)

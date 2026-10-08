"""已使用封面记录：保证封面图片不被重复选用"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .settings import settings_dir

_USED_FILE = "used_covers.json"


def used_path() -> Path:
    """已使用封面记录文件路径。"""
    return settings_dir() / _USED_FILE


@dataclass(frozen=True)
class UsedCovers:
    """不可变的已使用封面视图；写入返回新实例。"""
    paths: frozenset[str] = frozenset()


def load_used() -> UsedCovers:
    """读取记录；文件缺失或损坏时返回空记录。"""
    path = used_path()
    if not path.exists():
        return UsedCovers()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return UsedCovers()
    paths = raw.get("paths") if isinstance(raw, dict) else None
    if not isinstance(paths, list):
        return UsedCovers()
    return UsedCovers(paths=frozenset(str(p) for p in paths))


def save_used(used: UsedCovers) -> None:
    """原子写入记录（.tmp 再 replace），路径按字典序排列。"""
    directory = settings_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = used_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"paths": sorted(used.paths)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def register_used(used: UsedCovers, new_paths: tuple[str, ...]) -> UsedCovers:
    """并集登记新路径并持久化，返回更新后的记录。"""
    updated = UsedCovers(paths=used.paths | frozenset(new_paths))
    save_used(updated)
    return updated

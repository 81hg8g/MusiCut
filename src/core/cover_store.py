"""封面意境索引与向量矩阵的持久化（不可变数据，原子写入）"""
from __future__ import annotations

import json
import os
import pickle
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .settings import settings_dir

_INDEX_FILE = "cover_index.json"
_EMB_FILE = "cover_embeddings.npz"

DEFAULT_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"

_EMBED_DIM = 1024


@dataclass(frozen=True)
class CoverEntry:
    """单张封面的扫描状态（name 为封面目录下的相对文件名）。"""
    name: str
    mtime_ns: int
    size: int
    desc: str = ""
    status: str = "ok"   # ok / failed / missing


@dataclass(frozen=True)
class CoverIndex:
    """封面目录的意境索引视图；更新一律返回新实例。"""
    folder: str
    images: tuple[CoverEntry, ...] = ()
    vl_model: str = DEFAULT_VL_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL

    def entry(self, name: str) -> CoverEntry | None:
        for e in self.images:
            if e.name == name:
                return e
        return None

    def ok_entries(self) -> tuple[CoverEntry, ...]:
        """状态 ok 且已有意境描述的条目。"""
        return tuple(e for e in self.images if e.status == "ok" and e.desc)

    def with_entries(self, new: Iterable[CoverEntry]) -> "CoverIndex":
        """按 name 合并：同名替换、新名追加；原有相对顺序在前。"""
        merged = list(self.images)
        pos = {e.name: i for i, e in enumerate(merged)}
        for e in new:
            if e.name in pos:
                merged[pos[e.name]] = e
            else:
                pos[e.name] = len(merged)
                merged.append(e)
        return replace(self, images=tuple(merged))

    def mark_missing(self, names: Iterable[str]) -> "CoverIndex":
        """把给定 name 的状态置为 missing，其余条目不动。"""
        targets = frozenset(names)
        images = tuple(
            replace(e, status="missing") if e.name in targets else e
            for e in self.images
        )
        return replace(self, images=images)


def index_paths() -> tuple[Path, Path]:
    """返回（索引文件路径，向量文件路径）。"""
    directory = settings_dir()
    return directory / _INDEX_FILE, directory / _EMB_FILE


def load_index(folder: str) -> CoverIndex:
    """读取索引；文件缺失/损坏/folder 不一致时返回该 folder 的空索引。"""
    path = index_paths()[0]
    if not path.exists():
        return CoverIndex(folder=folder)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        images = tuple(
            CoverEntry(
                name=str(item["name"]),
                mtime_ns=int(item["mtime_ns"]),
                size=int(item["size"]),
                desc=str(item.get("desc", "")),
                status=str(item.get("status", "ok")),
            )
            for item in raw["images"]
        )
        index = CoverIndex(
            folder=str(raw["folder"]),
            images=images,
            vl_model=str(raw.get("vl_model", DEFAULT_VL_MODEL)),
            embed_model=str(raw.get("embed_model", DEFAULT_EMBED_MODEL)),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return CoverIndex(folder=folder)
    if os.path.normcase(index.folder) != os.path.normcase(folder):
        return CoverIndex(folder=folder)
    return index


def save_index(index: CoverIndex) -> None:
    """原子写入索引（.tmp 再 replace），UTF-8。"""
    directory = settings_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = index_paths()[0]
    tmp = path.with_suffix(".json.tmp")
    payload = {
        "folder": index.folder,
        "vl_model": index.vl_model,
        "embed_model": index.embed_model,
        "images": [asdict(e) for e in index.images],
    }
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_embeddings(
    expected_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]] | None:
    """读取 npz 向量矩阵并校验；任一不符返回 None。"""
    path = index_paths()[1]
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            names = tuple(str(n) for n in data["names"])
            embeddings = np.asarray(data["embeddings"])
    except (
        OSError,
        EOFError,
        KeyError,
        ValueError,
        pickle.UnpicklingError,
        zipfile.BadZipFile,
    ):
        return None
    if embeddings.ndim != 2 or embeddings.shape[1] != _EMBED_DIM:
        return None
    if embeddings.shape[0] != len(names) or names != tuple(expected_names):
        return None
    return embeddings, names


def save_embeddings(names: tuple[str, ...], embeddings: np.ndarray) -> None:
    """原子写入 npz：names（行名）+ embeddings（矩阵）。"""
    directory = settings_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = index_paths()[1]
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    np.savez(
        tmp,
        names=np.array(names, dtype=object),
        embeddings=np.asarray(embeddings),
    )
    tmp.replace(path)

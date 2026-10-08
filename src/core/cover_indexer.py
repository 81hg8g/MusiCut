"""封面增量索引：扫描变更 -> 视觉模型生成意境描述 -> 文本向量嵌入。

所有数据结构不可变；网络请求复用 ASR 的硅基流动平台配置。
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import requests
from PIL import Image

from .cover_store import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_VL_MODEL,
    CoverEntry,
    CoverIndex,
    load_embeddings,
    save_embeddings,
    save_index,
)
from .settings import Settings

_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_MAX_SIDE = 768
_JPEG_QUALITY = 85
_EMBED_BATCH = 32
_INDEX_WORKERS = 8
_RETRIES = 2
_MAX_LYRIC_HINT = 200

_EMBED_DIM = 1024

_SYSTEM_PROMPT = (
    "你是音乐封面编辑。用一句不超过80字的中文描述这张图片："
    "主体景物、色调、整体氛围与意境，使其能与诗意歌名做匹配。"
    "不要出现'图片''照片'等字样，直接输出描述。"
)


class CoverIndexError(RuntimeError):
    """封面索引层所有可预期错误。"""


# ---------------- 增量扫描 ----------------

@dataclass(frozen=True)
class ScanDiff:
    """一次扫描的结果：待索引条目 + 磁盘上已消失的 name。"""
    to_index: tuple[CoverEntry, ...]
    missing_names: tuple[str, ...]

    @staticmethod
    def scan_folder(index: CoverIndex) -> "ScanDiff":
        folder = Path(index.folder)
        if not folder.is_dir():
            raise CoverIndexError(f"封面目录不存在: {folder}")

        to_index: list[CoverEntry] = []
        on_disk: set[str] = set()
        for path in sorted(folder.glob("*")):
            if not path.is_file() or path.suffix.lower() not in _IMG_EXTS:
                continue
            on_disk.add(path.name)
            old = index.entry(path.name)
            stat = path.stat()
            if (
                old is None
                or old.mtime_ns != stat.st_mtime_ns
                or old.size != stat.st_size
                or old.status == "failed"
            ):
                to_index.append(
                    CoverEntry(
                        name=path.name,
                        mtime_ns=stat.st_mtime_ns,
                        size=stat.st_size,
                        desc="",
                        status="ok",
                    )
                )

        missing = tuple(e.name for e in index.images if e.name not in on_disk)
        return ScanDiff(to_index=tuple(to_index), missing_names=missing)


# ---------------- 图片编码 ----------------

def _encode_image(path: Path) -> str:
    """打开图片、压缩长边并编码为 JPEG base64。"""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((_MAX_SIDE, _MAX_SIDE))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------- 意境描述 ----------------

DescribeProgress = Callable[[int, int, str, bool], None]
CancelCb = Callable[[], bool]


def _post_json(url: str, settings: Settings, payload: dict) -> dict:
    """统一的 POST + 状态码错误映射，返回 JSON body。"""
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.asr_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.timeout_sec,
        )
    except requests.RequestException as e:
        raise CoverIndexError(f"网络请求失败: {e}") from e

    if resp.status_code == 401:
        raise CoverIndexError("视觉模型 API Key 无效或已过期")
    if resp.status_code == 429:
        raise CoverIndexError("请求过于频繁")
    if resp.status_code >= 400:
        raise CoverIndexError(f"接口返回 {resp.status_code}: {resp.text[:200]}")

    try:
        return resp.json()
    except ValueError as e:
        raise CoverIndexError(f"响应非 JSON: {resp.text[:200]}") from e


def describe_image(
    path: Path,
    settings: Settings,
    model: str = DEFAULT_VL_MODEL,
) -> str:
    """调用视觉模型为单张封面生成一句意境描述。"""
    try:
        b64 = _encode_image(path)
    except OSError as e:
        raise CoverIndexError(f"图片读取失败: {e}") from e

    payload = {
        "model": model,
        "temperature": 0.3,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                ],
            },
        ],
    }
    body = _post_json(
        f"{settings.asr_base_url.rstrip('/')}/chat/completions", settings, payload
    )
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise CoverIndexError(f"响应结构异常: {str(body)[:200]}") from e

    text = str(content).strip()
    if not text:
        raise CoverIndexError("模型未返回描述")
    return text


def _describe_one(entry: CoverEntry, path: Path, settings: Settings) -> CoverEntry:
    """单张图尝试 _RETRIES+1 次；最终失败置 failed。"""
    for _ in range(_RETRIES + 1):
        try:
            desc = describe_image(path, settings)
            return replace(entry, desc=desc, status="ok")
        except CoverIndexError:
            pass
    return replace(entry, status="failed")


def describe_many(
    index: CoverIndex,
    entries: Sequence[CoverEntry],
    settings: Settings,
    progress: DescribeProgress | None = None,
    should_cancel: CancelCb | None = None,
    workers: int = _INDEX_WORKERS,
) -> tuple[CoverEntry, ...]:
    """并发为多张封面生成描述，保持输入顺序。"""
    total = len(entries)
    outcomes: list[CoverEntry | None] = [None] * total
    base = Path(index.folder)
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, entry in enumerate(entries):
            if should_cancel is not None and should_cancel():
                break
            futures[
                pool.submit(_describe_one, entry, base / entry.name, settings)
            ] = i

        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            outcomes[i] = result
            done += 1
            if progress is not None:
                progress(done, total, result.name, result.status == "ok" and bool(result.desc))

    filled: list[CoverEntry] = []
    for i, entry in enumerate(entries):
        result = outcomes[i]
        if result is not None:
            filled.append(result)
        elif entry.status == "ok" and not entry.desc:
            filled.append(replace(entry, status="failed"))
        else:
            filled.append(entry)
    return tuple(filled)


# ---------------- 文本嵌入 ----------------

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return np.asarray(mat / safe, dtype=np.float32)


def _embed_batch(
    batch: Sequence[str], settings: Settings, model: str
) -> np.ndarray:
    body = _post_json(
        f"{settings.asr_base_url.rstrip('/')}/embeddings",
        settings,
        {"model": model, "input": list(batch)},
    )
    try:
        items = body["data"]
        if not isinstance(items, list) or len(items) != len(batch):
            raise CoverIndexError(f"响应结构异常: {str(body)[:200]}")
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        vectors = [
            np.asarray(item["embedding"], dtype=np.float32) for item in ordered
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise CoverIndexError(f"响应结构异常: {str(body)[:200]}") from e

    shape = vectors[0].shape
    if len(shape) != 1 or any(v.shape != shape for v in vectors):
        raise CoverIndexError("嵌入向量维度不一致")
    return np.stack(vectors)


def embed_texts(
    texts: Sequence[str],
    settings: Settings,
    model: str = DEFAULT_EMBED_MODEL,
) -> np.ndarray:
    """分批调用嵌入接口，校验维度并做 L2 行归一化。"""
    if not texts:
        return np.zeros((0, _EMBED_DIM), dtype=np.float32)

    parts: list[np.ndarray] = []
    dim: int | None = None
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start:start + _EMBED_BATCH]
        vectors = _embed_batch(batch, settings, model)
        if dim is None:
            dim = vectors.shape[1]
        elif vectors.shape[1] != dim:
            raise CoverIndexError("各批嵌入向量维度不一致")
        parts.append(vectors)

    merged = np.concatenate(parts, axis=0)
    if merged.shape[1] != _EMBED_DIM:
        raise CoverIndexError(f"向量维度应为 {_EMBED_DIM}，实际 {merged.shape[1]}")
    return _l2_normalize(merged)


# ---------------- 增量索引主流程 ----------------

ProgressCb = Callable[[str, int, int, str], None]


def _load_old_rows(names: tuple[str, ...], prev_ok: tuple[str, ...]) -> dict[str, np.ndarray]:
    """先按当前 names 精确读取；不符则按上次 ok names 读取旧矩阵。"""
    old = load_embeddings(names)
    if old is None and prev_ok:
        old = load_embeddings(prev_ok)
    if old is None:
        return {}
    matrix, old_names = old
    return {name: matrix[i] for i, name in enumerate(old_names)}


def run_incremental(
    index: CoverIndex,
    settings: Settings,
    progress: ProgressCb | None = None,
    should_cancel: CancelCb | None = None,
) -> CoverIndex:
    """扫描增量 -> 描述新图 -> 复用旧向量/嵌入新描述 -> 落盘。"""
    def report(phase: str, done: int, total: int, name: str = "") -> None:
        if progress is not None:
            progress(phase, done, total, name)

    diff = ScanDiff.scan_folder(index)
    report("scan", 1, 1)
    prev_ok = tuple(e.name for e in index.ok_entries())

    index = index.mark_missing(diff.missing_names)

    def on_describe(done: int, total: int, name: str, ok: bool) -> None:
        report("describe", done, total, name)

    described = describe_many(
        index,
        diff.to_index,
        settings,
        progress=on_describe if progress is not None else None,
        should_cancel=should_cancel,
    )
    merged = index.with_entries(described)

    names = tuple(e.name for e in merged.ok_entries())
    stale = frozenset(e.name for e in diff.to_index)
    old_rows = _load_old_rows(names, prev_ok)

    new_names = tuple(n for n in names if n in stale or n not in old_rows)
    if new_names:
        report("embed", 0, len(new_names))
        desc_of = {e.name: e.desc for e in merged.images}
        new_vectors = embed_texts(
            [desc_of[name] for name in new_names], settings, merged.embed_model
        )
        report("embed", len(new_names), len(new_names))
    else:
        new_vectors = np.zeros((0, _EMBED_DIM), dtype=np.float32)

    matrix = np.zeros((len(names), _EMBED_DIM), dtype=np.float32)
    row_of_new = {name: i for i, name in enumerate(new_names)}
    for i, name in enumerate(names):
        if name in row_of_new:
            matrix[i] = new_vectors[row_of_new[name]]
        else:
            matrix[i] = old_rows[name]

    save_index(merged)
    save_embeddings(names, matrix)
    return merged

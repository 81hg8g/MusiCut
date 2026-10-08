"""封面意境匹配：歌名/依据/歌词意象向量与封面意境向量做余弦匹配。

纯 numpy 计算；批内顺序 claim，保证一张封面不被同批多首歌重复选用。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .cover_indexer import _MAX_LYRIC_HINT, embed_texts
from .cover_registry import UsedCovers
from .cover_store import CoverIndex
from .settings import Settings

_EXHAUSTED_MSG = "封面素材已用尽"


@dataclass(frozen=True)
class CoverQuery:
    """单曲匹配查询；key 由调用方传歌曲文件绝对路径，用于回填。"""
    key: str
    title: str
    reason: str = ""
    lyrics: str = ""


@dataclass(frozen=True)
class CoverMatch:
    """单曲匹配结果。"""
    key: str
    cover_name: str | None
    score: float
    error: str | None = None


def build_query_text(q: CoverQuery) -> str:
    """把歌名、命名依据与歌词意象拼成嵌入文本。"""
    text = f"歌名：{q.title}\n命名依据：{q.reason}"
    lyrics = q.lyrics.strip()
    if lyrics:
        text += f"\n歌词意象：{lyrics[:_MAX_LYRIC_HINT]}"
    return text


def _candidate_positions(index: CoverIndex, names: Sequence[str], used: UsedCovers):
    """names 中条目存在、status ok 且未登记使用过的行位置。"""
    folder = Path(index.folder)
    positions = []
    for i, name in enumerate(names):
        entry = index.entry(name)
        if entry is None or entry.status != "ok":
            continue
        if str(folder / name) in used.paths:
            continue
        positions.append(i)
    return positions


def match_all(
    index: CoverIndex,
    embeddings: np.ndarray,
    names: Sequence[str],
    used: UsedCovers,
    queries: Sequence[CoverQuery],
    settings: Settings,
) -> list[CoverMatch]:
    """按 queries 顺序逐一匹配封面，批内 claim 不重复。"""
    positions = _candidate_positions(index, names, used)
    if not positions:
        return [
            CoverMatch(q.key, None, 0.0, _EXHAUSTED_MSG) for q in queries
        ]

    query_vectors = embed_texts(
        [build_query_text(q) for q in queries], settings, index.embed_model
    )
    cover_vectors = np.asarray(embeddings)[np.asarray(positions)]
    scores = query_vectors @ cover_vectors.T   # 余弦相似度（均已归一化）

    claimed: set[int] = set()
    results: list[CoverMatch] = []
    for qi, q in enumerate(queries):
        available = [i for i in range(len(positions)) if i not in claimed]
        if not available:
            results.append(CoverMatch(q.key, None, 0.0, _EXHAUSTED_MSG))
            continue
        row = scores[qi]
        best = available[0]
        best_score = row[best]
        for i in available[1:]:
            if row[i] > best_score:
                best = i
                best_score = row[i]
        claimed.add(best)
        score = float(best_score)
        score = max(-1.0, min(1.0, score))   # 消除浮点越界
        results.append(CoverMatch(q.key, names[positions[best]], score, None))
    return results

"""切分点计算与异常标记"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .ffmpeg_ops import SilenceInterval


class TrackFlag(Enum):
    OK = "ok"
    TOO_SHORT = "too_short"      # 短曲目，可能误检
    TOO_LONG = "too_long"        # 长曲目，可能漏检


@dataclass(frozen=True)
class TrackSegment:
    index: int            # 在该源文件中的序号，从 1 开始
    start: float
    end: float
    flag: TrackFlag = TrackFlag.OK

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class FileAnalysis:
    src_path: Path
    total_duration: float
    segments: tuple[TrackSegment, ...] = field(default_factory=tuple)
    error: str | None = None


def build_segments(
    total_duration: float,
    silences: list[SilenceInterval],
    min_track_sec: float = 60.0,
    long_track_sec: float = 600.0,
) -> tuple[TrackSegment, ...]:
    """根据静音区间构造切分段。静音中点作为切分点。

    标记规则：
      - duration < min_track_sec  → TOO_SHORT（黄）
      - duration > long_track_sec → TOO_LONG （红）
      - 其他                      → OK
    """
    if total_duration <= 0:
        return tuple()

    # 收集切分点：0、每个静音中点、末尾
    cut_points: list[float] = [0.0]
    for s in silences:
        mid = s.midpoint
        # 切点不应太靠近已有切点
        if mid - cut_points[-1] >= 1.0:
            cut_points.append(mid)
    if total_duration - cut_points[-1] < 1.0:
        cut_points[-1] = total_duration
    else:
        cut_points.append(total_duration)

    segments: list[TrackSegment] = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        dur = end - start
        if dur <= 0:
            continue
        if dur < min_track_sec:
            flag = TrackFlag.TOO_SHORT
        elif dur > long_track_sec:
            flag = TrackFlag.TOO_LONG
        else:
            flag = TrackFlag.OK
        segments.append(TrackSegment(index=len(segments) + 1, start=start, end=end, flag=flag))
    return tuple(segments)


_INVALID_CHARS_RE = re.compile(r'[\\/:*?"<>|]+')


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """清理 Windows 文件名非法字符，并裁剪长度。"""
    cleaned = _INVALID_CHARS_RE.sub(" ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip(" .")
    return cleaned or "track"


def short_title_for_file(src_path: Path, max_len: int = 50) -> str:
    """从源文件名提取专辑短标题作为前缀。

    文件名形如：`【原创·Playlist】xxx｜yyy - 1.xxx (Av...,P1).mp3`
    提取 `【...】xxx` 部分。
    """
    stem = src_path.stem
    # 去掉 " - 1.xxx" 及其后所有
    stem = re.split(r"\s+-\s+\d+\.", stem, maxsplit=1)[0]
    # 去掉括号尾巴
    stem = re.sub(r"\(Av[\w,]+\)\s*$", "", stem).strip()
    # 截掉 "|" 之后内容
    for sep in ("｜", "|"):
        if sep in stem:
            stem = stem.split(sep, 1)[0].strip()
    return sanitize_filename(stem, max_len=max_len)

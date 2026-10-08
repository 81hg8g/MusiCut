"""批量起名流水线：读歌词 → 音频特征 → 调大模型 → 汇总结果"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .ai_namer import AiNamerError, NamingInput, suggest_title
from .audio_features import AudioFeatureError, analyze
from .lyrics_reader import read_lyrics
from .settings import Settings


@dataclass(frozen=True)
class NamingRecord:
    path: Path
    title: str = ""
    reason: str = ""
    lyrics_source: str = "none"
    features_desc: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.title)


ProgressCb = Callable[[int, int, NamingRecord], None]
CancelCb = Callable[[], bool]


def naming_one(settings: Settings, path: Path) -> NamingRecord:
    """为单个文件起名。歌词与特征解析失败不阻断，仅降级。"""
    lyrics = read_lyrics(path)
    features_desc = ""
    duration = 0.0
    try:
        feats = analyze(path)
        features_desc = feats.describe()
        duration = feats.duration_sec
    except AudioFeatureError:
        pass

    data = NamingInput(
        file_name=path.name,
        duration_sec=duration,
        lyrics=lyrics.text,
        lyrics_source=lyrics.source,
        features_desc=features_desc,
    )
    try:
        result = suggest_title(settings, data)
    except AiNamerError as e:
        return NamingRecord(
            path=path,
            lyrics_source=lyrics.source,
            features_desc=features_desc,
            error=str(e),
        )
    return NamingRecord(
        path=path,
        title=result.title,
        reason=result.reason,
        lyrics_source=lyrics.source,
        features_desc=features_desc,
    )


def run_batch(
    settings: Settings,
    files: Iterable[Path],
    progress: ProgressCb | None = None,
    should_cancel: CancelCb | None = None,
) -> list[NamingRecord]:
    """并发为多个文件起名，保持输入顺序返回。"""
    file_list = list(files)
    total = len(file_list)
    records: dict[Path, NamingRecord] = {}
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as pool:
        futures = {pool.submit(naming_one, settings, f): f for f in file_list}
        for future in as_completed(futures):
            path = futures[future]
            if should_cancel is not None and should_cancel():
                break
            try:
                record = future.result()
            except Exception as e:  # noqa: BLE001 - 单曲失败不应中断整批
                record = NamingRecord(path=path, error=f"未预期错误: {e}")
            records[path] = record
            done += 1
            if progress is not None:
                progress(done, total, record)

    return [records[f] for f in file_list if f in records]

"""批量起名流水线：ASR 语种判定 → 歌词/特征 → 大模型起名 → 全局去重"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .ai_namer import AiContentError, AiNamerError, NamingInput, suggest_title
from .asr_client import LANG_UNKNOWN, transcribe
from .audio_features import AudioFeatureError, analyze
from .lyrics_reader import read_lyrics
from .settings import Settings
from .title_registry import TitleRegistry, normalize

_MAX_ATTEMPTS = 3
_AVOID_LIMIT = 20

_CN_NUMERALS = ("II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")


@dataclass(frozen=True)
class NamingRecord:
    path: Path
    title: str = ""
    reason: str = ""
    lyrics_source: str = "none"
    detected_language: str = LANG_UNKNOWN
    features_desc: str = ""
    error: str | None = None
    attempts: int = 1
    asr_note: str = ""          # ASR 调用决策说明
    local_vocal: str = "未知"   # 本地人声判定：有 / 无 / 未知

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.title)


ProgressCb = Callable[[int, int, NamingRecord], None]
CancelCb = Callable[[], bool]


class _UsedTitles:
    """线程安全的已用歌名集合（含本次运行新增）。"""

    def __init__(self, initial_keys: frozenset[str]) -> None:
        self._lock = threading.Lock()
        self._keys: set[str] = set(initial_keys)
        self._claimed: list[str] = []

    def try_claim(self, title: str) -> bool:
        key = normalize(title)
        if not key:
            return False
        with self._lock:
            if key in self._keys:
                return False
            self._keys.add(key)
            self._claimed.append(title)
            return True

    def snapshot(self, limit: int = _AVOID_LIMIT) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._claimed[-limit:])


def _uniquify(base: str, used: _UsedTitles) -> str:
    for numeral in _CN_NUMERALS:
        candidate = f"{base} {numeral}"
        if used.try_claim(candidate):
            return candidate
    n = 2
    candidate = base
    while not used.try_claim(candidate):
        candidate = f"{base} {n}"
        n += 1
    return candidate


def _should_call_asr(
    settings: Settings, local, feats
) -> tuple[bool, str]:
    """决定是否调用 ASR，返回 (是否调用, 说明)。"""
    if not settings.asr_enabled:
        return False, "ASR 未启用"
    if not settings.asr_configured:
        return False, "未配置 ASR Key"
    if local.has_lyrics:
        return False, "已用本地歌词"
    if settings.vocal_filter_enabled and feats is not None and not feats.has_vocal:
        return False, "本地判定无人声"
    return True, ""


def _resolve_text(
    path: Path, settings: Settings, local, feats
) -> tuple[str, str, str, str]:
    """返回 (文本, 来源, 语种, ASR决策说明)。"""
    call_asr, note = _should_call_asr(settings, local, feats)
    transcript = transcribe(path, settings) if call_asr else None

    if local.has_lyrics:
        text, source = local.text, local.source
    elif transcript is not None and transcript.has_vocal:
        text, source = transcript.text, "asr"
    else:
        text, source = "", "none"

    if transcript is not None and transcript.ok:
        language = transcript.language
        if note == "":
            note = f"已调用({language})" if transcript.has_vocal else "已调用（未识别出人声）"
    elif text:
        from .asr_client import detect_language
        language = detect_language(text)
    else:
        language = LANG_UNKNOWN
    if transcript is not None and transcript.error:
        note = f"ASR 调用失败 - {transcript.error}"
    return text, source, language, note


def naming_one(
    settings: Settings,
    path: Path,
    used: _UsedTitles,
    registry: TitleRegistry,
) -> NamingRecord:
    """为单个文件起名：本地歌词/特征 → 按需 ASR → 模型（含重名重试）。"""
    local = read_lyrics(path)

    features_desc = ""
    duration = 0.0
    feats = None
    try:
        feats = analyze(path, vocal_threshold=settings.vocal_threshold)
        features_desc = feats.describe()
        duration = feats.duration_sec
    except AudioFeatureError:
        pass

    lyrics_text, lyrics_source, language, asr_note = _resolve_text(
        path, settings, local, feats
    )
    local_vocal = "未知" if feats is None else ("有" if feats.has_vocal else "无")

    data = NamingInput(
        file_name=path.name,
        duration_sec=duration,
        lyrics=lyrics_text,
        lyrics_source=lyrics_source,
        features_desc=features_desc,
        detected_language=language,
    )

    last_title, last_reason = "", ""
    last_content_error = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        avoid = tuple(registry.recent()) + used.snapshot()
        # 前 _MAX_ATTEMPTS-1 次保持高温 1.0；最后一次降温 0.3，
        # 规避 DeepSeek 高温下高频回声 {"type": "json_object"} 的问题
        temperature = 0.3 if attempt == _MAX_ATTEMPTS else 1.0
        try:
            result = suggest_title(
                settings, data, avoid=avoid, temperature=temperature
            )
        except AiContentError as e:
            # 模型输出畸形（瞬时错误）：记录后继续下一次尝试
            last_content_error = str(e)
            continue
        except AiNamerError as e:
            # 配置/网络等硬错误：不重试，立即失败
            return NamingRecord(
                path=path, lyrics_source=lyrics_source, detected_language=language,
                features_desc=features_desc, error=str(e), attempts=attempt,
                asr_note=asr_note, local_vocal=local_vocal,
            )
        last_title, last_reason = result.title, result.reason
        if used.try_claim(result.title):
            return NamingRecord(
                path=path, title=result.title, reason=result.reason,
                lyrics_source=lyrics_source, detected_language=language,
                features_desc=features_desc, attempts=attempt,
                asr_note=asr_note, local_vocal=local_vocal,
            )

    # 三次均为内容畸形 → 返回失败记录
    if last_content_error:
        return NamingRecord(
            path=path, lyrics_source=lyrics_source, detected_language=language,
            features_desc=features_desc, error=last_content_error,
            attempts=_MAX_ATTEMPTS,
            asr_note=asr_note, local_vocal=local_vocal,
        )

    # 多次重名 → 追加序号兜底，确保绝不重名
    final = _uniquify(last_title, used)
    return NamingRecord(
        path=path, title=final, reason=last_reason,
        lyrics_source=lyrics_source, detected_language=language,
        features_desc=features_desc, attempts=_MAX_ATTEMPTS,
        asr_note=asr_note, local_vocal=local_vocal,
    )


def run_batch(
    settings: Settings,
    files: Iterable[Path],
    registry: TitleRegistry,
    progress: ProgressCb | None = None,
    should_cancel: CancelCb | None = None,
) -> list[NamingRecord]:
    """并发起名，全局去重，保持输入顺序返回。"""
    file_list = list(files)
    total = len(file_list)
    records: dict[Path, NamingRecord] = {}
    used = _UsedTitles(registry.keys)
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as pool:
        futures = {
            pool.submit(naming_one, settings, f, used, registry): f for f in file_list
        }
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

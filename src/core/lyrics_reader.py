"""歌词读取：MP3 内嵌歌词标签 + 同目录 .lrc/.txt 旁挂文件"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_FFPROBE_TIMEOUT = 30
_SIDECAR_EXTS = (".lrc", ".LRC", ".txt", ".TXT")
_LRC_TIME_RE = re.compile(r"\[\d{1,2}:\d{1,2}(?:[.:]\d{1,3})?\]")
_LRC_META_RE = re.compile(r"^\[(ar|ti|al|by|offset|re|ve):.*\]$", re.IGNORECASE)
_LYRIC_KEY_HINT = ("lyric", "uslt", "sylt", "unsynced")


@dataclass(frozen=True)
class Lyrics:
    source: str   # 'embedded' | 'lrc' | 'txt' | 'none'
    text: str

    @property
    def has_lyrics(self) -> bool:
        return bool(self.text.strip())


_NO_LYRICS = Lyrics(source="none", text="")


def read_lyrics(mp3_path: Path) -> Lyrics:
    """按 内嵌标签 → .lrc → .txt 的顺序读取歌词，取第一个命中的。"""
    embedded = _read_embedded(mp3_path)
    if embedded.has_lyrics:
        return embedded

    for ext in _SIDECAR_EXTS:
        sidecar = mp3_path.with_suffix(ext)
        if sidecar.exists():
            text = _read_sidecar(sidecar)
            if text.strip():
                return Lyrics(source=ext.lstrip(".").lower(), text=text)
    return _NO_LYRICS


def _read_embedded(mp3_path: Path) -> Lyrics:
    """通过 ffprobe 读取 ID3 内嵌歌词标签。"""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format_tags",
                "-print_format", "json",
                str(mp3_path),
            ],
            capture_output=True,
            timeout=_FFPROBE_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return _NO_LYRICS
    if proc.returncode != 0:
        return _NO_LYRICS
    try:
        data = json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return _NO_LYRICS

    tags = (data.get("format") or {}).get("tags") or {}
    for key, value in tags.items():
        if any(hint in key.lower() for hint in _LYRIC_KEY_HINT):
            text = _strip_lrc_tags(str(value))
            if text.strip():
                return Lyrics(source="embedded", text=text)
    return _NO_LYRICS


def _read_sidecar(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "big5"):
        try:
            raw = path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue
        return _strip_lrc_tags(raw) if path.suffix.lower() == ".lrc" else raw
    return ""


def _strip_lrc_tags(text: str) -> str:
    """去掉 LRC 时间戳与元信息行，保留纯歌词文本。"""
    lines: list[str] = []
    for line in text.splitlines():
        if _LRC_META_RE.match(line.strip()):
            continue
        cleaned = _LRC_TIME_RE.sub("", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)

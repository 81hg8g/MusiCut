"""写入 ID3 元数据标签（标题等）"""
from __future__ import annotations

import subprocess
from pathlib import Path

_TAG_TIMEOUT = 120


class TagWriteError(RuntimeError):
    pass


def write_title(path: Path, title: str, timeout: int = _TAG_TIMEOUT) -> None:
    """将歌名写入 MP3 的 ID3 Title 标签（音质与封面保持不变）。

    实现：写出临时文件后替换原文件，避免 ffmpeg 读写同一路径导致的损坏。
    """
    if not title.strip():
        raise TagWriteError("标题为空")
    if path.suffix.lower() != ".mp3":
        raise TagWriteError(f"暂不支持写入的格式: {path.suffix}")

    tmp = path.with_name(f"{path.stem}.__tag__.mp3")
    if tmp.exists():
        tmp.unlink()

    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", str(path),
        "-map", "0",
        "-c", "copy",
        "-metadata", f"title={title}",
        "-id3v2_version", "3",
        "-write_id3v1", "0",
        str(tmp),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as e:
        _cleanup(tmp)
        raise TagWriteError(f"ffmpeg 调用失败: {e}") from e

    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        _cleanup(tmp)
        err = proc.stderr.decode("utf-8", "replace")[-200:]
        raise TagWriteError(f"写入标签失败: {err}")

    try:
        tmp.replace(path)
    except OSError as e:
        _cleanup(tmp)
        raise TagWriteError(f"替换原文件失败: {e}") from e


def read_title(path: Path, timeout: int = 30) -> str:
    """读取现有 Title 标签（用于校验）。"""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format_tags=title",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.decode("utf-8", "replace").strip()


def _cleanup(tmp: Path) -> None:
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass

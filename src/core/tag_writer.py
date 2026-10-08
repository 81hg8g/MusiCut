"""写入 ID3 元数据标签（标题，可选附带封面）"""
from __future__ import annotations

import subprocess
from pathlib import Path

_TAG_TIMEOUT = 120


class TagWriteError(RuntimeError):
    pass


def write_title(path: Path, title: str, timeout: int = _TAG_TIMEOUT) -> None:
    """将歌名写入 MP3 的 ID3 Title 标签（音质与封面保持不变）。

    兼容包装：内部统一走 write_tags。
    """
    return write_tags(path, title, cover=None, timeout=timeout)


def write_tags(
    path: Path,
    title: str,
    cover: Path | None = None,
    timeout: int = _TAG_TIMEOUT,
) -> None:
    """将标题（可选封面）一次写入 MP3 的 ID3 标签。

    实现：写出临时文件后替换原文件，避免 ffmpeg 读写同一路径导致的损坏。
    标题与封面由同一条 ffmpeg 命令写入，整体成功或整体失败。
    """
    if not title.strip():
        raise TagWriteError("标题为空")
    if path.suffix.lower() != ".mp3":
        raise TagWriteError(f"暂不支持写入的格式: {path.suffix}")
    if cover is not None and not cover.is_file():
        raise TagWriteError(f"封面文件不存在: {cover}")

    tmp = path.with_name(f"{path.stem}.__tag__.mp3")
    if tmp.exists():
        tmp.unlink()

    cmd = _build_command(path, title, cover)
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


def _build_command(path: Path, title: str, cover: Path | None) -> list[str]:
    """构造 ffmpeg 命令：无封面时与原命令完全一致。"""
    tmp = path.with_name(f"{path.stem}.__tag__.mp3")
    if cover is None:
        return [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-i", str(path),
            "-map", "0",
            "-c", "copy",
            "-metadata", f"title={title}",
            "-id3v2_version", "3",
            "-write_id3v1", "0",
            str(tmp),
        ]
    return [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", str(path), "-i", str(cover),
        "-map", "0:a", "-map", "1:v",
        "-c", "copy", "-c:v", "mjpeg",
        "-disposition:v", "attached_pic",
        "-metadata", f"title={title}",
        "-id3v2_version", "3",
        "-write_id3v1", "0",
        str(tmp),
    ]


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

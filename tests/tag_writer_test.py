"""tag_writer 单元测试：标题（可选封面）一次写入，不调用真实 ffmpeg。"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from src.core import tag_writer as tw
from src.core.tag_writer import TagWriteError, write_tags, write_title


# ---- 辅助 ----

class _FakeProc:
    """记录命令并按需生成输出文件的 subprocess.run 替身。"""

    def __init__(
        self,
        *,
        returncode: int = 0,
        write_output: bool = True,
        output: bytes = b"x",
        stderr: bytes = b"",
        raise_exc: BaseException | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self._returncode = returncode
        self._write_output = write_output
        self._output = output
        self._stderr = stderr
        self._raise_exc = raise_exc

    def __call__(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(cmd)
        if self._raise_exc is not None:
            raise self._raise_exc
        if self._write_output:
            Path(cmd[-1]).write_bytes(self._output)
        return subprocess.CompletedProcess(cmd, self._returncode, stderr=self._stderr)


@pytest.fixture
def mp3(tmp_path: Path) -> Path:
    """含中文与空格路径的假 MP3。"""
    path = tmp_path / "周杰伦 夜曲.mp3"
    path.write_bytes(b"FAKEAUDIO")
    return path


@pytest.fixture
def cover(tmp_path: Path) -> Path:
    path = tmp_path / "封面.jpg"
    path.write_bytes(b"FAKEIMAGE")
    return path


def _tmp_of(path: Path) -> Path:
    return path.with_name(f"{path.stem}.__tag__.mp3")


# ---- 无封面命令 ----

def test_no_cover_command_matches_current(mp3, monkeypatch):
    fake = _FakeProc()
    monkeypatch.setattr(tw.subprocess, "run", fake)

    write_tags(mp3, "夜的第七章")

    cmd = fake.calls[0]
    tmp = _tmp_of(mp3)
    assert cmd == [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", str(mp3),
        "-map", "0",
        "-c", "copy",
        "-metadata", "title=夜的第七章",
        "-id3v2_version", "3",
        "-write_id3v1", "0",
        str(tmp),
    ]
    assert "-disposition" not in cmd
    # 成功后临时文件替换原文件
    assert not tmp.exists()
    assert mp3.read_bytes() == b"x"


# ---- 有封面命令 ----

def test_with_cover_command(mp3, cover, monkeypatch):
    fake = _FakeProc()
    monkeypatch.setattr(tw.subprocess, "run", fake)

    write_tags(mp3, "夜的第七章", cover=cover)

    cmd = fake.calls[0]
    tmp = _tmp_of(mp3)
    assert cmd == [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", str(mp3), "-i", str(cover),
        "-map", "0:a", "-map", "1:v",
        "-c", "copy", "-c:v", "mjpeg",
        "-disposition:v", "attached_pic",
        "-metadata", "title=夜的第七章",
        "-id3v2_version", "3",
        "-write_id3v1", "0",
        str(tmp),
    ]


def test_with_cover_command_key_params(mp3, cover, monkeypatch):
    fake = _FakeProc()
    monkeypatch.setattr(tw.subprocess, "run", fake)

    write_tags(mp3, "标题", cover=cover)

    cmd = fake.calls[0]
    assert "-i" in cmd and str(cover) in cmd
    assert "-map" in cmd and "0:a" in cmd and "1:v" in cmd
    assert "-c:v" in cmd and "mjpeg" in cmd
    assert "-disposition:v" in cmd and "attached_pic" in cmd


# ---- 入参校验 ----

def test_empty_title_raises(mp3, monkeypatch):
    fake = _FakeProc()
    monkeypatch.setattr(tw.subprocess, "run", fake)

    with pytest.raises(TagWriteError, match="标题为空"):
        write_tags(mp3, "   ")
    assert fake.calls == []


def test_non_mp3_raises(tmp_path, monkeypatch):
    path = tmp_path / "a.flac"
    path.write_bytes(b"x")
    fake = _FakeProc()
    monkeypatch.setattr(tw.subprocess, "run", fake)

    with pytest.raises(TagWriteError, match="暂不支持写入的格式"):
        write_tags(path, "标题")
    assert fake.calls == []


def test_cover_not_found_raises(mp3, tmp_path, monkeypatch):
    fake = _FakeProc()
    monkeypatch.setattr(tw.subprocess, "run", fake)
    missing = tmp_path / "不存在.jpg"

    with pytest.raises(
        TagWriteError,
        match=re.escape(f"封面文件不存在: {missing}"),
    ):
        write_tags(mp3, "标题", cover=missing)
    assert fake.calls == []


# ---- ffmpeg 失败路径：临时文件必须清理 ----

def test_returncode_nonzero_raises_and_cleans(mp3, monkeypatch):
    fake = _FakeProc(returncode=1, stderr=b"boom error")
    monkeypatch.setattr(tw.subprocess, "run", fake)

    with pytest.raises(TagWriteError, match="写入标签失败"):
        write_tags(mp3, "标题")
    assert not _tmp_of(mp3).exists()


def test_tmp_zero_size_raises_and_cleans(mp3, monkeypatch):
    fake = _FakeProc(output=b"")
    monkeypatch.setattr(tw.subprocess, "run", fake)

    with pytest.raises(TagWriteError, match="写入标签失败"):
        write_tags(mp3, "标题")
    assert not _tmp_of(mp3).exists()


def test_tmp_missing_raises(mp3, monkeypatch):
    fake = _FakeProc(write_output=False)
    monkeypatch.setattr(tw.subprocess, "run", fake)

    with pytest.raises(TagWriteError, match="写入标签失败"):
        write_tags(mp3, "标题")
    assert not _tmp_of(mp3).exists()


def test_subprocess_oserror_raises_and_cleans(mp3, monkeypatch):
    fake = _FakeProc(raise_exc=OSError("not found"))
    monkeypatch.setattr(tw.subprocess, "run", fake)

    with pytest.raises(TagWriteError, match="ffmpeg 调用失败"):
        write_tags(mp3, "标题")
    assert not _tmp_of(mp3).exists()


def test_subprocess_timeout_raises_and_cleans(mp3, monkeypatch):
    timeout_exc = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=1)
    fake = _FakeProc(raise_exc=timeout_exc)
    monkeypatch.setattr(tw.subprocess, "run", fake)

    with pytest.raises(TagWriteError, match="ffmpeg 调用失败"):
        write_tags(mp3, "标题")
    assert not _tmp_of(mp3).exists()


def test_replace_failure_raises_and_cleans(mp3, monkeypatch):
    fake = _FakeProc()
    monkeypatch.setattr(tw.subprocess, "run", fake)

    def fake_replace(self, target):
        raise OSError("拒绝访问")

    monkeypatch.setattr(Path, "replace", fake_replace)

    with pytest.raises(TagWriteError, match="替换原文件失败"):
        write_tags(mp3, "标题")
    assert not _tmp_of(mp3).exists()


# ---- 兼容包装 ----

def test_write_title_still_works(mp3, monkeypatch):
    fake = _FakeProc()
    monkeypatch.setattr(tw.subprocess, "run", fake)

    write_title(mp3, "老式标题")

    cmd = fake.calls[0]
    assert cmd[cmd.index("-metadata") + 1] == "title=老式标题"
    assert "-i" in cmd and str(mp3) in cmd
    assert mp3.read_bytes() == b"x"

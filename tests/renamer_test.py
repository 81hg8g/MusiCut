"""renamer 单元测试：封面透传、标题+封面一次写入、日志标签与回滚。"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.core import renamer as rn
from src.core.renamer import (
    RenamePlan,
    RenameResult,
    _tag_label,
    execute_renames,
    plan_renames,
    rollback_from_log,
    write_log,
)
from src.core.tag_writer import TagWriteError


# ---- 辅助 ----

def _mp3(directory: Path, name: str = "周杰伦 夜曲.mp3") -> Path:
    path = directory / name
    path.write_bytes(b"FAKEAUDIO")
    return path


def _cover(directory: Path) -> Path:
    path = directory / "封面 图.jpg"
    path.write_bytes(b"FAKEIMAGE")
    return path


def _result(
    *,
    tagged: bool = False,
    tag_error: str | None = None,
    covered: bool = False,
    cover_error: str | None = None,
) -> RenameResult:
    src = Path("a.mp3")
    return RenameResult(
        src=src, dst=Path("b.mp3"), success=True,
        tagged=tagged, tag_error=tag_error,
        covered=covered, cover_error=cover_error,
    )


# ---- plan_renames ----

def test_plan_passes_cover_through(tmp_path):
    src = _mp3(tmp_path)
    cover = _cover(tmp_path)

    plans = plan_renames([(src, "新 标题", cover)])

    assert len(plans) == 1
    assert plans[0].cover == cover
    assert plans[0].new_stem == "新 标题"
    assert plans[0].dst == tmp_path / "新 标题.mp3"


def test_plan_cover_defaults_none(tmp_path):
    src = _mp3(tmp_path)

    plans = plan_renames([(src, "标题", None)])

    assert plans[0].cover is None


def test_plan_empty_items():
    assert plan_renames([]) == []


def test_plan_dedupe_unchanged(tmp_path):
    src1 = _mp3(tmp_path, "a.mp3")
    src2 = _mp3(tmp_path, "b.mp3")

    plans = plan_renames([(src1, "同名", None), (src2, "同名", None)])

    assert plans[0].new_stem == "同名"
    assert plans[1].new_stem == "同名_2"
    assert plans[1].cover is None


# ---- execute_renames：成功路径透传 cover ----

def test_execute_success_passes_cover_to_write_tags(tmp_path, monkeypatch):
    src = _mp3(tmp_path)
    cover = _cover(tmp_path)
    plans = plan_renames([(src, "夜的第七章", cover)])
    calls: list[tuple] = []

    def fake_write_tags(path, title, cover_arg):
        calls.append((path, title, cover_arg))

    monkeypatch.setattr(rn, "write_tags", fake_write_tags)

    results = execute_renames(plans, write_metadata=True)

    assert len(calls) == 1
    called_path, called_title, called_cover = calls[0]
    assert called_path == src
    assert called_title == "夜的第七章"
    assert called_cover == cover

    r = results[0]
    assert r.success and r.tagged and r.covered
    assert r.tag_error is None and r.cover_error is None
    assert r.dst.exists()
    assert not src.exists()


def test_execute_without_metadata_does_not_call_write_tags(tmp_path, monkeypatch):
    src = _mp3(tmp_path)
    cover = _cover(tmp_path)
    plans = plan_renames([(src, "标题", cover)])
    calls: list[tuple] = []
    monkeypatch.setattr(rn, "write_tags", lambda *a: calls.append(a))

    results = execute_renames(plans, write_metadata=False)

    assert calls == []
    r = results[0]
    assert r.success
    assert not r.tagged and not r.covered
    assert r.tag_error is None and r.cover_error is None


# ---- execute_renames：标题/封面整体失败 ----

def test_execute_tag_and_cover_both_fail(tmp_path, monkeypatch):
    src = _mp3(tmp_path)
    cover = _cover(tmp_path)
    plans = plan_renames([(src, "标题", cover)])

    def fail(path, title, cover_arg):
        raise TagWriteError("ffmpeg 爆炸了")

    monkeypatch.setattr(rn, "write_tags", fail)

    results = execute_renames(plans, write_metadata=True)

    r = results[0]
    assert r.success  # 标签失败不阻断重命名（沿用现状）
    assert not r.tagged and not r.covered
    assert r.tag_error == "ffmpeg 爆炸了"
    assert r.cover_error == "ffmpeg 爆炸了"
    assert r.dst.exists()


def test_execute_title_only_fail_has_no_cover_error(tmp_path, monkeypatch):
    src = _mp3(tmp_path)
    plans = plan_renames([(src, "标题", None)])

    def fail(path, title, cover_arg):
        raise TagWriteError("boom")

    monkeypatch.setattr(rn, "write_tags", fail)

    r = execute_renames(plans, write_metadata=True)[0]

    assert not r.tagged and not r.covered
    assert r.tag_error == "boom"
    assert r.cover_error is None


# ---- 重命名失败但标签已写入 ----

def test_rename_fails_but_tag_written(tmp_path, monkeypatch):
    src = _mp3(tmp_path)
    cover = _cover(tmp_path)
    plans = plan_renames([(src, "新名字", cover)])
    monkeypatch.setattr(rn, "write_tags", lambda *a: None)

    def fake_rename(self, target):
        raise OSError("拒绝访问")

    monkeypatch.setattr(Path, "rename", fake_rename)

    r = execute_renames(plans, write_metadata=True)[0]

    assert not r.success
    assert r.error.startswith("重命名失败")
    assert r.tagged and r.covered
    assert r.tag_error is None and r.cover_error is None


# ---- _tag_label 纯函数 ----

def test_tag_label_title_success_only():
    assert _tag_label(_result(tagged=True), has_cover=False) == "标题已写入"


def test_tag_label_title_failure():
    label = _tag_label(_result(tag_error="ffmpeg 失败了"), has_cover=False)
    assert label == "标题失败(ffmpeg 失败了)"


def test_tag_label_title_and_cover_success():
    label = _tag_label(_result(tagged=True, covered=True), has_cover=True)
    assert label == "标题已写入；封面已写入"


def test_tag_label_title_and_cover_failure():
    label = _tag_label(
        _result(tag_error="ffmpeg 炸了", cover_error="ffmpeg 炸了"),
        has_cover=True,
    )
    assert label == "标题失败(ffmpeg 炸了)；封面失败(ffmpeg 炸了)"


def test_tag_label_not_run_is_dash():
    assert _tag_label(_result(), has_cover=False) == "-"


def test_tag_label_truncates_to_30_chars():
    long_msg = "非常长的错误信息" * 10
    label = _tag_label(_result(tag_error=long_msg), has_cover=False)
    assert label == f"标题失败({long_msg[:30]})"


# ---- write_log 内容 ----

def test_write_log_contains_new_status_texts(tmp_path, monkeypatch):
    ok_src = _mp3(tmp_path, "ok.mp3")
    bad_src = _mp3(tmp_path, "bad.mp3")
    cover = _cover(tmp_path)
    plans = plan_renames([
        (ok_src, "成功曲", cover),
        (bad_src, "失败曲", cover),
    ])

    def fake_write_tags(path, title, cover_arg):
        if path.name == "bad.mp3":
            raise TagWriteError("ffmpeg 炸了")

    monkeypatch.setattr(rn, "write_tags", fake_write_tags)
    log_path = tmp_path / "log.csv"

    execute_renames(plans, log_path=log_path, write_metadata=True)
    text = log_path.read_text(encoding="utf-8-sig")

    assert "标题已写入" in text
    assert "封面已写入" in text
    assert "标题失败(ffmpeg 炸了)" in text
    assert "封面失败(ffmpeg 炸了)" in text

    rows = list(csv.reader(text.splitlines()))
    assert rows[0] == ["时间", "目录", "原文件名", "新文件名", "标签", "结果", "说明"]
    assert all(len(row) == 7 for row in rows)


def test_write_log_dash_when_metadata_disabled(tmp_path, monkeypatch):
    src = _mp3(tmp_path)
    cover = _cover(tmp_path)
    plans = plan_renames([(src, "标题", cover)])
    monkeypatch.setattr(rn, "write_tags", lambda *a: None)
    log_path = tmp_path / "log.csv"

    execute_renames(plans, log_path=log_path, write_metadata=False)

    text = log_path.read_text(encoding="utf-8-sig")
    assert "-" in text.splitlines()[1].split(",")


# ---- 回滚冒烟（行为不变） ----

def test_rollback_roundtrip(tmp_path, monkeypatch):
    src = _mp3(tmp_path)
    plans = plan_renames([(src, "新名字", None)])
    monkeypatch.setattr(rn, "write_tags", lambda *a: None)
    log_path = tmp_path / "log.csv"

    execute_renames(plans, log_path=log_path, write_metadata=True)
    assert not src.exists()

    ok, fail = rollback_from_log(log_path)
    assert (ok, fail) == (1, 0)
    assert src.exists()


def test_rollback_missing_log_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rollback_from_log(tmp_path / "nope.csv")


def test_rollback_empty_log(tmp_path):
    log_path = tmp_path / "log.csv"
    write_log(log_path, [])
    assert rollback_from_log(log_path) == (0, 0)


# ---- 边界：空 items 执行 ----

def test_execute_empty_plans_writes_header_only(tmp_path):
    log_path = tmp_path / "log.csv"
    results = execute_renames([], log_path=log_path)
    assert results == []
    rows = list(csv.reader(log_path.read_text(encoding="utf-8-sig").splitlines()))
    assert len(rows) == 1

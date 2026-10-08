"""封面 GUI 集成测试：纯函数计数/映射 + SettingsDialog 勾选与目录回填。

全程不联网、不写真实配置；使用 offscreen 平台插件，直接构造 QApplication 单例。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from src.core.cover_matcher import CoverMatch
from src.core.cover_registry import UsedCovers
from src.core.cover_store import CoverEntry, CoverIndex
from src.core.rename_pipeline import NamingRecord
from src.core.renamer import RenamePlan, RenameResult
from src.core.settings import Settings
from src.gui.rename_window import (
    cover_summary,
    plan_items,
    resolve_do_cover,
    used_paths_from_results,
)
from src.gui.settings_dialog import SettingsDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _entry(name: str, *, status: str = "ok", desc: str = "d") -> CoverEntry:
    return CoverEntry(name=name, mtime_ns=1, size=1, desc=desc, status=status)


def _record(path: Path, title: str = "歌名", reason: str = "依据") -> NamingRecord:
    return NamingRecord(path=path, title=title, reason=reason)


# ---------- resolve_do_cover ----------

def test_resolve_do_cover_requires_both_enabled():
    assert resolve_do_cover(True, True) is True
    assert resolve_do_cover(True, False) is False
    assert resolve_do_cover(False, True) is False
    assert resolve_do_cover(False, False) is False


# ---------- cover_summary ----------

def test_cover_summary_normal_counts(tmp_path):
    folder = tmp_path / "covers"
    index = CoverIndex(
        folder=str(folder),
        images=(
            _entry("a.jpg"),
            _entry("b.jpg"),
            _entry("d.jpg", status="failed"),
            _entry("c.jpg", status="missing"),
        ),
    )
    used = UsedCovers(paths=frozenset({str(folder / "a.jpg")}))

    assert cover_summary(index, used) == (3, 1, 1)


def test_cover_summary_empty_index(tmp_path):
    index = CoverIndex(folder=str(tmp_path / "nope"))
    assert cover_summary(index, UsedCovers()) == (0, 0, 0)


def test_cover_summary_used_in_other_dir_not_counted(tmp_path):
    folder = tmp_path / "covers"
    index = CoverIndex(
        folder=str(folder),
        images=(_entry("a.jpg"), _entry("b.jpg")),
    )
    used = UsedCovers(paths=frozenset({str(tmp_path / "other" / "z.jpg")}))

    assert cover_summary(index, used) == (2, 0, 2)


# ---------- plan_items ----------

def test_plan_items_order_and_mapping(tmp_path):
    folder = tmp_path / "covers"
    r1 = _record(tmp_path / "1.mp3", "曲一")
    r2 = _record(tmp_path / "2.mp3", "曲二")
    r3 = _record(tmp_path / "3.mp3", "曲三")
    matches = {
        str(r1.path): CoverMatch(str(r1.path), "a.jpg", 0.9),
        str(r2.path): CoverMatch(str(r2.path), "b.jpg", 0.8),
    }

    items = plan_items([r1, r2, r3], matches, str(folder))

    assert items == [
        (r1.path, "曲一", folder / "a.jpg"),
        (r2.path, "曲二", folder / "b.jpg"),
        (r3.path, "曲三", None),
    ]


def test_plan_items_exhausted_match_is_none(tmp_path):
    folder = tmp_path / "covers"
    record = _record(tmp_path / "1.mp3")
    matches = {
        str(record.path): CoverMatch(
            str(record.path), None, 0.0, "封面素材已用尽"
        )
    }

    items = plan_items([record], matches, str(folder))

    assert items == [(record.path, "歌名", None)]


def test_plan_items_same_cover_name_reuse(tmp_path):
    folder = tmp_path / "covers"
    r1 = _record(tmp_path / "1.mp3")
    r2 = _record(tmp_path / "2.mp3")
    matches = {
        str(r1.path): CoverMatch(str(r1.path), "a.jpg", 0.9),
        str(r2.path): CoverMatch(str(r2.path), "a.jpg", 0.8),
    }

    items = plan_items([r1, r2], matches, str(folder))

    assert [t[2] for t in items] == [folder / "a.jpg", folder / "a.jpg"]


# ---------- used_paths_from_results ----------

def _plan(cover: Path | None) -> RenamePlan:
    return RenamePlan(src=Path("x.mp3"), new_stem="x", cover=cover)


def _result(*, covered: bool) -> RenameResult:
    return RenameResult(
        src=Path("x.mp3"),
        dst=Path("y.mp3"),
        success=True,
        covered=covered,
    )


def test_used_paths_only_covered_ordered_and_deduped(tmp_path):
    c1 = (tmp_path / "covers" / "a.jpg").absolute()
    c2 = (tmp_path / "covers" / "b.jpg").absolute()
    plans = [_plan(c1), _plan(c2), _plan(c1)]
    results = [_result(covered=True), _result(covered=False), _result(covered=True)]

    assert used_paths_from_results(results, plans) == (str(c1),)


def test_used_paths_none_covered(tmp_path):
    cover = (tmp_path / "covers" / "a.jpg").absolute()
    plans = [_plan(cover), _plan(None)]
    results = [_result(covered=False), _result(covered=False)]

    assert used_paths_from_results(results, plans) == ()


# ---------- SettingsDialog ----------

def test_settings_dialog_check_and_dir_collect(qapp):
    dialog = SettingsDialog(Settings(cover_enabled=False, cover_dir=""))
    try:
        assert dialog.chk_cover.isChecked() is False

        dialog.chk_cover.setChecked(True)
        dialog.edit_cover_dir.setText(r"D:\my covers")
        collected = dialog._collect()

        assert collected.cover_enabled is True
        assert collected.cover_dir == r"D:\my covers"
    finally:
        dialog.close()


def test_settings_dialog_initial_checked_state(qapp):
    dialog = SettingsDialog(Settings(cover_enabled=True, cover_dir=r"K:\c"))
    try:
        assert dialog.chk_cover.isChecked() is True
        assert dialog.edit_cover_dir.text() == r"K:\c"
    finally:
        dialog.close()


def test_settings_dialog_blank_dir_falls_back_to_default(qapp):
    dialog = SettingsDialog(Settings(cover_enabled=True))
    try:
        dialog.edit_cover_dir.setText("   ")
        collected = dialog._collect()

        assert collected.cover_dir == r"H:\图\AlbumCover"
    finally:
        dialog.close()

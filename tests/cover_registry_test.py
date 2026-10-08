"""cover_registry 单元测试：已使用封面图片记录。"""
from __future__ import annotations

import json

import pytest

from src.core import cover_registry as cr
from src.core.cover_registry import (
    UsedCovers,
    load_used,
    register_used,
    save_used,
    used_path,
)


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """把配置目录打桩到临时目录。"""
    monkeypatch.setattr(cr, "settings_dir", lambda: tmp_path)
    return tmp_path


def test_used_path(tmp_settings):
    assert used_path() == tmp_settings / "used_covers.json"


def test_default_paths_is_frozenset():
    used = UsedCovers()
    assert used.paths == frozenset()
    assert isinstance(used.paths, frozenset)


def test_used_covers_is_frozen():
    with pytest.raises(Exception):
        UsedCovers().paths = frozenset(("x",))  # type: ignore[misc]


def test_load_used_missing_returns_empty(tmp_settings):
    assert load_used() == UsedCovers()


def test_save_and_load_roundtrip(tmp_settings):
    used = UsedCovers(paths=frozenset((r"H:\图\a 封面.jpg", r"H:\图\b.jpg")))
    save_used(used)
    assert load_used() == used


def test_load_used_corrupted_returns_empty(tmp_settings):
    used_path().write_text("{ 坏 json", encoding="utf-8")
    assert load_used() == UsedCovers()


def test_load_used_bad_payload_returns_empty(tmp_settings):
    used_path().write_text(json.dumps({"paths": "不是列表"}), encoding="utf-8")
    assert load_used() == UsedCovers()


def test_saved_json_paths_sorted(tmp_settings):
    save_used(UsedCovers(paths=frozenset(("c.jpg", "a.jpg", "b.jpg"))))
    raw = json.loads(used_path().read_text(encoding="utf-8"))
    assert raw == {"paths": ["a.jpg", "b.jpg", "c.jpg"]}


def test_register_used_union_and_dedup(tmp_settings):
    used = UsedCovers(paths=frozenset(("a.jpg", "b.jpg")))
    updated = register_used(used, ("b.jpg", "c 图.jpg"))
    assert updated.paths == frozenset(("a.jpg", "b.jpg", "c 图.jpg"))


def test_register_used_persists(tmp_settings):
    register_used(UsedCovers(), ("a.jpg",))
    assert load_used().paths == frozenset(("a.jpg",))


def test_register_used_empty_tuple(tmp_settings):
    used = UsedCovers(paths=frozenset(("a.jpg",)))
    updated = register_used(used, ())
    assert updated == used
    assert used_path().exists()


def test_register_used_returns_new_instance(tmp_settings):
    used = UsedCovers(paths=frozenset(("a.jpg",)))
    updated = register_used(used, ("b.jpg",))
    assert updated is not used
    assert used.paths == frozenset(("a.jpg",))

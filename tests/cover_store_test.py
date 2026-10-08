"""cover_store 单元测试：封面意境索引与向量矩阵的持久化。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.core import cover_store as cs
from src.core.cover_store import (
    CoverEntry,
    CoverIndex,
    index_paths,
    load_embeddings,
    load_index,
    save_embeddings,
    save_index,
)

FOLDER = r"H:\图\AlbumCover"


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """把配置目录打桩到临时目录。"""
    monkeypatch.setattr(cs, "settings_dir", lambda: tmp_path)
    return tmp_path


def _entry(name: str, desc: str = "", status: str = "ok") -> CoverEntry:
    return CoverEntry(name=name, mtime_ns=123, size=456, desc=desc, status=status)


# ---- 路径与常量 ----

def test_index_paths_uses_settings_dir(tmp_settings):
    index_path, emb_path = index_paths()
    assert index_path == tmp_settings / "cover_index.json"
    assert emb_path == tmp_settings / "cover_embeddings.npz"


def test_default_models_constants():
    assert cs.DEFAULT_VL_MODEL == "Qwen/Qwen3-VL-8B-Instruct"
    assert cs.DEFAULT_EMBED_MODEL == "BAAI/bge-m3"


# ---- CoverEntry ----

def test_entry_defaults():
    e = CoverEntry(name="a.jpg", mtime_ns=1, size=2)
    assert e.desc == ""
    assert e.status == "ok"


def test_entry_is_frozen():
    e = _entry("a.jpg")
    with pytest.raises(Exception):
        e.status = "failed"  # type: ignore[misc]


# ---- CoverIndex.entry / ok_entries ----

def test_entry_lookup_found_and_missing():
    index = CoverIndex(folder=FOLDER, images=(_entry("a.jpg"), _entry("b.jpg")))
    assert index.entry("a.jpg").name == "a.jpg"
    assert index.entry("nope.jpg") is None


def test_ok_entries_requires_ok_status_and_desc():
    index = CoverIndex(
        folder=FOLDER,
        images=(
            _entry("a.jpg", desc="夜色"),
            _entry("b.jpg", desc=""),          # ok 但无描述
            _entry("c.jpg", desc="雨", status="failed"),
            _entry("d.jpg", desc="晴", status="missing"),
        ),
    )
    names = tuple(e.name for e in index.ok_entries())
    assert names == ("a.jpg",)


# ---- CoverIndex.with_entries ----

def test_with_entries_replaces_keeping_original_order():
    index = CoverIndex(
        folder=FOLDER,
        images=(_entry("a.jpg", "旧"), _entry("b.jpg", "原")),
    )
    updated = index.with_entries((_entry("b.jpg", "新"),))
    assert tuple(e.name for e in updated.images) == ("a.jpg", "b.jpg")
    assert updated.entry("b.jpg").desc == "新"
    assert updated.entry("a.jpg").desc == "旧"


def test_with_entries_appends_new_after_originals():
    index = CoverIndex(folder=FOLDER, images=(_entry("a.jpg"),))
    updated = index.with_entries((_entry("b 图.jpg"), _entry("c.jpg")))
    assert tuple(e.name for e in updated.images) == ("a.jpg", "b 图.jpg", "c.jpg")


def test_with_entries_empty_tuple_preserves_all():
    index = CoverIndex(folder=FOLDER, images=(_entry("a.jpg"), _entry("b.jpg")))
    updated = index.with_entries(())
    assert updated == index


def test_with_entries_duplicate_name_in_new_keeps_last_once():
    index = CoverIndex(folder=FOLDER, images=(_entry("a.jpg"),))
    updated = index.with_entries(
        (_entry("b.jpg", "第一版"), _entry("b.jpg", "第二版"))
    )
    assert tuple(e.name for e in updated.images) == ("a.jpg", "b.jpg")
    assert updated.entry("b.jpg").desc == "第二版"


def test_with_entries_returns_new_instance():
    index = CoverIndex(folder=FOLDER, images=(_entry("a.jpg"),))
    updated = index.with_entries((_entry("a.jpg", "新"),))
    assert updated is not index
    assert index.entry("a.jpg").desc == ""


# ---- CoverIndex.mark_missing ----

def test_mark_missing_only_targets_matching_names():
    index = CoverIndex(
        folder=FOLDER,
        images=(_entry("a.jpg"), _entry("b.jpg"), _entry("c.jpg")),
    )
    updated = index.mark_missing(("a.jpg", "c.jpg"))
    assert updated.entry("a.jpg").status == "missing"
    assert updated.entry("b.jpg").status == "ok"
    assert updated.entry("c.jpg").status == "missing"


def test_mark_missing_ignores_unknown_names():
    index = CoverIndex(folder=FOLDER, images=(_entry("a.jpg"),))
    updated = index.mark_missing(("x.jpg",))
    assert tuple(e.name for e in updated.images) == ("a.jpg",)
    assert updated.entry("a.jpg").status == "ok"


def test_mark_missing_empty_tuple():
    index = CoverIndex(folder=FOLDER, images=(_entry("a.jpg"),))
    assert index.mark_missing(()) == index


# ---- load_index / save_index ----

def test_load_index_missing_file_returns_empty(tmp_settings):
    index = load_index(FOLDER)
    assert index == CoverIndex(folder=FOLDER)
    assert index.images == ()
    assert index.vl_model == cs.DEFAULT_VL_MODEL
    assert index.embed_model == cs.DEFAULT_EMBED_MODEL


def test_load_index_corrupted_returns_empty(tmp_settings):
    index_paths()[0].write_text("{ 坏 json", encoding="utf-8")
    assert load_index(FOLDER) == CoverIndex(folder=FOLDER)


def test_load_index_folder_mismatch_returns_empty(tmp_settings):
    save_index(CoverIndex(folder=r"E:\其他目录", images=(_entry("a.jpg"),)))
    assert load_index(FOLDER) == CoverIndex(folder=FOLDER)


def test_load_index_folder_case_and_separator_normalized(tmp_settings):
    saved = CoverIndex(folder="C:/Covers Dir", images=(_entry("a 图.jpg", "夜"),))
    save_index(saved)
    loaded = load_index(r"c:\covers dir")
    assert loaded.folder == "C:/Covers Dir"
    assert loaded.images == saved.images


def test_save_index_creates_dir_and_roundtrips(tmp_path, monkeypatch):
    nested = tmp_path / "sub" / "cfg"
    monkeypatch.setattr(cs, "settings_dir", lambda: nested)
    saved = CoverIndex(
        folder=FOLDER,
        images=(
            _entry("a 图.jpg", "夜色 温柔"),
            _entry("b.jpg", "", status="failed"),
            _entry("gone.jpg", status="missing"),
        ),
    )
    save_index(saved)
    loaded = load_index(FOLDER)
    assert loaded == saved
    assert not index_paths()[0].with_suffix(".json.tmp").exists()


def test_save_index_json_structure(tmp_settings):
    save_index(CoverIndex(folder=FOLDER, images=(_entry("a.jpg", "夜"),)))
    raw = json.loads(index_paths()[0].read_text(encoding="utf-8"))
    assert set(raw.keys()) == {"folder", "vl_model", "embed_model", "images"}
    assert raw["folder"] == FOLDER
    assert set(raw["images"][0].keys()) == {
        "name", "mtime_ns", "size", "desc", "status",
    }


# ---- load_embeddings / save_embeddings ----

def test_embeddings_roundtrip(tmp_settings):
    names = ("a.jpg", "b 图.jpg")
    vectors = np.random.default_rng(0).standard_normal((2, 1024))
    save_embeddings(names, vectors)
    result = load_embeddings(names)
    assert result is not None
    embeddings, loaded_names = result
    assert loaded_names == names
    assert embeddings.shape == (2, 1024)
    np.testing.assert_array_equal(embeddings, vectors)


def test_embeddings_empty_names_roundtrip(tmp_settings):
    vectors = np.zeros((0, 1024))
    save_embeddings((), vectors)
    result = load_embeddings(())
    assert result is not None
    embeddings, names = result
    assert names == ()
    assert embeddings.shape == (0, 1024)


def test_load_embeddings_missing_file(tmp_settings):
    assert load_embeddings(("a.jpg",)) is None


def test_load_embeddings_wrong_dim(tmp_settings):
    _, path = index_paths()
    np.savez(
        path,
        names=np.array(["a.jpg", "b.jpg"], dtype=object),
        embeddings=np.zeros((2, 512)),
    )
    assert load_embeddings(("a.jpg", "b.jpg")) is None


def test_load_embeddings_row_count_mismatch(tmp_settings):
    _, path = index_paths()
    np.savez(
        path,
        names=np.array(["a.jpg", "b.jpg"], dtype=object),
        embeddings=np.zeros((3, 1024)),
    )
    assert load_embeddings(("a.jpg", "b.jpg")) is None


def test_load_embeddings_names_mismatch(tmp_settings):
    save_embeddings(("a.jpg", "b.jpg"), np.zeros((2, 1024)))
    assert load_embeddings(("a.jpg", "c.jpg")) is None


def test_load_embeddings_corrupted_file(tmp_settings):
    index_paths()[1].write_text("not a npz file", encoding="utf-8")
    assert load_embeddings(("a.jpg",)) is None

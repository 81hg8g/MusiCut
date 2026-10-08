"""cover_indexer 单元测试：封面增量扫描、意境描述、向量嵌入与增量索引。

全程不联网：requests.post 一律打桩为 FakeApi。
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from src.core import cover_indexer as ci
from src.core import cover_store as cs
from src.core.cover_indexer import (
    CoverIndexError,
    ScanDiff,
    _encode_image,
    describe_image,
    describe_many,
    embed_texts,
    run_incremental,
)
from src.core.cover_store import CoverEntry, CoverIndex, load_embeddings
from src.core.settings import Settings

DIM = 1024


# ---------- 测试辅助 ----------

def make_image(path: Path, color, size=(64, 64)) -> Path:
    Image.new("RGB", size, color).save(path, "JPEG", quality=90)
    return path


def color_name(rgb) -> str:
    r, g, b = rgb
    if r > 150 and g < 120 and b < 120:
        return "red"
    if g > 100 and r < 120 and b < 120:
        return "green"
    if b > 150 and r < 120 and g < 120:
        return "blue"
    if r > 150 and g > 150 and b < 120:
        return "yellow"
    return "other"


def color_of_chat_payload(payload: dict) -> str:
    url = payload["messages"][1]["content"][0]["image_url"]["url"]
    b64 = url.split(",", 1)[1]
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    return color_name(im.convert("RGB").getpixel((im.width // 2, im.height // 2)))


class FakeResp:
    def __init__(self, status=200, payload=None, text="error body"):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeApi:
    """按图片颜色返回描述；embeddings 返回带标记的 1024 维向量。"""

    def __init__(self, embed_shuffle=False):
        self.calls: list[dict] = []
        self.attempts: dict[str, int] = {}
        # color -> 前 n 次失败（500），之后成功
        self.fail_first: dict[str, int] = {}
        self.embed_shuffle = embed_shuffle
        self.embed_status = 200
        self.embed_bad = ""   # "" / "no_data" / "wrong_count" / "bad_dim" / "inconsistent"

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "payload": json})
        if url.endswith("/chat/completions"):
            return self._chat(json)
        if url.endswith("/embeddings"):
            return self._embed(json)
        return FakeResp(404, text="not found")

    def _chat(self, payload):
        color = color_of_chat_payload(payload)
        n = self.attempts.get(color, 0) + 1
        self.attempts[color] = n
        if n <= self.fail_first.get(color, 0):
            return FakeResp(500, text="server busy")
        return FakeResp(
            200,
            {"choices": [{"message": {"content": f"{color}意境描述"}}]},
        )

    def _embed(self, payload):
        if self.embed_status != 200:
            return FakeResp(self.embed_status, text="upstream error")
        batch = list(payload["input"])
        if self.embed_bad == "no_data":
            return FakeResp(200, {"oops": []})
        data = []
        for i, _text in enumerate(batch):
            if self.embed_bad == "bad_dim":
                vec = np.zeros(512, dtype=np.float32)
            else:
                vec = np.zeros(DIM, dtype=np.float32)
                vec[0] = i + 1            # 批内位置标记
                vec[1] = 1.0              # 基准分量
                vec[2] = len(self.calls)  # 批次标记
            if self.embed_bad == "inconsistent" and i == len(batch) - 1:
                vec = np.zeros(DIM // 2, dtype=np.float32)
            data.append({"index": i, "embedding": vec.tolist()})
        if self.embed_bad == "wrong_count":
            data = data[:-1]
        if self.embed_shuffle:
            data = list(reversed(data))
        return FakeResp(200, {"data": data})


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """把配置/索引目录打桩到临时目录。"""
    monkeypatch.setattr(cs, "settings_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def settings():
    return Settings(asr_base_url="https://api.example.com/v1", asr_api_key="test-key")


@pytest.fixture
def api(monkeypatch):
    fake = FakeApi()
    monkeypatch.setattr(ci.requests, "post", fake.post)
    return fake


# ---------- 常量 ----------

def test_constants():
    assert ci._IMG_EXTS == {".jpg", ".jpeg", ".png"}
    assert ci._MAX_SIDE == 768
    assert ci._JPEG_QUALITY == 85
    assert ci._EMBED_BATCH == 32
    assert ci._INDEX_WORKERS == 8
    assert ci._RETRIES == 2
    assert ci._MAX_LYRIC_HINT == 200


# ---------- scan_folder ----------

def test_scan_new_images_sorted_and_ignores_non_images(tmp_path):
    folder = tmp_path / "covers"
    folder.mkdir()
    make_image(folder / "b.jpg", "blue")
    make_image(folder / "a.png", "red")
    make_image(folder / "c.jpeg", "green")
    (folder / "note.txt").write_text("ignore me", encoding="utf-8")
    (folder / "subdir").mkdir()

    diff = ScanDiff.scan_folder(CoverIndex(folder=str(folder)))

    assert tuple(e.name for e in diff.to_index) == ("a.png", "b.jpg", "c.jpeg")
    for e in diff.to_index:
        st = (folder / e.name).stat()
        assert e.mtime_ns == st.st_mtime_ns
        assert e.size == st.st_size
        assert e.desc == ""
        assert e.status == "ok"
    assert diff.missing_names == ()


def test_scan_changed_entry_clears_desc(tmp_path):
    folder = tmp_path / "covers"
    folder.mkdir()
    p = make_image(folder / "a.jpg", "red")
    st = p.stat()
    index = CoverIndex(
        folder=str(folder),
        images=(CoverEntry("a.jpg", mtime_ns=1, size=st.st_size + 10, desc="旧描述"),),
    )

    diff = ScanDiff.scan_folder(index)

    assert len(diff.to_index) == 1
    e = diff.to_index[0]
    assert e.desc == ""
    assert e.status == "ok"
    assert e.mtime_ns == st.st_mtime_ns
    assert e.size == st.st_size
    assert diff.missing_names == ()


def test_scan_unchanged_entry_not_included(tmp_path):
    folder = tmp_path / "covers"
    folder.mkdir()
    p = make_image(folder / "a.jpg", "red")
    st = p.stat()
    index = CoverIndex(
        folder=str(folder),
        images=(CoverEntry("a.jpg", st.st_mtime_ns, st.st_size, desc="红意境"),),
    )

    diff = ScanDiff.scan_folder(index)

    assert diff.to_index == ()
    assert diff.missing_names == ()


def test_scan_missing_names(tmp_path):
    folder = tmp_path / "covers"
    folder.mkdir()
    make_image(folder / "a.jpg", "red")
    st = (folder / "a.jpg").stat()
    index = CoverIndex(
        folder=str(folder),
        images=(
            CoverEntry("a.jpg", st.st_mtime_ns, st.st_size, "红"),
            CoverEntry("gone 图.jpg", 2, 2, "旧"),
        ),
    )

    diff = ScanDiff.scan_folder(index)

    assert diff.missing_names == ("gone 图.jpg",)
    assert tuple(e.name for e in diff.to_index) == ()


def test_scan_failed_entry_retried_even_unchanged(tmp_path):
    folder = tmp_path / "covers"
    folder.mkdir()
    p = make_image(folder / "a.jpg", "red")
    st = p.stat()
    index = CoverIndex(
        folder=str(folder),
        images=(CoverEntry("a.jpg", st.st_mtime_ns, st.st_size, status="failed"),),
    )

    diff = ScanDiff.scan_folder(index)

    assert tuple(e.name for e in diff.to_index) == ("a.jpg",)
    assert diff.to_index[0].desc == ""
    assert diff.to_index[0].status == "ok"


def test_scan_folder_not_exists_raises(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(CoverIndexError, match="封面目录不存在"):
        ScanDiff.scan_folder(CoverIndex(folder=str(missing)))


# ---------- _encode_image ----------

def test_encode_image_nonempty_and_resized(tmp_path):
    p = make_image(tmp_path / "big.jpg", "green", size=(1200, 900))

    b64 = _encode_image(p)

    assert isinstance(b64, str) and b64
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw))
    assert max(im.size) <= 768
    assert im.size == (768, 576)


def test_encode_image_small_kept_small(tmp_path):
    p = make_image(tmp_path / "small.png", "blue", size=(100, 200))
    raw = base64.b64decode(_encode_image(p))
    im = Image.open(io.BytesIO(raw))
    assert im.size == (100, 200)


# ---------- describe_image ----------

def test_describe_image_success(tmp_path, settings, api):
    p = make_image(tmp_path / "a.jpg", "red")
    desc = describe_image(p, settings)

    assert desc == "red意境描述"
    call = api.calls[-1]
    assert call["url"] == "https://api.example.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    payload = call["payload"]
    assert payload["model"] == cs.DEFAULT_VL_MODEL
    assert payload["temperature"] == 0.3
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert "一句不超过80字" in payload["messages"][0]["content"]
    assert payload["messages"][1]["content"][0]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_describe_image_custom_model(tmp_path, settings, api):
    p = make_image(tmp_path / "a.jpg", "red")
    describe_image(p, settings, model="custom/vl")
    assert api.calls[-1]["payload"]["model"] == "custom/vl"


@pytest.mark.parametrize(
    "status, hint",
    [(401, "视觉模型 API Key 无效或已过期"), (429, "请求过于频繁"), (500, "500")],
)
def test_describe_image_http_errors(tmp_path, settings, monkeypatch, status, hint):
    make_image(tmp_path / "a.jpg", "red")

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResp(status, text=f"upstream {status}")

    monkeypatch.setattr(ci.requests, "post", fake_post)
    with pytest.raises(CoverIndexError, match=hint):
        describe_image(tmp_path / "a.jpg", settings)


def test_describe_image_network_error(tmp_path, settings, monkeypatch):
    make_image(tmp_path / "a.jpg", "red")

    def fake_post(url, headers=None, json=None, timeout=None):
        raise requests.ConnectionError("dns failed")

    monkeypatch.setattr(ci.requests, "post", fake_post)
    with pytest.raises(CoverIndexError, match="网络请求失败"):
        describe_image(tmp_path / "a.jpg", settings)


def test_describe_image_bad_structure(tmp_path, settings, monkeypatch):
    make_image(tmp_path / "a.jpg", "red")

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResp(200, {"choices": []})

    monkeypatch.setattr(ci.requests, "post", fake_post)
    with pytest.raises(CoverIndexError, match="响应结构异常"):
        describe_image(tmp_path / "a.jpg", settings)


def test_describe_image_empty_content(tmp_path, settings, monkeypatch):
    make_image(tmp_path / "a.jpg", "red")

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResp(200, {"choices": [{"message": {"content": "   "}}]})

    monkeypatch.setattr(ci.requests, "post", fake_post)
    with pytest.raises(CoverIndexError, match="模型未返回描述"):
        describe_image(tmp_path / "a.jpg", settings)


def test_describe_image_non_json(tmp_path, settings, monkeypatch):
    make_image(tmp_path / "a.jpg", "red")

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResp(200, payload=ValueError("no json"), text="oops")

    monkeypatch.setattr(ci.requests, "post", fake_post)
    with pytest.raises(CoverIndexError, match="响应非 JSON"):
        describe_image(tmp_path / "a.jpg", settings)


# ---------- embed_texts ----------

def test_embed_empty_returns_1024_zero(settings, api):
    out = embed_texts([], settings)
    assert out.shape == (0, 1024)
    assert out.dtype == np.float32
    assert api.calls == []


def test_embed_success_and_l2_normalized(settings, api):
    out = embed_texts(["a", "b"], settings)
    assert out.shape == (2, 1024)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
    assert api.calls[-1]["url"] == "https://api.example.com/v1/embeddings"
    assert api.calls[-1]["payload"]["model"] == cs.DEFAULT_EMBED_MODEL


def test_embed_zero_vector_stays_zero(settings, monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        data = [{"index": i, "embedding": [0.0] * DIM} for i, _ in enumerate(json["input"])]
        return FakeResp(200, {"data": data})

    monkeypatch.setattr(ci.requests, "post", fake_post)
    out = embed_texts(["x"], settings)
    assert out.shape == (1, 1024)
    np.testing.assert_array_equal(out, np.zeros((1, 1024)))


def test_embed_batching_64_texts_and_order(settings, monkeypatch):
    fake = FakeApi()
    calls = fake.calls
    monkeypatch.setattr(ci.requests, "post", fake.post)
    texts = [f"text-{i}" for i in range(64)]

    out = embed_texts(texts, settings)

    assert len(calls) == 2
    assert calls[0]["payload"]["input"] == texts[:32]
    assert calls[1]["payload"]["input"] == texts[32:]
    assert out.shape == (64, 1024)
    # 标记向量（位置 i+1、基准 1、批次号）归一化后可验证拼接顺序
    for global_i, row in enumerate(out):
        local = global_i % 32
        batch_no = 1 if global_i < 32 else 2
        assert row[0] / row[1] == pytest.approx(local + 1, rel=1e-5)
        assert row[2] / row[1] == pytest.approx(batch_no, rel=1e-5)


def test_embed_uses_index_field_for_order(settings, monkeypatch):
    fake = FakeApi(embed_shuffle=True)
    monkeypatch.setattr(ci.requests, "post", fake.post)

    out = embed_texts(["a", "b", "c"], settings)

    # 响应 data 被故意倒序，但按 index 还原后顺序仍是 0,1,2
    assert out[0, 0] / out[0, 1] == pytest.approx(1, rel=1e-5)
    assert out[2, 0] / out[2, 1] == pytest.approx(3, rel=1e-5)


@pytest.mark.parametrize(
    "status, hint",
    [(401, "API Key"), (429, "请求过于频繁"), (500, "500")],
)
def test_embed_http_errors(settings, monkeypatch, status, hint):
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResp(status, text=f"upstream {status}")

    monkeypatch.setattr(ci.requests, "post", fake_post)
    with pytest.raises(CoverIndexError, match=hint):
        embed_texts(["a"], settings)


def test_embed_network_error(settings, monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        raise requests.Timeout("slow")

    monkeypatch.setattr(ci.requests, "post", fake_post)
    with pytest.raises(CoverIndexError, match="网络请求失败"):
        embed_texts(["a"], settings)


@pytest.mark.parametrize("bad", ["no_data", "wrong_count", "bad_dim", "inconsistent"])
def test_embed_bad_response_structure(settings, monkeypatch, bad):
    fake = FakeApi()
    fake.embed_bad = bad
    monkeypatch.setattr(ci.requests, "post", fake.post)
    with pytest.raises(CoverIndexError):
        embed_texts(["a", "b"], settings)


# ---------- describe_many ----------

def _entry(path: Path) -> CoverEntry:
    st = path.stat()
    return CoverEntry(path.name, st.st_mtime_ns, st.st_size)


def test_describe_many_mixed_order_and_progress(tmp_path, settings, api):
    folder = tmp_path / "covers"
    folder.mkdir()
    red = _entry(make_image(folder / "red.jpg", "red"))
    blue = _entry(make_image(folder / "blue.jpg", "blue"))
    green = _entry(make_image(folder / "green.jpg", "green"))
    api.fail_first["blue"] = 99   # 永远失败
    events = []

    out = describe_many(
        CoverIndex(folder=str(folder)),
        (red, blue, green),
        settings,
        progress=lambda d, t, n, ok: events.append((d, t, n, ok)),
        workers=1,
    )

    assert tuple(e.name for e in out) == ("red.jpg", "blue.jpg", "green.jpg")
    assert out[0].status == "ok" and out[0].desc == "red意境描述"
    assert out[1].status == "failed" and out[1].desc == ""
    assert out[2].status == "ok" and out[2].desc == "green意境描述"
    assert events == [
        (1, 3, "red.jpg", True),
        (2, 3, "blue.jpg", False),
        (3, 3, "green.jpg", True),
    ]


def test_describe_many_retries_then_succeeds(tmp_path, settings, api):
    folder = tmp_path / "covers"
    folder.mkdir()
    green = _entry(make_image(folder / "green.jpg", "green"))
    api.fail_first["green"] = 2   # 前两次失败，第三次成功

    out = describe_many(
        CoverIndex(folder=str(folder)), (green,), settings, workers=1
    )

    assert out[0].status == "ok"
    assert out[0].desc == "green意境描述"
    assert api.attempts["green"] == 3


def test_describe_many_cancel_immediate(tmp_path, settings, api):
    folder = tmp_path / "covers"
    folder.mkdir()
    entries = tuple(
        _entry(make_image(folder / f"{c}.jpg", c)) for c in ("red", "green", "blue")
    )
    events = []

    out = describe_many(
        CoverIndex(folder=str(folder)),
        entries,
        settings,
        progress=lambda *a: events.append(a),
        should_cancel=lambda: True,
        workers=1,
    )

    assert tuple(e.status for e in out) == ("failed", "failed", "failed")
    assert all(e.desc == "" for e in out)
    assert events == []
    assert api.calls == []


def test_describe_many_cancel_partial(tmp_path, settings, api):
    folder = tmp_path / "covers"
    folder.mkdir()
    red = _entry(make_image(folder / "red.jpg", "red"))
    blue = _entry(make_image(folder / "blue.jpg", "blue"))
    green = _entry(make_image(folder / "green.jpg", "green"))
    api.fail_first["blue"] = 99
    checks = {"n": 0}

    def should_cancel():
        checks["n"] += 1
        return checks["n"] >= 3   # 提交两张后停止

    events = []
    out = describe_many(
        CoverIndex(folder=str(folder)),
        (red, blue, green),
        settings,
        progress=lambda *a: events.append(a),
        should_cancel=should_cancel,
        workers=1,
    )

    assert tuple(e.name for e in out) == ("red.jpg", "blue.jpg", "green.jpg")
    assert out[0].status == "ok"
    assert out[1].status == "failed"
    assert out[2].status == "failed"   # 未提交：ok 且 desc 空 → failed
    assert len(events) == 2


def test_describe_many_empty_entries(tmp_path, settings, api):
    out = describe_many(
        CoverIndex(folder=str(tmp_path)), (), settings, workers=1
    )
    assert out == ()
    assert api.calls == []


# ---------- run_incremental ----------

def _cover_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "covers"
    folder.mkdir()
    make_image(folder / "red.jpg", "red")
    make_image(folder / "green.jpg", "green")
    make_image(folder / "blue.jpg", "blue")
    return folder


def test_run_incremental_first_build(tmp_path, settings, cfg, api):
    folder = _cover_folder(tmp_path)
    events = []

    merged = run_incremental(
        CoverIndex(folder=str(folder)),
        settings,
        progress=lambda *a: events.append(a),
    )

    assert tuple(e.name for e in merged.ok_entries()) == (
        "blue.jpg", "green.jpg", "red.jpg",
    )
    assert merged.entry("red.jpg").desc == "red意境描述"
    saved = load_embeddings(tuple(e.name for e in merged.ok_entries()))
    assert saved is not None
    matrix, emb_names = saved
    assert emb_names == ("blue.jpg", "green.jpg", "red.jpg")
    assert matrix.shape == (3, 1024)
    assert (cfg / "cover_index.json").exists()
    assert (cfg / "cover_embeddings.npz").exists()
    phases = {e[0] for e in events}
    assert phases == {"scan", "describe", "embed"}


def test_run_incremental_no_changes_no_model_calls(tmp_path, settings, cfg, api):
    folder = _cover_folder(tmp_path)
    index = CoverIndex(folder=str(folder))
    first = run_incremental(index, settings)

    api.calls.clear()
    second = run_incremental(first, settings)

    assert api.calls == []   # 视觉模型与嵌入均不调用
    assert tuple(e.name for e in second.ok_entries()) == (
        "blue.jpg", "green.jpg", "red.jpg",
    )


def test_run_incremental_add_one_reuses_old_vectors(tmp_path, settings, cfg, api):
    folder = _cover_folder(tmp_path)
    first = run_incremental(CoverIndex(folder=str(folder)), settings)
    old = load_embeddings(tuple(e.name for e in first.ok_entries()))[0]

    api.calls.clear()
    make_image(folder / "yellow.jpg", "yellow")
    second = run_incremental(first, settings)

    names = tuple(e.name for e in second.ok_entries())
    assert names == ("blue.jpg", "green.jpg", "red.jpg", "yellow.jpg")
    matrix = load_embeddings(names)[0]
    np.testing.assert_array_equal(matrix[:3], old)   # 旧向量逐行复用
    embed_calls = [c for c in api.calls if c["url"].endswith("/embeddings")]
    assert len(embed_calls) == 1
    assert embed_calls[0]["payload"]["input"] == ["yellow意境描述"]


def test_run_incremental_missing_marked_and_vectors_removed(tmp_path, settings, cfg, api):
    folder = _cover_folder(tmp_path)
    first = run_incremental(CoverIndex(folder=str(folder)), settings)

    api.calls.clear()
    (folder / "green.jpg").unlink()
    second = run_incremental(first, settings)

    assert second.entry("green.jpg").status == "missing"
    names = tuple(e.name for e in second.ok_entries())
    assert names == ("blue.jpg", "red.jpg")
    matrix, emb_names = load_embeddings(names)
    assert emb_names == names
    assert matrix.shape == (2, 1024)
    assert not any(c["url"].endswith("/chat/completions") for c in api.calls)


def test_run_incremental_changed_file_reembedded(tmp_path, settings, cfg, api):
    folder = _cover_folder(tmp_path)
    first = run_incremental(CoverIndex(folder=str(folder)), settings)
    old = load_embeddings(tuple(e.name for e in first.ok_entries()))[0]

    api.calls.clear()
    make_image(folder / "red.jpg", "yellow", size=(128, 48))  # 内容变更
    second = run_incremental(first, settings)

    names = tuple(e.name for e in second.ok_entries())
    matrix = load_embeddings(names)[0]
    np.testing.assert_array_equal(matrix[:2], old[:2])   # 未变更行复用
    embed_calls = [c for c in api.calls if c["url"].endswith("/embeddings")]
    assert embed_calls[0]["payload"]["input"] == ["yellow意境描述"]


def test_run_incremental_cancel_saves_partial(tmp_path, settings, cfg, api):
    folder = _cover_folder(tmp_path)
    checks = {"n": 0}

    def should_cancel():
        checks["n"] += 1
        return checks["n"] >= 2   # 只提交第一张

    merged = run_incremental(
        CoverIndex(folder=str(folder)),
        settings,
        should_cancel=should_cancel,
    )

    assert merged.entry("blue.jpg").status == "ok"
    assert merged.entry("blue.jpg").desc
    assert merged.entry("green.jpg").status == "failed"
    assert merged.entry("red.jpg").status == "failed"
    names = tuple(e.name for e in merged.ok_entries())
    saved = load_embeddings(names)
    assert saved is not None
    assert saved[1] == ("blue.jpg",)


def test_run_incremental_empty_folder(tmp_path, settings, cfg, api):
    folder = tmp_path / "empty"
    folder.mkdir()
    merged = run_incremental(CoverIndex(folder=str(folder)), settings)

    assert merged.images == ()
    assert (cfg / "cover_index.json").exists()
    saved = load_embeddings(())
    assert saved is not None
    assert saved[1] == ()

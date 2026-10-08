"""cover_matcher 单元测试：查询文本构建与意境匹配（批内 claim、used 排除）。

全程不联网：embed_texts 底层 requests.post 打桩。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core import cover_indexer as ci
from src.core.cover_matcher import CoverMatch, CoverQuery, build_query_text, match_all
from src.core.cover_registry import UsedCovers
from src.core.cover_store import CoverEntry, CoverIndex
from src.core.settings import Settings

DIM = 1024


def _cover_entry(name: str, desc: str) -> CoverEntry:
    return CoverEntry(name=name, mtime_ns=1, size=1, desc=desc)


def _make_index(folder, names):
    return CoverIndex(
        folder=str(folder),
        images=tuple(_cover_entry(n, f"desc-{n}") for n in names),
    )


class QueryEmbedFake:
    """对 /embeddings 请求返回调用方指定的查询向量；其余请求报错。"""

    def __init__(self, wanted_rows: list[list[int]]):
        self.calls: list[dict] = []
        self.wanted_rows = wanted_rows

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "payload": json})
        assert url.endswith("/embeddings")
        data = []
        for i, _text in enumerate(json["input"]):
            vec = np.zeros(DIM, dtype=np.float32)
            for row in self.wanted_rows[i]:
                vec[row] = 1.0
            data.append({"index": i, "embedding": vec.tolist()})
        return _Resp(200, {"data": data})


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self.text = ""
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def settings():
    return Settings(asr_base_url="https://api.example.com/v1", asr_api_key="k")


def _wire(monkeypatch, wanted_rows):
    fake = QueryEmbedFake(wanted_rows)
    monkeypatch.setattr(ci.requests, "post", fake.post)
    return fake


# ---------- build_query_text ----------

def test_build_query_text_full():
    q = CoverQuery(key="k", title="夜雨", reason="雨声意象", lyrics="窗外的雨 不停")
    text = build_query_text(q)
    assert text == "歌名：夜雨\n命名依据：雨声意象\n歌词意象：窗外的雨 不停"


def test_build_query_text_empty_reason_and_lyrics():
    text = build_query_text(CoverQuery(key="k", title="夜"))
    assert text == "歌名：夜\n命名依据："


def test_build_query_text_blank_lyrics_treated_as_empty():
    text = build_query_text(CoverQuery(key="k", title="夜", lyrics="   "))
    assert text == "歌名：夜\n命名依据："


def test_build_query_text_long_lyrics_truncated():
    lyrics = "雨" * 300
    text = build_query_text(CoverQuery(key="k", title="夜", lyrics=lyrics))
    assert text.endswith("雨" * 200)
    assert len(text) == len("歌名：夜\n命名依据：\n歌词意象：") + 200


# ---------- match_all 主流程 ----------

def test_match_order_preserved_and_highest_score(tmp_path, settings, monkeypatch):
    names = ("a.jpg", "b.jpg", "c.jpg", "d.jpg")
    folder = tmp_path / "covers"
    index = _make_index(folder, names)
    covers = np.eye(4, DIM, dtype=np.float32)
    queries = [
        CoverQuery(key=r"E:\songs\01.mp3", title="q1"),
        CoverQuery(key=r"E:\songs\02.mp3", title="q2"),
        CoverQuery(key=r"E:\songs\03.mp3", title="q3"),
    ]
    fake = _wire(monkeypatch, [[3], [0], [1]])

    matches = match_all(index, covers, names, UsedCovers(), queries, settings)

    assert [m.key for m in matches] == [
        r"E:\songs\01.mp3", r"E:\songs\02.mp3", r"E:\songs\03.mp3",
    ]
    assert matches[0].cover_name == "d.jpg"
    assert matches[0].score == pytest.approx(1.0, abs=1e-6)
    assert matches[1].cover_name == "a.jpg"
    assert matches[2].cover_name == "b.jpg"
    assert all(m.error is None for m in matches)
    # 查询文本确实经过 build_query_text 组装
    inputs = fake.calls[0]["payload"]["input"]
    assert inputs[0].startswith("歌名：q1")


def test_match_claim_within_batch_no_duplicate(tmp_path, settings, monkeypatch):
    names = ("a.jpg", "b.jpg", "c.jpg")
    index = _make_index(tmp_path / "covers", names)
    covers = np.eye(3, DIM, dtype=np.float32)
    queries = [
        CoverQuery(key="1", title="x"),
        CoverQuery(key="2", title="y"),
    ]
    _wire(monkeypatch, [[0], [0]])   # 两个 query 都最匹配第 0 行

    matches = match_all(index, covers, names, UsedCovers(), queries, settings)

    assert matches[0].cover_name == "a.jpg"
    assert matches[0].score == pytest.approx(1.0, abs=1e-6)
    chosen = {m.cover_name for m in matches}
    assert len(chosen) == 2   # 批内 claim，第二首选别的行
    assert matches[1].score == pytest.approx(0.0, abs=1e-6)
    assert matches[1].error is None


def test_match_excludes_used_paths(tmp_path, settings, monkeypatch):
    names = ("a.jpg", "b.jpg", "c.jpg")
    folder = tmp_path / "covers"
    index = _make_index(folder, names)
    covers = np.eye(3, DIM, dtype=np.float32)
    used = UsedCovers(paths=frozenset({str(folder / "a.jpg")}))
    _wire(monkeypatch, [[0]])

    matches = match_all(
        index, covers, names, used, [CoverQuery(key="1", title="x")], settings
    )

    assert matches[0].cover_name != "a.jpg"
    assert matches[0].cover_name == "b.jpg"   # 其余同分，按序取首个


def test_match_exhausted(tmp_path, settings, monkeypatch):
    names = ("a.jpg", "b.jpg")
    index = _make_index(tmp_path / "covers", names)
    covers = np.eye(2, DIM, dtype=np.float32)
    queries = [CoverQuery(key=str(i), title=f"q{i}") for i in range(3)]
    _wire(monkeypatch, [[0], [1], [0]])

    matches = match_all(index, covers, names, UsedCovers(), queries, settings)

    assert matches[0].cover_name == "a.jpg"
    assert matches[1].cover_name == "b.jpg"
    assert matches[2].cover_name is None
    assert matches[2].score == 0.0
    assert matches[2].error == "封面素材已用尽"


def test_match_empty_index_no_network(tmp_path, settings, monkeypatch):
    index = CoverIndex(folder=str(tmp_path / "covers"))
    fake = _wire(monkeypatch, [])
    queries = [CoverQuery(key="1", title="x"), CoverQuery(key="2", title="y")]

    matches = match_all(index, np.zeros((0, DIM)), (), UsedCovers(), queries, settings)

    assert len(matches) == 2
    assert all(m.cover_name is None for m in matches)
    assert all(m.error == "封面素材已用尽" for m in matches)
    assert fake.calls == []   # 无候选不发请求


def test_match_all_candidates_used_up(tmp_path, settings, monkeypatch):
    names = ("a.jpg", "b.jpg")
    folder = tmp_path / "covers"
    index = _make_index(folder, names)
    covers = np.eye(2, DIM, dtype=np.float32)
    used = UsedCovers(
        paths=frozenset({str(folder / "a.jpg"), str(folder / "b.jpg")})
    )
    fake = _wire(monkeypatch, [])

    matches = match_all(
        index, covers, names, used, [CoverQuery(key="1", title="x")], settings
    )

    assert matches[0].cover_name is None
    assert matches[0].error == "封面素材已用尽"
    assert fake.calls == []


def test_match_scores_in_range(tmp_path, settings, monkeypatch):
    rng = np.random.default_rng(42)
    names = tuple(f"c{i}.jpg" for i in range(6))
    index = _make_index(tmp_path / "covers", names)
    covers = rng.standard_normal((6, DIM)).astype(np.float32)
    covers /= np.linalg.norm(covers, axis=1, keepdims=True)
    queries = [CoverQuery(key=str(i), title=f"q{i}") for i in range(6)]

    def fake_post(url, headers=None, json=None, timeout=None):
        data = []
        for i, _ in enumerate(json["input"]):
            vec = rng.standard_normal(DIM).astype(np.float32)
            data.append({"index": i, "embedding": vec.tolist()})
        return _Resp(200, {"data": data})

    monkeypatch.setattr(ci.requests, "post", fake_post)

    matches = match_all(index, covers, names, UsedCovers(), queries, settings)

    assert len({m.cover_name for m in matches}) == 6
    for m in matches:
        assert -1.0 <= m.score <= 1.0
        assert m.error is None


def test_match_empty_queries(tmp_path, settings, monkeypatch):
    names = ("a.jpg",)
    index = _make_index(tmp_path / "covers", names)
    fake = _wire(monkeypatch, [])

    matches = match_all(
        index, np.eye(1, DIM, dtype=np.float32), names, UsedCovers(), [], settings
    )

    assert matches == []
    assert fake.calls == []


def test_match_result_is_immutable():
    m = CoverMatch(key="k", cover_name="a.jpg", score=0.5)
    assert m.error is None
    with pytest.raises(Exception):
        m.score = 1.0  # type: ignore[misc]


def test_query_is_immutable():
    q = CoverQuery(key="k", title="t")
    assert q.reason == "" and q.lyrics == ""
    with pytest.raises(Exception):
        q.title = "x"  # type: ignore[misc]

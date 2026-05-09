import sys
import types

from seg_pipeline.rag_integration import rag_search_topk


def test_rag_search_topk_adapts_results_and_forwards_kwargs(monkeypatch):
    captured = {"init_kwargs": None, "search_args": None}

    class FakeRAGSystem:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = dict(kwargs)

        def search(self, query_image: str, k: int):
            captured["search_args"] = {"query_image": query_image, "k": k}
            return [
                ("img1.png", "mask1.png", 1.25),
                ("img2.png", "mask2.png", "2.5"),
            ]

    rag_pkg = types.ModuleType("rag")
    rag_mod = types.ModuleType("rag.rag_system")
    rag_mod.RAGSystem = FakeRAGSystem
    rag_pkg.rag_system = rag_mod

    monkeypatch.setitem(sys.modules, "rag", rag_pkg)
    monkeypatch.setitem(sys.modules, "rag.rag_system", rag_mod)

    items, meta = rag_search_topk(
        query_image_path="query.png",
        k=5,
        rag_kwargs={"index_dir": "/tmp/index", "embedder_device": "cpu"},
    )

    assert captured["init_kwargs"] == {"index_dir": "/tmp/index", "embedder_device": "cpu"}
    assert captured["search_args"] == {"query_image": "query.png", "k": 5}

    assert [it.image_path for it in items] == ["img1.png", "img2.png"]
    assert [it.mask_path for it in items] == ["mask1.png", "mask2.png"]
    assert [it.distance for it in items] == [1.25, 2.5]

    assert meta.k == 5
    assert meta.raw_result_count == 2

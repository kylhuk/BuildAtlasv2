from backend.engine.archive import (
    ArchiveStore,
    DescriptorAxisSpec,
    ExploitEmitter,
    NoveltyEmitter,
    UncertaintyEmitter,
    deterministic_allocator,
)


def test_archive_store_metrics_and_replacement():
    axes = (
        DescriptorAxisSpec("x", "x", bins=4, min_value=0.0, max_value=4.0),
        DescriptorAxisSpec("y", "y", bins=2, min_value=0.0, max_value=2.0),
    )
    store = ArchiveStore(axes=axes)
    assert store.insert("first", score=1.0, descriptor=(0.0, 0.0))
    assert store.insert("second", score=2.0, descriptor=(0.0, 0.0))
    first_entry = store.entry_for_bin("0-0")
    assert first_entry is not None
    assert first_entry.build_id == "second"

    assert store.insert("tied-a", score=2.0, descriptor=(1.1, 1.1))
    assert store.insert("tied-b", score=2.0, descriptor=(1.1, 1.1)) is False
    tied_entry = store.entry_for_bin("1-1")
    assert tied_entry is not None
    assert tied_entry.build_id == "tied-a"

    assert store.insert("third", score=3.5, descriptor=(3.9, 1.9))

    metrics = store.metrics()
    assert metrics.total_bins == 8
    assert metrics.bins_filled == 3
    assert metrics.coverage == metrics.bins_filled / metrics.total_bins
    assert metrics.qd_score == 2.0 + 2.0 + 3.5


def test_emitters_and_allocator():
    axes = (DescriptorAxisSpec("a", "a", bins=2, min_value=0.0, max_value=2.0),)
    store = ArchiveStore(axes=axes)
    store.insert("high", score=100.0, descriptor=(0.1,))
    store.insert("low", score=10.0, descriptor=(1.9,))
    entries = store.entries()

    emitters = (ExploitEmitter(), NoveltyEmitter(), UncertaintyEmitter())
    allocation = deterministic_allocator(5, emitters)
    assert allocation == {"exploit": 2, "novelty": 2, "uncertainty": 1}

    exploit_selected = emitters[0].select(entries, allocation["exploit"])
    assert [entry.build_id for entry in exploit_selected] == ["high", "low"]

    novelty_selected = emitters[1].select(entries, allocation["novelty"])
    assert len(novelty_selected) == 2
    assert novelty_selected[0].bin_key <= novelty_selected[1].bin_key

    assert emitters[2].select(entries, allocation["uncertainty"]) == []

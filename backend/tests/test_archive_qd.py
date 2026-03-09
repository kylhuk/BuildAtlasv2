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
    store.insert("high", score=100.0, descriptor=(0.1,), metadata={"pass_probability": 0.6})
    store.insert("low", score=10.0, descriptor=(1.9,), metadata={"pass_probability": 0.2})
    entries = store.entries()

    emitters = (ExploitEmitter(), NoveltyEmitter(), UncertaintyEmitter())
    allocation = deterministic_allocator(5, emitters)
    assert allocation == {"exploit": 2, "novelty": 2, "uncertainty": 1}

    exploit_selected = emitters[0].select(entries, allocation["exploit"])
    assert [entry.build_id for entry in exploit_selected] == ["high", "low"]

    novelty_selected = emitters[1].select(entries, allocation["novelty"])
    assert len(novelty_selected) == 2
    assert novelty_selected[0].bin_key <= novelty_selected[1].bin_key

    uncertainty_selected = emitters[2].select(entries, allocation["uncertainty"])
    assert [entry.build_id for entry in uncertainty_selected] == ["high"]


def test_default_descriptor_axes_use_log_damage_and_max_hit():
    store = ArchiveStore()
    assert len(store.axes) == 3
    assert store.axes[0].metric_key == "full_dps"
    assert store.axes[0].transform == "log10"
    assert store.axes[1].metric_key == "max_hit"
    assert store.axes[1].transform == "log10"
    assert store.axes[2].metric_key == "utility_score"
    assert store.axes[2].transform == "identity"


def test_high_dps_values_fill_multiple_damage_bins():
    store = ArchiveStore()
    constant_max_hit = 1200.0
    constant_utility_score = 2.0
    dps_values = [5000.0, 20000.0, 1e7]
    for idx, dps in enumerate(dps_values):
        assert store.insert(
            f"build-{idx}",
            score=float(idx),
            descriptor=(dps, constant_max_hit, constant_utility_score),
        )
    entries = store.entries()
    assert len(entries) == len(dps_values)
    damage_bins = {entry.bin_key.split("-")[0] for entry in entries}
    assert len(damage_bins) == len(dps_values)

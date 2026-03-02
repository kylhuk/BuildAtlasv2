from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app, get_artifact_base_path
from backend.engine.archive import ArchiveStore, DescriptorAxisSpec, load_archive_artifact, persist_archive


def test_archive_endpoints(tmp_path: Path) -> None:
    app.dependency_overrides.clear()
    app.dependency_overrides[get_artifact_base_path] = lambda: tmp_path
    client = TestClient(app)
    try:
        store = ArchiveStore(
            axes=(DescriptorAxisSpec("alpha", "alpha", bins=2, min_value=0.0, max_value=2.0),)
        )
        store.insert("entry", score=5.0, descriptor=(0.1,), metadata={"seed": 42})
        persist_archive("test-run", store, base_path=tmp_path, created_at="2025-01-01T00:00:00Z")

        artifact = load_archive_artifact(
            "test-run", base_path=tmp_path
        )
        axis_entry = artifact["axes"][0]
        assert axis_entry["name"] == "alpha"
        assert axis_entry["transform"] == "identity"

        response = client.get("/archives/test-run")
        assert response.status_code == 200
        payload = response.json()
        assert payload["run_id"] == "test-run"
        assert payload["metrics"]["bins_filled"] == 1
        assert len(payload["bins"]) == 1
        bin_key = payload["bins"][0]["bin_key"]

        bin_response = client.get(f"/archives/test-run/bins/{bin_key}")
        assert bin_response.status_code == 200
        bin_payload = bin_response.json()
        assert bin_payload["build_id"] == "entry"
        assert bin_payload["metadata"]["seed"] == 42

        missing = client.get("/archives/test-run/bins/missing")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "archive_bin_not_found"
    finally:
        client.close()
        app.dependency_overrides.clear()



def test_default_axes_persist_transform_metadata(tmp_path: Path) -> None:
    store = ArchiveStore()
    store.insert(
        "default",
        score=1.0,
        descriptor=(1000.0, 1200.0),
    )
    persist_archive(
        "default-axes",
        store,
        base_path=tmp_path,
        created_at="2025-01-03T00:00:00Z",
    )
    artifact = load_archive_artifact(
        "default-axes", base_path=tmp_path
    )
    axes = {axis["name"]: axis for axis in artifact["axes"]}
    assert axes["damage"]["transform"] == "log10"
    assert axes["max_hit"]["transform"] == "log10"


def test_archive_frontier_endpoint(tmp_path: Path) -> None:
    app.dependency_overrides.clear()
    app.dependency_overrides[get_artifact_base_path] = lambda: tmp_path
    client = TestClient(app)
    try:
        store = ArchiveStore(
            axes=(
                DescriptorAxisSpec("damage", "damage", bins=4, min_value=0.0, max_value=10.0),
                DescriptorAxisSpec("utility", "utility", bins=4, min_value=0.0, max_value=10.0),
            )
        )
        store.insert("alpha_focus", score=9.0, descriptor=(9.0, 3.0))
        store.insert("beta_focus", score=8.0, descriptor=(3.0, 9.0))
        store.insert("dominated", score=5.0, descriptor=(4.0, 2.0))
        persist_archive(
            "frontier-run",
            store,
            base_path=tmp_path,
            created_at="2025-01-02T00:00:00Z",
        )

        response = client.get("/archives/frontier-run/frontier")
        assert response.status_code == 200
        payload = response.json()
        assert {
            bin_entry["build_id"]
            for bin_entry in payload["bins"]
        } == {"alpha_focus", "beta_focus"}
        for bin_entry in payload["bins"]:
            assert bin_entry["tradeoff_reasons"]
            links = bin_entry["artifact_links"]
            assert any(
                link["label"] == "build_detail"
                and link["url"] == f"/builds/{bin_entry['build_id']}"
                for link in links
            )
            assert any(link["label"] == "export_xml" for link in links)
            assert any(link["label"] == "export_code" for link in links)
    finally:
        client.close()
        app.dependency_overrides.clear()


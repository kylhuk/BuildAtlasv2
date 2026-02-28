from __future__ import annotations

from backend.tests.test_api_flow import _prepare_client


def test_catalog_columns_registry_v1_endpoint(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        response = client.get("/catalog/columns/v1")
        assert response.status_code == 200
        payload = response.json()

        assert payload["registry_version"] == "catalog_columns_v1"
        columns = payload["columns"]
        assert isinstance(columns, list)
        assert columns

        ids = [column["id"] for column in columns]
        assert len(ids) == len(set(ids))

        required = {
            "build_id",
            "created_at",
            "class",
            "ascendancy",
            "main_skill",
            "full_dps",
            "max_hit",
            "total_cost_chaos",
            "price_snapshot_id",
        }
        assert required.issubset(set(ids))
    finally:
        client.close()

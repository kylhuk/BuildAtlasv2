import importlib
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from clickhouse_connect.driver.exceptions import ClickHouseError

from ..app.db.ch import (
    BuildCostRow,
    BuildInsertPayload,
    BuildListFilters,
    ClickhouseRepository,
    ScenarioMetricRow,
)


class DummyResult:
    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def named_results(self):
        return iter(self._rows)

    def close(self):
        self.closed = True


def sample_build_payload() -> BuildInsertPayload:
    return BuildInsertPayload(
        build_id="build-123",
        created_at=datetime(2025, 1, 1, 0, 0),
        ruleset_id="ruleset",
        profile_id="profile",
        class_="Marauder",
        ascendancy="Slayer",
        main_skill="Sunder",
        damage_type="physical",
        defence_type="armor",
        complexity_bucket="base",
        pob_xml_path="path/to/xml",
        pob_code_path="path/to/code",
        genome_path="path/to/genome",
        tags=["a", "b"],
        status="complete",
    )


def sample_scenario_row() -> ScenarioMetricRow:
    return ScenarioMetricRow(
        build_id="build-123",
        ruleset_id="ruleset",
        scenario_id="boss",
        gate_pass=True,
        gate_fail_reasons=[],
        pob_warnings=["warning"],
        evaluated_at=datetime(2025, 1, 2, 0, 0),
        full_dps=123.4,
        max_hit=56.7,
        armour=890,
        evasion=12,
        life=1000,
        mana=200,
        utility_score=0.5,
    )


def sample_build_cost_row() -> BuildCostRow:
    return BuildCostRow(
        build_id="build-123",
        ruleset_id="ruleset",
        price_snapshot_id="sentinel-2026-02-26T120000Z",
        total_cost_chaos=321.5,
        unknown_cost_count=1,
        slot_costs_json_path="data/builds/build-123/slot_costs.json",
        gem_costs_json_path="data/builds/build-123/gem_costs.json",
        calculated_at=datetime(2025, 1, 3, 0, 0),
    )


def test_insert_build_calls_insert_with_ordered_columns():
    client = MagicMock()
    repo = ClickhouseRepository(client=client)
    payload = sample_build_payload()

    repo.insert_build(payload)

    client.insert.assert_called_once()
    call_kwargs = client.insert.call_args.kwargs
    assert call_kwargs["table"] == "builds"
    assert call_kwargs["column_names"][0] == "build_id"
    assert call_kwargs["data"][0][0] == "build-123"
    assert call_kwargs["data"][0][4] == "Marauder"
    assert call_kwargs["database"]


def test_insert_scenario_metrics_batches_rows():
    client = MagicMock()
    repo = ClickhouseRepository(client=client)
    rows = [sample_scenario_row(), sample_scenario_row()]

    repo.insert_scenario_metrics(rows)

    client.insert.assert_called_once()
    kwargs = client.insert.call_args.kwargs
    assert kwargs["table"] == "scenario_metrics"
    assert kwargs["data"][0][3] == 1
    assert len(kwargs["data"]) == 2


def test_insert_scenario_metrics_raises_schema_mismatch_hint():
    client = MagicMock()
    repo = ClickhouseRepository(client=client)
    missing_error = ClickHouseError(
        "Unrecognized column 'metrics_source' in table scenario_metrics"
    )
    client.insert.side_effect = missing_error

    with pytest.raises(RuntimeError, match="make db-init"):
        repo.insert_scenario_metrics([sample_scenario_row()])

    assert client.insert.call_count == 1
    first_columns = client.insert.call_args.kwargs["column_names"]
    assert "metrics_source" in first_columns


def test_insert_scenario_metrics_propagates_unrelated_errors():
    client = MagicMock()
    repo = ClickhouseRepository(client=client)
    client.insert.side_effect = ClickHouseError("Syntax error near 'full_dps'")

    with pytest.raises(ClickHouseError):
        repo.insert_scenario_metrics([sample_scenario_row()])

    assert client.insert.call_count == 1


def test_insert_build_costs_batches_rows():
    client = MagicMock()
    repo = ClickhouseRepository(client=client)
    rows = [sample_build_cost_row(), sample_build_cost_row()]

    repo.insert_build_costs(rows)

    client.insert.assert_called_once()
    kwargs = client.insert.call_args.kwargs
    assert kwargs["table"] == "build_costs"
    assert kwargs["data"][0][2] == "sentinel-2026-02-26T120000Z"
    assert kwargs["data"][0][3] == 321.5
    assert len(kwargs["data"]) == 2


def test_get_build_returns_row_mapping_and_closes_cursor():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123"}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)

    result = repo.get_build("build-123")

    assert result == {"build_id": "build-123"}
    client.query.assert_called_once()
    assert dummy.closed


def test_get_latest_build_cost_returns_row_mapping_and_closes_cursor():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123", "total_cost_chaos": 321.5}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)

    result = repo.get_latest_build_cost("build-123")

    assert result == {"build_id": "build-123", "total_cost_chaos": 321.5}
    client.query.assert_called_once()
    assert dummy.closed


def test_list_builds_applies_filters_and_sort():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123"}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)
    filters = BuildListFilters(
        ruleset_id="ruleset",
        profile_id="profile",
        class_="Marauder",
        ascendancy="Slayer",
        status="complete",
        created_after=datetime(2025, 1, 1),
        created_before=datetime(2025, 1, 3),
    )

    result = repo.list_builds(
        filters=filters,
        sort_by="ruleset_id",
        sort_dir="asc",
        limit=5,
        offset=2,
    )

    assert result == [{"build_id": "build-123"}]
    assert client.query.call_count == 1
    args, query_kwargs = client.query.call_args
    assert query_kwargs["parameters"]["ruleset_id"] == "ruleset"
    assert query_kwargs["parameters"]["class"] == "Marauder"
    assert "ORDER BY b.ruleset_id ASC" in args[0]
    assert dummy.closed


def test_list_builds_cost_filter_limits_max_cost():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123"}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)
    filters = BuildListFilters(max_cost_chaos=250.0)

    repo.list_builds(filters=filters)

    args, query_kwargs = client.query.call_args
    assert "bc.total_cost_chaos <= {max_cost_chaos:Float64}" in args[0]
    assert query_kwargs["parameters"]["max_cost_chaos"] == 250.0


def test_list_builds_exclude_unknown_cost_filter():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123"}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)
    filters = BuildListFilters(exclude_unknown_cost=True)

    repo.list_builds(filters=filters)

    args, _ = client.query.call_args
    assert "bc.unknown_cost_count = 0" in args[0]


def test_list_builds_price_snapshot_and_cost_window_filters():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123"}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)
    filters = BuildListFilters(
        price_snapshot_id="snapshot-a",
        cost_calculated_after=datetime(2025, 1, 1),
        cost_calculated_before=datetime(2025, 1, 2),
    )

    repo.list_builds(filters=filters)

    args, query_kwargs = client.query.call_args
    assert "bc.price_snapshot_id = {price_snapshot_id:String}" in args[0]
    assert "bc.calculated_at >= {cost_calculated_after:DateTime}" in args[0]
    assert "bc.calculated_at <= {cost_calculated_before:DateTime}" in args[0]
    assert "ORDER BY calculated_at DESC" in args[0]
    assert "LIMIT 1 BY build_id, ruleset_id" in args[0]
    assert query_kwargs["parameters"]["price_snapshot_id"] == "snapshot-a"


def test_list_builds_medium_priority_sort_requires_narrow_filter():
    repo = ClickhouseRepository(client=MagicMock())

    with pytest.raises(ValueError, match="requires a narrowing filter"):
        repo.list_builds(sort_by="total_cost_chaos", sort_dir="desc", limit=50)


def test_list_builds_medium_priority_sort_requires_small_limit():
    repo = ClickhouseRepository(client=MagicMock())
    filters = BuildListFilters(profile_id="mapping")

    with pytest.raises(ValueError, match="limit <= 100"):
        repo.list_builds(filters=filters, sort_by="total_cost_chaos", sort_dir="desc", limit=101)


def test_list_builds_scenario_aggregates_metrics_and_closes_result():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123"}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)
    filters = BuildListFilters(scenario_id="boss", gate_pass=True)

    result = repo.list_builds(filters=filters, sort_by="full_dps", sort_dir="asc")

    assert result == [{"build_id": "build-123"}]
    assert client.query.call_count == 1
    args, query_kwargs = client.query.call_args
    assert "LEFT JOIN (SELECT build_id, ruleset_id, scenario_id" in args[0]
    assert "GROUP BY build_id, ruleset_id, scenario_id" in args[0]
    assert "ON b.build_id = sm.build_id AND b.ruleset_id = sm.ruleset_id" in args[0]
    assert "sm.scenario_id = {scenario_id:String}" in args[0]
    assert "ORDER BY sm.full_dps ASC" in args[0]
    assert query_kwargs["parameters"]["scenario_id"] == "boss"
    assert query_kwargs["parameters"]["gate_pass"] == 1
    assert "FROM build_costs" in args[0]
    assert dummy.closed


def test_list_builds_metric_sort_requires_scenario_id():
    repo = ClickhouseRepository(client=MagicMock())

    with pytest.raises(ValueError, match="requires scenario_id"):
        repo.list_builds(sort_by="full_dps")

    for sort_field in ("dps_per_chaos", "max_hit_per_chaos"):
        with pytest.raises(ValueError, match="requires scenario_id"):
            repo.list_builds(sort_by=sort_field)


def test_list_builds_gate_pass_requires_scenario_id():
    repo = ClickhouseRepository(client=MagicMock())
    filters = BuildListFilters(gate_pass=True)

    with pytest.raises(ValueError, match="gate_pass filter requires scenario_id"):
        repo.list_builds(filters=filters)


def test_list_builds_rejects_unknown_sort_field():
    repo = ClickhouseRepository(client=MagicMock())

    with pytest.raises(ValueError):
        repo.list_builds(sort_by="not-a-field")


def test_list_builds_excludes_stale_by_default():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123"}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)
    filters = BuildListFilters()
    repo.list_builds(filters=filters)
    args, _ = client.query.call_args
    assert "b.is_stale = 0" in args[0]


def test_list_builds_include_stale_option():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123"}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)
    filters = BuildListFilters(include_stale=True)
    repo.list_builds(filters=filters)
    args, _ = client.query.call_args
    assert "b.is_stale = 0" not in args[0]


def test_update_build_ruleset_updates_requested_columns():
    client = MagicMock()
    repo = ClickhouseRepository(client=client)
    repo.update_build_ruleset(
        "build-123",
        ruleset_id="ruleset-new",
        is_stale=True,
    )
    client.command.assert_called_once()
    command_args = client.command.call_args
    query = command_args.args[0]
    assert "ruleset_id = {ruleset_id:String}" in query
    assert "is_stale = {is_stale:UInt8}" in query
    params = command_args.kwargs["parameters"]
    assert params["ruleset_id"] == "ruleset-new"
    assert params["is_stale"] == 1


def test_mark_builds_stale_except_updates_all_other_rulesets():
    client = MagicMock()
    repo = ClickhouseRepository(client=client)
    repo.mark_builds_stale_except("current-ruleset")
    client.command.assert_called_once()
    command_args = client.command.call_args
    query = command_args.args[0]
    assert "is_stale = 1" in query
    assert "WHERE ruleset_id != {ruleset_id:String}" in query
    assert command_args.kwargs["parameters"]["ruleset_id"] == "current-ruleset"


def test_mark_builds_stale_except_can_scope_profile():
    client = MagicMock()
    repo = ClickhouseRepository(client=client)
    repo.mark_builds_stale_except("current-ruleset", profile_id="pinnacle")
    client.command.assert_called_once()
    command_args = client.command.call_args
    query = command_args.args[0]
    assert "AND profile_id = {profile_id:String}" in query
    assert command_args.kwargs["parameters"] == {
        "ruleset_id": "current-ruleset",
        "profile_id": "pinnacle",
    }


def test_list_builds_constraint_filters():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123"}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)
    filters = BuildListFilters(
        constraint_status="pass",
        constraint_reason_code="reason",
        violated_constraint="budget_cost",
        constraint_checked_after=datetime(2025, 1, 1),
        constraint_checked_before=datetime(2025, 1, 2),
    )

    repo.list_builds(filters=filters)

    args, kwargs = client.query.call_args
    query = args[0]
    assert "b.constraint_status = {constraint_status:String}" in query
    assert "b.constraint_reason_code = {constraint_reason_code:String}" in query
    assert "has(b.violated_constraints, {violated_constraint:String})" in query
    assert "b.constraint_checked_at >= {constraint_checked_after:DateTime}" in query
    assert "b.constraint_checked_at <= {constraint_checked_before:DateTime}" in query
    params = kwargs["parameters"]
    assert params["constraint_status"] == "pass"
    assert params["constraint_reason_code"] == "reason"
    assert params["violated_constraint"] == "budget_cost"


def test_list_builds_supports_constraint_sorts():
    client = MagicMock()
    dummy = DummyResult([{"build_id": "build-123"}])
    client.query.return_value = dummy
    repo = ClickhouseRepository(client=client)

    for sort_field in ("constraint_status", "constraint_checked_at", "constraint_reason_code"):
        repo.list_builds(sort_by=sort_field)
        query_string = client.query.call_args.args[0]
        assert f"ORDER BY {repo._SORTABLE_FIELDS[sort_field]} DESC" in query_string
        client.query.reset_mock()


def test_update_build_constraints_issues_command():
    client = MagicMock()
    repo = ClickhouseRepository(client=client)
    checked_at = datetime(2025, 1, 5, 12, 0)

    repo.update_build_constraints(
        "build-123",
        constraint_status="fail",
        constraint_reason_code="reason",
        violated_constraints=["rule-a"],
        constraint_checked_at=checked_at,
    )

    client.command.assert_called_once()
    query = client.command.call_args.args[0]
    assert "constraint_status = {constraint_status:String}" in query
    assert "violated_constraints = {violated_constraints:Array(String)}" in query
    params = client.command.call_args.kwargs["parameters"]
    assert params["constraint_status"] == "fail"
    assert params["constraint_reason_code"] == "reason"
    assert params["violated_constraints"] == ["rule-a"]
    assert params["constraint_checked_at"] is checked_at


def test_default_client_reuses_cached_instance(monkeypatch):
    ch_module = importlib.import_module(ClickhouseRepository.__module__)

    ch_module._default_clickhouse_client = None
    shared_client = MagicMock()
    client_factory = MagicMock(return_value=shared_client)
    monkeypatch.setattr(ch_module.clickhouse_connect, "get_client", client_factory)

    repo_a = ClickhouseRepository()
    repo_b = ClickhouseRepository()

    assert repo_a._client is shared_client
    assert repo_b._client is shared_client
    assert repo_a._client is repo_b._client
    assert client_factory.call_count == 1

    ch_module._default_clickhouse_client = None

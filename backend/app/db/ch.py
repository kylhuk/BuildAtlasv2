from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Sequence

import clickhouse_connect
from backend.engine.metrics_source import METRICS_SOURCE_POB
from clickhouse_connect.driver.client import Client
from pydantic import BaseModel, ConfigDict, Field

from backend.app.settings import settings

ClickhouseClient = Client

BUILD_COLUMNS = [
    "build_id",
    "created_at",
    "ruleset_id",
    "profile_id",
    "class",
    "ascendancy",
    "main_skill",
    "damage_type",
    "defence_type",
    "complexity_bucket",
    "pob_xml_path",
    "pob_code_path",
    "genome_path",
    "tags",
    "status",
    "is_stale",
]

SCENARIO_METRIC_COLUMNS = [
    "build_id",
    "ruleset_id",
    "scenario_id",
    "gate_pass",
    "gate_fail_reasons",
    "pob_warnings",
    "evaluated_at",
    "full_dps",
    "max_hit",
    "armour",
    "evasion",
    "life",
    "mana",
    "utility_score",
    "metrics_source",
]

BUILD_COST_COLUMNS = [
    "build_id",
    "ruleset_id",
    "price_snapshot_id",
    "total_cost_chaos",
    "unknown_cost_count",
    "slot_costs_json_path",
    "gem_costs_json_path",
    "calculated_at",
]


class BuildInsertPayload(BaseModel):
    build_id: str
    created_at: datetime
    ruleset_id: str
    profile_id: str
    class_: str = Field(..., alias="class")
    ascendancy: str
    main_skill: str
    damage_type: str
    defence_type: str
    complexity_bucket: str
    pob_xml_path: str
    pob_code_path: str
    genome_path: str
    tags: List[str] = Field(default_factory=list)
    status: str
    is_stale: bool = False

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ScenarioMetricRow(BaseModel):
    build_id: str
    ruleset_id: str
    scenario_id: str
    gate_pass: bool
    gate_fail_reasons: List[str] = Field(default_factory=list)
    pob_warnings: List[str] = Field(default_factory=list)
    evaluated_at: datetime
    full_dps: float
    max_hit: float
    armour: float
    evasion: float
    life: float
    mana: float
    utility_score: float
    metrics_source: str = Field(default=METRICS_SOURCE_POB)

    model_config = ConfigDict(extra="forbid")


class BuildCostRow(BaseModel):
    build_id: str
    ruleset_id: str
    price_snapshot_id: str
    total_cost_chaos: float
    unknown_cost_count: int
    slot_costs_json_path: str
    gem_costs_json_path: str
    calculated_at: datetime

    model_config = ConfigDict(extra="forbid")


class BuildListFilters(BaseModel):
    ruleset_id: str | None = None
    profile_id: str | None = None
    status: str | None = None
    class_: str | None = Field(None, alias="class")
    ascendancy: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    scenario_id: str | None = None
    gate_pass: bool | None = None
    max_cost_chaos: float | None = None
    exclude_unknown_cost: bool | None = None
    price_snapshot_id: str | None = None
    cost_calculated_after: datetime | None = None
    cost_calculated_before: datetime | None = None
    constraint_status: str | None = None
    constraint_reason_code: str | None = None
    violated_constraint: str | None = None
    constraint_checked_after: datetime | None = None
    constraint_checked_before: datetime | None = None
    include_stale: bool = False
    verified_only: bool = False

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def get_clickhouse_client() -> ClickhouseClient:
    return clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_db,
    )


class ClickhouseRepository:
    _SORTABLE_FIELDS: Dict[str, str] = {
        "created_at": "b.created_at",
        "ruleset_id": "b.ruleset_id",
        "profile_id": "b.profile_id",
        "status": "b.status",
        "ascendancy": "b.ascendancy",
        "class": "b.`class`",
        "full_dps": "sm.full_dps",
        "max_hit": "sm.max_hit",
        "total_cost_chaos": "bc.total_cost_chaos",
        "dps_per_chaos": "dps_per_chaos",
        "max_hit_per_chaos": "max_hit_per_chaos",
        "constraint_status": "b.constraint_status",
        "constraint_checked_at": "b.constraint_checked_at",
        "constraint_reason_code": "b.constraint_reason_code",
    }
    _DEFAULT_SORT = "created_at"
    _DEFAULT_LIMIT = 100
    _MEDIUM_PRIORITY_SORT_FIELDS = {"total_cost_chaos", "dps_per_chaos", "max_hit_per_chaos"}

    def __init__(self, client: ClickhouseClient | None = None) -> None:
        self._client = client or get_clickhouse_client()

    def close(self) -> None:
        client = self._client
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
                return
            except Exception:
                pass
        disconnect_fn = getattr(client, "disconnect", None)
        if callable(disconnect_fn):
            try:
                disconnect_fn()
            except Exception:
                pass

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def insert_build(self, payload: BuildInsertPayload) -> None:
        row = [
            payload.build_id,
            payload.created_at,
            payload.ruleset_id,
            payload.profile_id,
            payload.class_,
            payload.ascendancy,
            payload.main_skill,
            payload.damage_type,
            payload.defence_type,
            payload.complexity_bucket,
            payload.pob_xml_path,
            payload.pob_code_path,
            payload.genome_path,
            payload.tags,
            payload.status,
            int(payload.is_stale),
        ]
        self._client.insert(
            table="builds",
            data=[row],
            column_names=BUILD_COLUMNS,
            database=settings.clickhouse_db,
        )

    def insert_scenario_metrics(self, rows: Sequence[ScenarioMetricRow]) -> None:
        if not rows:
            return
        data = []
        for row in rows:
            data.append(
                [
                    row.build_id,
                    row.ruleset_id,
                    row.scenario_id,
                    int(row.gate_pass),
                    row.gate_fail_reasons,
                    row.pob_warnings,
                    row.evaluated_at,
                    row.full_dps,
                    row.max_hit,
                    row.armour,
                    row.evasion,
                    row.life,
                    row.mana,
                    row.utility_score,
                    row.metrics_source,
                ]
            )
        self._client.insert(
            table="scenario_metrics",
            data=data,
            column_names=SCENARIO_METRIC_COLUMNS,
            database=settings.clickhouse_db,
        )

    def insert_build_costs(self, rows: Sequence[BuildCostRow]) -> None:
        if not rows:
            return
        data = []
        for row in rows:
            data.append(
                [
                    row.build_id,
                    row.ruleset_id,
                    row.price_snapshot_id,
                    row.total_cost_chaos,
                    row.unknown_cost_count,
                    row.slot_costs_json_path,
                    row.gem_costs_json_path,
                    row.calculated_at,
                ]
            )
        self._client.insert(
            table="build_costs",
            data=data,
            column_names=BUILD_COST_COLUMNS,
            database=settings.clickhouse_db,
        )

    def fetch_scenario_metric_rows(self, ruleset_id: str) -> List[Dict[str, Any]]:
        query = (
            "SELECT scenario_id, gate_pass, full_dps, max_hit, utility_score "
            "FROM scenario_metrics "
            "WHERE ruleset_id = {ruleset_id:String}"
        )
        result = self._client.query(query, parameters={"ruleset_id": ruleset_id})
        try:
            rows = list(result.named_results())
        finally:
            result.close()
        return rows

    def get_build(self, build_id: str) -> Dict[str, Any] | None:
        query = "SELECT * FROM builds WHERE build_id = {build_id:String} LIMIT 1"
        result = self._client.query(query, parameters={"build_id": build_id})
        rows = list(result.named_results())
        result.close()
        return rows[0] if rows else None

    def list_builds(
        self,
        filters: BuildListFilters | None = None,
        sort_by: str | None = None,
        sort_dir: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> List[Dict[str, Any]]:
        sort_key = sort_by or self._DEFAULT_SORT
        if sort_key not in self._SORTABLE_FIELDS:
            raise ValueError(f"Sort field '{sort_key}' not allowed")
        order_dir = (sort_dir or "desc").lower()
        if order_dir not in {"asc", "desc"}:
            raise ValueError("sort_dir must be 'asc' or 'desc'")
        limit_value = limit or self._DEFAULT_LIMIT
        offset_value = offset or 0
        if limit_value <= 0:
            raise ValueError("limit must be positive")
        if offset_value < 0:
            raise ValueError("offset must be non-negative")

        active_filters = filters if filters is not None else BuildListFilters.model_construct()
        scenario_sort_fields = {"full_dps", "max_hit", "dps_per_chaos", "max_hit_per_chaos"}
        if sort_key in scenario_sort_fields and not active_filters.scenario_id:
            raise ValueError("sorting by metric fields requires scenario_id")
        if sort_key in self._MEDIUM_PRIORITY_SORT_FIELDS and limit_value > 100:
            raise ValueError("sorting by medium-priority fields requires limit <= 100")
        if sort_key in self._MEDIUM_PRIORITY_SORT_FIELDS and not any(
            [
                active_filters.ruleset_id,
                active_filters.profile_id,
                active_filters.class_,
                active_filters.ascendancy,
                active_filters.scenario_id,
                active_filters.price_snapshot_id,
            ]
        ):
            raise ValueError(
                "sorting by medium-priority fields requires a narrowing filter "
                "(ruleset_id/profile_id/class/ascendancy/scenario_id/price_snapshot_id)"
            )
        if active_filters.gate_pass is not None and not active_filters.scenario_id:
            raise ValueError("gate_pass filter requires scenario_id")
        scenario_join_needed = bool(active_filters.scenario_id)
        where_clauses: List[str] = []
        parameters: Dict[str, Any] = {
            "limit": limit_value,
            "offset": offset_value,
        }
        if not active_filters.include_stale:
            where_clauses.append("b.is_stale = 0")
        if active_filters.ruleset_id:
            where_clauses.append("b.ruleset_id = {ruleset_id:String}")
            parameters["ruleset_id"] = active_filters.ruleset_id
        if active_filters.profile_id:
            where_clauses.append("b.profile_id = {profile_id:String}")
            parameters["profile_id"] = active_filters.profile_id
        if active_filters.status:
            where_clauses.append("b.status = {status:String}")
            parameters["status"] = active_filters.status
        if active_filters.verified_only:
            where_clauses.append("b.status IN ('evaluated', 'failed')")
        if active_filters.class_:
            where_clauses.append("b.`class` = {class:String}")
            parameters["class"] = active_filters.class_
        if active_filters.ascendancy:
            where_clauses.append("b.ascendancy = {ascendancy:String}")
            parameters["ascendancy"] = active_filters.ascendancy
        if active_filters.created_after:
            where_clauses.append("b.created_at >= {created_after:DateTime}")
            parameters["created_after"] = active_filters.created_after
        if active_filters.created_before:
            where_clauses.append("b.created_at <= {created_before:DateTime}")
            parameters["created_before"] = active_filters.created_before
        if active_filters.scenario_id:
            where_clauses.append("sm.scenario_id = {scenario_id:String}")
            parameters["scenario_id"] = active_filters.scenario_id
        if active_filters.gate_pass is not None:
            where_clauses.append("sm.gate_pass = {gate_pass:UInt8}")
            parameters["gate_pass"] = int(active_filters.gate_pass)
        if active_filters.max_cost_chaos is not None:
            where_clauses.append("bc.total_cost_chaos <= {max_cost_chaos:Float64}")
            parameters["max_cost_chaos"] = active_filters.max_cost_chaos
        if active_filters.exclude_unknown_cost:
            where_clauses.append("bc.unknown_cost_count = 0")
        if active_filters.price_snapshot_id:
            where_clauses.append("bc.price_snapshot_id = {price_snapshot_id:String}")
            parameters["price_snapshot_id"] = active_filters.price_snapshot_id
        if active_filters.cost_calculated_after:
            where_clauses.append("bc.calculated_at >= {cost_calculated_after:DateTime}")
            parameters["cost_calculated_after"] = active_filters.cost_calculated_after
        if active_filters.cost_calculated_before:
            where_clauses.append("bc.calculated_at <= {cost_calculated_before:DateTime}")
            parameters["cost_calculated_before"] = active_filters.cost_calculated_before
        if active_filters.constraint_status:
            where_clauses.append("b.constraint_status = {constraint_status:String}")
            parameters["constraint_status"] = active_filters.constraint_status
        if active_filters.constraint_reason_code:
            where_clauses.append("b.constraint_reason_code = {constraint_reason_code:String}")
            parameters["constraint_reason_code"] = active_filters.constraint_reason_code
        if active_filters.violated_constraint:
            where_clauses.append("has(b.violated_constraints, {violated_constraint:String})")
            parameters["violated_constraint"] = active_filters.violated_constraint
        if active_filters.constraint_checked_after:
            where_clauses.append("b.constraint_checked_at >= {constraint_checked_after:DateTime}")
            parameters["constraint_checked_after"] = active_filters.constraint_checked_after
        if active_filters.constraint_checked_before:
            where_clauses.append("b.constraint_checked_at <= {constraint_checked_before:DateTime}")
            parameters["constraint_checked_before"] = active_filters.constraint_checked_before

        where_clause = f"WHERE {' AND '.join(where_clauses)} " if where_clauses else ""
        sort_column = self._SORTABLE_FIELDS[sort_key]
        scenario_join_sql = ""
        if scenario_join_needed:
            scenario_join_sql = (
                "LEFT JOIN ("
                "SELECT "
                "build_id, ruleset_id, scenario_id, "
                "argMax(gate_pass, evaluated_at) AS gate_pass, "
                "argMax(full_dps, evaluated_at) AS full_dps, "
                "argMax(max_hit, evaluated_at) AS max_hit "
                "FROM scenario_metrics "
                "WHERE scenario_id = {scenario_id:String} "
                "GROUP BY build_id, ruleset_id, scenario_id"
                ") AS sm ON b.build_id = sm.build_id AND b.ruleset_id = sm.ruleset_id "
            )
        cost_join_sql = (
            "LEFT JOIN ("
            "SELECT "
            "build_id, ruleset_id, price_snapshot_id, total_cost_chaos, "
            "unknown_cost_count, slot_costs_json_path, gem_costs_json_path, calculated_at "
            "FROM build_costs "
            "ORDER BY calculated_at DESC "
            "LIMIT 1 BY build_id, ruleset_id"
            ") AS bc ON b.build_id = bc.build_id AND b.ruleset_id = bc.ruleset_id "
        )
        ratio_columns = (
            (
                "if(bc.total_cost_chaos <= 0 OR isNull(bc.total_cost_chaos), NULL, "
                "sm.full_dps / bc.total_cost_chaos) AS dps_per_chaos, "
                "if(bc.total_cost_chaos <= 0 OR isNull(bc.total_cost_chaos), NULL, "
                "sm.max_hit / bc.total_cost_chaos) AS max_hit_per_chaos"
            )
            if scenario_join_needed
            else "NULL AS dps_per_chaos, NULL AS max_hit_per_chaos"
        )
        select_prefix = (
            "SELECT b.*, bc.price_snapshot_id, bc.total_cost_chaos, bc.unknown_cost_count"
        )
        base_query = (
            f"{select_prefix}, {ratio_columns} FROM builds AS b {scenario_join_sql}{cost_join_sql}"
        )
        query = (
            f"{base_query}"
            f"{where_clause}"
            f"ORDER BY {sort_column} {order_dir.upper()} "
            "LIMIT {limit:UInt32} OFFSET {offset:UInt32}"
        )
        result = self._client.query(query, parameters=parameters)
        try:
            rows = list(result.named_results())
        finally:
            result.close()
        return rows

    def build_inventory_stats(self) -> Dict[str, Any]:
        total_query = (
            "SELECT count() AS total_builds, countIf(is_stale = 1) AS stale_builds FROM builds"
        )
        total_result = self._client.query(total_query)
        try:
            total_rows = list(total_result.named_results())
        finally:
            total_result.close()

        status_query = "SELECT status, count() AS count FROM builds GROUP BY status"
        status_result = self._client.query(status_query)
        try:
            status_rows = list(status_result.named_results())
        finally:
            status_result.close()

        total_builds = int(total_rows[0].get("total_builds", 0)) if total_rows else 0
        stale_builds = int(total_rows[0].get("stale_builds", 0)) if total_rows else 0
        status_counts: Dict[str, int] = {}
        for row in status_rows:
            status = str(row.get("status") or "unknown")
            status_counts[status] = int(row.get("count") or 0)

        return {
            "total_builds": total_builds,
            "stale_builds": stale_builds,
            "status_counts": status_counts,
        }

    def update_build_status(self, build_id: str, status: str) -> None:
        query = (
            "ALTER TABLE builds UPDATE status = {status:String} WHERE build_id = {build_id:String}"
        )
        self._client.command(query, parameters={"build_id": build_id, "status": status})

    def update_build_constraints(
        self,
        build_id: str,
        constraint_status: str | None = None,
        constraint_reason_code: str | None = None,
        violated_constraints: Sequence[str] | None = None,
        constraint_checked_at: datetime | None = None,
    ) -> None:
        updates: list[str] = []
        parameters: Dict[str, Any] = {"build_id": build_id}
        if constraint_status is not None:
            updates.append("constraint_status = {constraint_status:String}")
            parameters["constraint_status"] = constraint_status
        if constraint_reason_code is not None:
            updates.append("constraint_reason_code = {constraint_reason_code:String}")
            parameters["constraint_reason_code"] = constraint_reason_code
        if violated_constraints is not None:
            updates.append("violated_constraints = {violated_constraints:Array(String)}")
            parameters["violated_constraints"] = list(violated_constraints)
        if constraint_checked_at is not None:
            updates.append("constraint_checked_at = {constraint_checked_at:DateTime}")
            parameters["constraint_checked_at"] = constraint_checked_at
        if not updates:
            return
        query = (
            "ALTER TABLE builds UPDATE "
            + ", ".join(updates)
            + " WHERE build_id = {build_id:String}"
        )
        self._client.command(query, parameters=parameters)

    def update_build_ruleset(
        self,
        build_id: str,
        ruleset_id: str | None = None,
        is_stale: bool | None = None,
    ) -> None:
        updates: list[str] = []
        parameters: Dict[str, Any] = {"build_id": build_id}
        if ruleset_id is not None:
            updates.append("ruleset_id = {ruleset_id:String}")
            parameters["ruleset_id"] = ruleset_id
        if is_stale is not None:
            updates.append("is_stale = {is_stale:UInt8}")
            parameters["is_stale"] = int(is_stale)
        if not updates:
            return
        query = (
            "ALTER TABLE builds UPDATE "
            + ", ".join(updates)
            + " WHERE build_id = {build_id:String}"
        )
        self._client.command(query, parameters=parameters)

    def mark_builds_stale_except(self, ruleset_id: str, profile_id: str | None = None) -> None:
        query = "ALTER TABLE builds UPDATE is_stale = 1 WHERE ruleset_id != {ruleset_id:String}"
        parameters: Dict[str, Any] = {"ruleset_id": ruleset_id}
        if profile_id:
            query += " AND profile_id = {profile_id:String}"
            parameters["profile_id"] = profile_id
        self._client.command(query, parameters=parameters)

    def purge_build(self, build_id: str) -> None:
        parameters: Dict[str, Any] = {"build_id": build_id}
        settings: Dict[str, Any] = {"mutations_sync": 1}
        for table in ("scenario_metrics", "build_costs", "builds"):
            query = f"ALTER TABLE {table} DELETE WHERE build_id = {{build_id:String}}"
            self._client.command(query, parameters=parameters, settings=settings)

    def list_scenario_metrics(self, build_id: str) -> List[Dict[str, Any]]:
        query = (
            "SELECT * FROM scenario_metrics WHERE build_id = {build_id:String} "
            "ORDER BY scenario_id ASC, evaluated_at DESC"
        )
        result = self._client.query(query, parameters={"build_id": build_id})
        try:
            rows = list(result.named_results())
        finally:
            result.close()
        return rows

    def get_latest_build_cost(self, build_id: str) -> Dict[str, Any] | None:
        query = (
            "SELECT * FROM build_costs WHERE build_id = {build_id:String} "
            "ORDER BY calculated_at DESC LIMIT 1"
        )
        result = self._client.query(query, parameters={"build_id": build_id})
        try:
            rows = list(result.named_results())
        finally:
            result.close()
        return rows[0] if rows else None

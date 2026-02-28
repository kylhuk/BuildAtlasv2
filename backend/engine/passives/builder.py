from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from ..genome import CLASS_ASCENDANCIES, DEFENSE_ARCHETYPES, GenomeV0

GRAPH_SCHEMA_VERSION = "v0"
TARGETS_SCHEMA_VERSION = "v0"

GRAPH_DATA_FILE = Path(__file__).parent / "tree_graph_v0.json"
TARGETS_DATA_FILE = Path(__file__).parent / "targets_v0.json"


class PassiveTreeError(ValueError):
    pass


class PassiveTreeGraphValidationError(PassiveTreeError):
    pass


class PassiveTreeTargetsValidationError(PassiveTreeError):
    pass


class PassiveTreePlanError(PassiveTreeError):
    pass


@dataclass(frozen=True)
class PassiveGraphNode:
    id: str
    kind: str
    pob_id: str | None = None


@dataclass(frozen=True)
class PassiveGraphEdge:
    from_node: str
    to_node: str


@dataclass(frozen=True)
class PassiveTreeGraph:
    schema_version: str
    nodes: Mapping[str, PassiveGraphNode]
    edges: Tuple[PassiveGraphEdge, ...]
    start_nodes: Mapping[str, str]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PassiveTreeGraph":
        schema_version = payload.get("schema_version")
        if schema_version != GRAPH_SCHEMA_VERSION:
            raise PassiveTreeGraphValidationError(
                f"unsupported graph schema_version={schema_version}"
            )

        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list):
            raise PassiveTreeGraphValidationError("graph nodes must be a list")
        nodes: Dict[str, PassiveGraphNode] = {}
        for entry in raw_nodes:
            if not isinstance(entry, Mapping):
                raise PassiveTreeGraphValidationError("node entries must be mappings")
            node_id = entry.get("id")
            kind = entry.get("kind")
            if not isinstance(node_id, str) or not isinstance(kind, str):
                raise PassiveTreeGraphValidationError("nodes must define id and kind")
            if node_id in nodes:
                raise PassiveTreeGraphValidationError(f"duplicate node id={node_id}")
            pob_id = entry.get("pob_id")
            if pob_id is not None and not isinstance(pob_id, str):
                raise PassiveTreeGraphValidationError("pob_id must be a string")
            nodes[node_id] = PassiveGraphNode(id=node_id, kind=kind, pob_id=pob_id)

        raw_edges = payload.get("edges", [])
        if not isinstance(raw_edges, list):
            raise PassiveTreeGraphValidationError("graph edges must be a list")
        edges: list[PassiveGraphEdge] = []
        for entry in raw_edges:
            if not isinstance(entry, Mapping):
                raise PassiveTreeGraphValidationError("edge entries must be mappings")
            from_node = entry.get("from")
            to_node = entry.get("to")
            if not isinstance(from_node, str) or not isinstance(to_node, str):
                raise PassiveTreeGraphValidationError("edges must define from/to ids")
            if from_node not in nodes or to_node not in nodes:
                raise PassiveTreeGraphValidationError(
                    f"edge references unknown node from={from_node} to={to_node}"
                )
            edges.append(PassiveGraphEdge(from_node=from_node, to_node=to_node))

        raw_start = payload.get("start_nodes")
        if not isinstance(raw_start, Mapping):
            raise PassiveTreeGraphValidationError("start_nodes must be a mapping")
        start_nodes: Dict[str, str] = {}
        for class_name, node_id in raw_start.items():
            if not isinstance(class_name, str) or not isinstance(node_id, str):
                raise PassiveTreeGraphValidationError("start node entries must be strings")
            if node_id not in nodes:
                raise PassiveTreeGraphValidationError(
                    f"start node {node_id} for class {class_name} is undefined"
                )
            start_nodes[class_name] = node_id

        missing = set(CLASS_ASCENDANCIES) - set(start_nodes)
        if missing:
            raise PassiveTreeGraphValidationError(
                f"missing start nodes for classes={sorted(missing)}"
            )

        return PassiveTreeGraph(
            schema_version=schema_version,
            nodes=nodes,
            edges=tuple(edges),
            start_nodes=start_nodes,
        )

    def adjacency(self) -> Dict[str, list[str]]:
        neighbors: Dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            neighbors[edge.from_node].append(edge.to_node)
            neighbors[edge.to_node].append(edge.from_node)
        for node_neighbors in neighbors.values():
            node_neighbors.sort()
        return neighbors


@dataclass(frozen=True)
class PassiveTargetSet:
    required: Tuple[str, ...]
    desired: Tuple[str, ...]


@dataclass(frozen=True)
class PassiveTreeTargets:
    schema_version: str
    defense_archetypes: Mapping[str, PassiveTargetSet]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PassiveTreeTargets":
        schema_version = payload.get("schema_version")
        if schema_version != TARGETS_SCHEMA_VERSION:
            raise PassiveTreeTargetsValidationError(
                f"unsupported targets schema_version={schema_version}"
            )

        raw_arch = payload.get("defense_archetypes")
        if not isinstance(raw_arch, Mapping):
            raise PassiveTreeTargetsValidationError("defense_archetypes must be a mapping")
        defense_map: Dict[str, PassiveTargetSet] = {}
        for archetype, entry in raw_arch.items():
            if archetype not in DEFENSE_ARCHETYPES:
                raise PassiveTreeTargetsValidationError(
                    f"unsupported defense_archetype={archetype}"
                )
            if not isinstance(entry, Mapping):
                raise PassiveTreeTargetsValidationError("target entry must be mapping")
            required = _parse_node_list(entry.get("required", []))
            desired = _parse_node_list(entry.get("desired", []))
            defense_map[archetype] = PassiveTargetSet(required=required, desired=desired)
        return PassiveTreeTargets(schema_version=schema_version, defense_archetypes=defense_map)

    def validate_against_graph(self, graph: PassiveTreeGraph) -> None:
        node_ids = set(graph.nodes)
        for target_set in self.defense_archetypes.values():
            for node_id in (*target_set.required, *target_set.desired):
                if node_id not in node_ids:
                    raise PassiveTreeTargetsValidationError(
                        f"target node {node_id} missing from graph"
                    )


@dataclass(frozen=True)
class PassiveTreePlan:
    genome: GenomeV0
    nodes: Tuple[PassiveGraphNode, ...]
    required_targets: Tuple[str, ...]

    @property
    def node_ids(self) -> Tuple[str, ...]:
        return tuple(node.id for node in self.nodes)


def load_default_passive_tree_graph() -> PassiveTreeGraph:
    payload = _read_json_file(GRAPH_DATA_FILE)
    return PassiveTreeGraph.from_dict(payload)


def load_default_passive_tree_targets(graph: PassiveTreeGraph | None = None) -> PassiveTreeTargets:
    payload = _read_json_file(TARGETS_DATA_FILE)
    targets = PassiveTreeTargets.from_dict(payload)
    if graph is not None:
        targets.validate_against_graph(graph)
    return targets


def build_passive_tree_plan(
    genome: GenomeV0,
    point_budget: int,
    graph: PassiveTreeGraph | None = None,
    targets: PassiveTreeTargets | None = None,
) -> PassiveTreePlan:
    if point_budget <= 0:
        raise PassiveTreePlanError("point_budget must be positive")
    graph = graph or load_default_passive_tree_graph()
    targets = targets or load_default_passive_tree_targets(graph)
    targets.validate_against_graph(graph)

    target_set = targets.defense_archetypes.get(genome.defense_archetype)
    if not target_set:
        raise PassiveTreePlanError(
            f"no targets defined for defense_archetype={genome.defense_archetype}"
        )

    start_node = graph.start_nodes.get(genome.class_name)
    if not start_node:
        raise PassiveTreePlanError(f"no start node defined for class={genome.class_name}")

    adjacency = graph.adjacency()
    required_targets = tuple(sorted(set(target_set.required)))
    plan_nodes: list[str] = []
    seen: set[str] = set()

    if not required_targets:
        plan_nodes.append(start_node)
        seen.add(start_node)
    else:
        for target in required_targets:
            path = _shortest_path_bfs(start_node, target, adjacency)
            for node_id in path:
                if node_id not in seen:
                    seen.add(node_id)
                    plan_nodes.append(node_id)
    if not plan_nodes:
        plan_nodes.append(start_node)
        seen.add(start_node)

    if len(plan_nodes) > point_budget:
        raise PassiveTreePlanError(
            f"plan requires {len(plan_nodes)} nodes but budget is {point_budget}"
        )

    plan_nodes_tuple = tuple(graph.nodes[node_id] for node_id in plan_nodes)
    return PassiveTreePlan(genome=genome, nodes=plan_nodes_tuple, required_targets=required_targets)


def export_passive_tree_plan(plan: PassiveTreePlan) -> Dict[str, Any]:
    node_refs = sorted(node.pob_id if node.pob_id is not None else node.id for node in plan.nodes)
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "class": plan.genome.class_name,
        "defense_archetype": plan.genome.defense_archetype,
        "node_ids": node_refs,
    }


def _parse_node_list(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise PassiveTreeTargetsValidationError("target nodes must be a list of strings")
    nodes: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise PassiveTreeTargetsValidationError("target node ids must be strings")
        nodes.append(entry)
    return tuple(nodes)


def _shortest_path_bfs(
    start: str,
    target: str,
    adjacency: Mapping[str, Sequence[str]],
) -> list[str]:
    if start == target:
        return [start]
    queue = deque([(start, [start])])
    seen: set[str] = {start}
    while queue:
        current, path = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor in seen:
                continue
            if neighbor == target:
                return path + [neighbor]
            seen.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    raise PassiveTreePlanError(f"unable to reach target node {target} from start {start}")


def _read_json_file(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)

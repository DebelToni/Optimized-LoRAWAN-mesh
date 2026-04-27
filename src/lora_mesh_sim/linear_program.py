from __future__ import annotations

import math
from collections import defaultdict
from typing import cast

import pulp

from .models import LinkMetric, Node, OptimizationResult


class OptimizationSettings:
    def __init__(
        self,
        min_battery: float = 0.24,
        min_trust: float = 0.40,
        nodes_per_relay: int = 14,
        max_children: int = 20,
        max_relays_per_zone: int = 3,
    ) -> None:
        self.min_battery = min_battery
        self.min_trust = min_trust
        self.nodes_per_relay = nodes_per_relay
        self.max_children = max_children
        self.max_relays_per_zone = max_relays_per_zone


def optimize_mesh(
    nodes: list[Node],
    gateway: Node,
    links: dict[tuple[int, int], LinkMetric],
    settings: OptimizationSettings | None = None,
) -> OptimizationResult:
    settings = settings or OptimizationSettings()
    notes: list[str] = []
    zone_targets = _zone_targets(nodes, settings)
    duplex_links = _duplex_links(links)

    try:
        result = _solve_milp(nodes, gateway, duplex_links, zone_targets, settings)
        notes.extend(result.notes)
        return result
    except Exception as exc:  # pragma: no cover - runtime safety fallback
        notes.append(f"MILP fallback: {exc}")
        return _heuristic_solution(nodes, gateway, duplex_links, zone_targets, settings, notes)


def _solve_milp(
    nodes: list[Node],
    gateway: Node,
    links: dict[tuple[int, int], LinkMetric],
    zone_targets: dict[str, int],
    settings: OptimizationSettings,
) -> OptimizationResult:
    node_ids = [node.node_id for node in nodes]
    nodes_by_id = {node.node_id: node for node in nodes}
    candidates = {
        node.node_id
        for node in nodes
        if (not node.is_battery_powered or node.battery_level >= 0.10)
    }

    parent_candidates: dict[int, list[int]] = defaultdict(list)
    direct_gateway: set[int] = set()
    for node in nodes:
        if (node.node_id, gateway.node_id) in links:
            direct_gateway.add(node.node_id)
        for other in nodes:
            if node.node_id == other.node_id:
                continue
            if other.node_id not in candidates:
                continue
            if (node.node_id, other.node_id) in links:
                parent_candidates[node.node_id].append(other.node_id)

    if any(not parent_candidates[node_id] and node_id not in direct_gateway for node_id in node_ids):
        return _heuristic_solution(nodes, gateway, links, zone_targets, settings, ["Sparse duplex links forced heuristic routing."])

    problem = pulp.LpProblem("sofia_mesh_relay_selection", pulp.LpMinimize)
    relay_var = {
        node_id: pulp.LpVariable(f"relay_{node_id}", cat="Binary")
        for node_id in candidates
    }
    assign_gateway_var = {
        node_id: pulp.LpVariable(f"gw_{node_id}", cat="Binary")
        for node_id in node_ids
        if node_id in direct_gateway
    }
    assign_relay_var = {
        (node_id, relay_id): pulp.LpVariable(f"edge_{node_id}_{relay_id}", cat="Binary")
        for node_id, relay_ids in parent_candidates.items()
        for relay_id in relay_ids
    }

    problem += (
        pulp.lpSum(_relay_cost(nodes_by_id[node_id]) * var for node_id, var in relay_var.items())
        + pulp.lpSum(_link_cost(links[(node_id, gateway.node_id)], gateway=True) * var for node_id, var in assign_gateway_var.items())
        + pulp.lpSum(
            _link_cost(links[(node_id, relay_id)], gateway=False) * var
            for (node_id, relay_id), var in assign_relay_var.items()
        )
    )

    for node_id in node_ids:
        terms = []
        if node_id in assign_gateway_var:
            terms.append(assign_gateway_var[node_id])
        for relay_id in parent_candidates.get(node_id, []):
            terms.append(assign_relay_var[(node_id, relay_id)])
        problem += pulp.lpSum(terms) == 1, f"one_parent_{node_id}"

    for (node_id, relay_id), var in assign_relay_var.items():
        problem += var <= relay_var[relay_id], f"selected_relay_{node_id}_{relay_id}"

    for relay_id in candidates:
        incoming = [
            assign_relay_var[(node_id, relay_id)]
            for node_id in node_ids
            if (node_id, relay_id) in assign_relay_var
        ]
        if incoming:
            problem += pulp.lpSum(incoming) <= settings.max_children * relay_var[relay_id], f"fanout_{relay_id}"

    zone_members: dict[str, list[int]] = defaultdict(list)
    eligible_zone_members: dict[str, list[int]] = defaultdict(list)
    for node in nodes:
        zone_members[node.zone].append(node.node_id)
        if node.node_id in candidates:
            eligible_zone_members[node.zone].append(node.node_id)

    for zone, target in zone_targets.items():
        eligible_ids = eligible_zone_members.get(zone, [])
        if not eligible_ids:
            continue
        bounded_target = min(target, len(eligible_ids))
        zone_sum = pulp.lpSum(relay_var[node_id] for node_id in eligible_ids)
        problem += zone_sum >= 1, f"zone_min_{zone}"
        problem += zone_sum <= bounded_target, f"zone_max_{zone}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = problem.solve(solver)
    status = pulp.LpStatus.get(status_code, "Unknown")
    if status not in {"Optimal", "Feasible"}:
        return _heuristic_solution(nodes, gateway, links, zone_targets, settings, [f"MILP status {status}."])

    raw_selected_relays = {
        node_id
        for node_id, var in relay_var.items()
        if (var_value := var.value()) is not None and var_value > 0.5
    }
    if not raw_selected_relays:
        return _heuristic_solution(nodes, gateway, links, zone_targets, settings, ["MILP returned no relays."])

    backbone_parent, backbone_notes = _build_backbone(raw_selected_relays, gateway.node_id, nodes_by_id, links)
    reachable_relays = set(backbone_parent)
    notes = list(backbone_notes)
    if not reachable_relays:
        return _heuristic_solution(nodes, gateway, links, zone_targets, settings, ["Relay backbone could not reach gateway."])

    zone_relays: dict[str, list[int]] = defaultdict(list)
    for relay_id in sorted(reachable_relays):
        zone_relays[nodes_by_id[relay_id].zone].append(relay_id)

    assignment_by_node: dict[int, int] = {}
    for node in nodes:
        if node.node_id in reachable_relays:
            continue
        choice = _choose_assigned_parent(node, reachable_relays, gateway.node_id, links)
        if choice is None:
            promoted = _promote_bridge_relay(
                node.node_id,
                raw_selected_relays | set(node_ids),
                reachable_relays,
                gateway.node_id,
                links,
            )
            if promoted is not None:
                relay_id, parent_id = promoted
                if relay_id not in reachable_relays:
                    reachable_relays.add(relay_id)
                    backbone_parent[relay_id] = parent_id
                    zone_relays[nodes_by_id[relay_id].zone].append(relay_id)
                if node.node_id == relay_id:
                    continue
                choice = relay_id
        if choice is None:
            return _heuristic_solution(nodes, gateway, links, zone_targets, settings, [f"Node {node.node_id} could not be assigned to the optimized backbone."])
        assignment_by_node[node.node_id] = choice

    parent_by_node = {**backbone_parent, **assignment_by_node}

    objective_expr = problem.objective
    if objective_expr is None:
        objective_value = 0.0
    else:
        objective_raw = objective_expr.value()
        objective_value = 0.0 if objective_raw is None else cast(float, objective_raw)

    return OptimizationResult(
        selected_relays=reachable_relays,
        parent_by_node=parent_by_node,
        assignment_by_node=assignment_by_node,
        zone_targets=zone_targets,
        zone_relays={zone: sorted(relays) for zone, relays in zone_relays.items()},
        objective_value=objective_value,
        status=status,
        notes=notes,
    )


def _heuristic_solution(
    nodes: list[Node],
    gateway: Node,
    links: dict[tuple[int, int], LinkMetric],
    zone_targets: dict[str, int],
    settings: OptimizationSettings,
    notes: list[str] | None = None,
) -> OptimizationResult:
    notes = list(notes or [])
    nodes_by_id = {node.node_id: node for node in nodes}
    zone_nodes: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        zone_nodes[node.zone].append(node)

    backbone_parent, reachable_relays, zone_relays, connected_notes = _connected_zone_backbone(
        zone_nodes,
        zone_targets,
        gateway.node_id,
        links,
        settings,
    )
    notes.extend(connected_notes)
    if not reachable_relays:
        notes.append("Heuristic could not grow a connected relay backbone.")

    assignment_by_node: dict[int, int] = {}
    for node in nodes:
        if node.node_id in reachable_relays:
            continue
        choice = _choose_assigned_parent(node, reachable_relays, gateway.node_id, links)
        if choice is None and (node.node_id, gateway.node_id) in links:
            choice = gateway.node_id
        if choice is not None:
            assignment_by_node[node.node_id] = choice

    objective = 0.0
    for relay_id in reachable_relays:
        objective += _relay_cost(nodes_by_id[relay_id])
    for child_id, parent_id in assignment_by_node.items():
        objective += _link_cost(links[(child_id, parent_id)], gateway=parent_id == gateway.node_id)
    for child_id, parent_id in backbone_parent.items():
        objective += _link_cost(links[(child_id, parent_id)], gateway=parent_id == gateway.node_id)

    return OptimizationResult(
        selected_relays=reachable_relays,
        parent_by_node={**backbone_parent, **assignment_by_node},
        assignment_by_node=assignment_by_node,
        zone_targets=zone_targets,
        zone_relays=zone_relays,
        objective_value=objective,
        status="Fallback",
        used_fallback=True,
        notes=notes,
    )


def _connected_zone_backbone(
    zone_nodes: dict[str, list[Node]],
    zone_targets: dict[str, int],
    gateway_id: int,
    links: dict[tuple[int, int], LinkMetric],
    settings: OptimizationSettings,
) -> tuple[dict[int, int], set[int], dict[str, list[int]], list[str]]:
    notes: list[str] = []
    selected_relays: set[int] = set()
    backbone_parent: dict[int, int] = {}
    zone_relays: dict[str, list[int]] = defaultdict(list)
    zone_candidates = {
        zone: sorted(
            [node for node in members if _is_relay_eligible(node, settings)],
            key=_relay_score,
            reverse=True,
        )
        for zone, members in zone_nodes.items()
    }

    ordered_zones = sorted(
        zone_candidates,
        key=lambda zone: min(
            [_link_cost(links[(node.node_id, gateway_id)], gateway=True) for node in zone_candidates[zone] if (node.node_id, gateway_id) in links]
            or [999.0]
        ),
    )

    progress = True
    while progress:
        progress = False
        for zone in ordered_zones:
            candidates = zone_candidates[zone]
            target = min(zone_targets.get(zone, 1), len(candidates))
            if target <= 0 or len(zone_relays[zone]) >= target:
                continue
            scored_options: list[tuple[float, int, int]] = []
            for node in candidates:
                if node.node_id in selected_relays:
                    continue
                parent_id = _best_connected_parent(node.node_id, selected_relays, gateway_id, links)
                if parent_id is None:
                    continue
                score = _relay_score(node) - 0.55 * _link_cost(links[(node.node_id, parent_id)], gateway=parent_id == gateway_id)
                scored_options.append((score, node.node_id, parent_id))
            if not scored_options:
                continue
            _score, relay_id, parent_id = max(scored_options, key=lambda item: item[0])
            selected_relays.add(relay_id)
            backbone_parent[relay_id] = parent_id
            zone_relays[zone].append(relay_id)
            progress = True

    for zone, target in zone_targets.items():
        if len(zone_relays.get(zone, [])) < min(target, len(zone_candidates.get(zone, []))):
            notes.append(f"Zone {zone} could not reach its relay target with connected candidates.")

    return backbone_parent, selected_relays, dict(zone_relays), notes


def _best_connected_parent(
    relay_id: int,
    connected_relays: set[int],
    gateway_id: int,
    links: dict[tuple[int, int], LinkMetric],
) -> int | None:
    options: list[tuple[float, int]] = []
    if (relay_id, gateway_id) in links:
        options.append((_link_cost(links[(relay_id, gateway_id)], gateway=True), gateway_id))
    for upstream_id in connected_relays:
        if upstream_id == relay_id:
            continue
        if (relay_id, upstream_id) in links:
            options.append((_link_cost(links[(relay_id, upstream_id)], gateway=False) + 0.25, upstream_id))
    if not options:
        return None
    return min(options, key=lambda item: item[0])[1]


def _promote_bridge_relay(
    node_id: int,
    relay_pool: set[int],
    connected_relays: set[int],
    gateway_id: int,
    links: dict[tuple[int, int], LinkMetric],
) -> tuple[int, int] | None:
    bridge_options: list[tuple[float, int, int]] = []
    for relay_id in relay_pool:
        if relay_id in connected_relays:
            continue
        if relay_id != node_id and (node_id, relay_id) not in links:
            continue
        parent_id = _best_connected_parent(relay_id, connected_relays, gateway_id, links)
        if parent_id is None:
            continue
        cost = _link_cost(links[(relay_id, parent_id)], gateway=parent_id == gateway_id)
        if relay_id != node_id:
            cost += _link_cost(links[(node_id, relay_id)], gateway=False)
        bridge_options.append((cost, relay_id, parent_id))
    if not bridge_options:
        return None
    _cost, relay_id, parent_id = min(bridge_options, key=lambda item: item[0])
    return relay_id, parent_id


def _duplex_links(links: dict[tuple[int, int], LinkMetric]) -> dict[tuple[int, int], LinkMetric]:
    return {
        key: metric
        for key, metric in links.items()
        if metric.link_quality >= 0.10 and metric.reliability >= 0.35
    }


def _build_backbone(
    selected_relays: set[int],
    gateway_id: int,
    nodes_by_id: dict[int, Node],
    links: dict[tuple[int, int], LinkMetric],
) -> tuple[dict[int, int], list[str]]:
    notes: list[str] = []
    if not selected_relays:
        return {}, ["No selected relays for backbone build."]

    ordered_relays = sorted(
        selected_relays,
        key=lambda relay_id: (
            links[(relay_id, gateway_id)].distance_m if (relay_id, gateway_id) in links else math.inf,
            -_relay_score(nodes_by_id[relay_id]),
        ),
    )

    parent: dict[int, int] = {}
    connected = {gateway_id}
    pending = ordered_relays[:]
    while pending:
        progress = False
        next_pending: list[int] = []
        for relay_id in pending:
            options: list[tuple[float, int]] = []
            if (relay_id, gateway_id) in links:
                options.append((_link_cost(links[(relay_id, gateway_id)], gateway=True), gateway_id))
            for upstream_id in connected:
                if upstream_id == gateway_id:
                    continue
                if upstream_id == relay_id:
                    continue
                if (relay_id, upstream_id) in links:
                    metric = links[(relay_id, upstream_id)]
                    options.append((_link_cost(metric, gateway=False) + 0.35, upstream_id))
            if not options:
                next_pending.append(relay_id)
                continue
            _, chosen_parent = min(options, key=lambda item: item[0])
            parent[relay_id] = chosen_parent
            connected.add(relay_id)
            progress = True
        if not progress:
            for relay_id in next_pending:
                notes.append(f"Relay {relay_id} could not be tied into the backbone.")
            break
        pending = next_pending
    return parent, notes


def _choose_assigned_parent(
    node: Node,
    relays: set[int],
    gateway_id: int,
    links: dict[tuple[int, int], LinkMetric],
) -> int | None:
    options: list[tuple[float, int]] = []
    if (node.node_id, gateway_id) in links:
        options.append((_link_cost(links[(node.node_id, gateway_id)], gateway=True), gateway_id))
    for relay_id in relays:
        if relay_id == node.node_id:
            continue
        metric = links.get((node.node_id, relay_id))
        if metric is None:
            continue
        options.append((_link_cost(metric, gateway=False), relay_id))
    if not options:
        return None
    return min(options, key=lambda item: item[0])[1]


def _zone_targets(nodes: list[Node], settings: OptimizationSettings) -> dict[str, int]:
    zone_counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        zone_counts[node.zone] += 1
    return {
        zone: min(settings.max_relays_per_zone, max(1, math.ceil(count / settings.nodes_per_relay)))
        for zone, count in zone_counts.items()
    }


def _relay_cost(node: Node) -> float:
    power_penalty = 0.25 if not node.is_battery_powered else 1.10
    battery_penalty = 1.25 * (1.0 - node.battery_level)
    trust_penalty = 0.95 * (1.0 - node.trust_score)
    fairness_penalty = 0.65 * node.relay_history
    load_penalty = 0.45 * node.last_load
    range_bonus = -0.30 * min(1.8, node.transfer_range_m / 2_600.0)
    speed_bonus = -0.25 * min(1.6, node.transfer_speed_kbps / 18.0)
    return power_penalty + battery_penalty + trust_penalty + fairness_penalty + load_penalty + range_bonus + speed_bonus


def _link_cost(metric: LinkMetric, gateway: bool) -> float:
    quality_penalty = 1.40 * (1.0 - metric.link_quality)
    speed_penalty = 0.90 * (1.0 - min(1.0, metric.effective_speed_kbps / 18.0))
    reliability_penalty = 1.10 * (1.0 - metric.reliability)
    energy_penalty = 0.75 * metric.energy_cost
    gateway_penalty = 0.35 if gateway else 0.0
    return quality_penalty + speed_penalty + reliability_penalty + energy_penalty + gateway_penalty


def _relay_capacity(node: Node) -> float:
    base = 10.0 + node.transfer_speed_kbps * 0.80
    if not node.is_battery_powered:
        base *= 1.30
    return base


def _node_demand(node: Node) -> float:
    return 1.0 + 2.2 * node.traffic_bias + 0.6 * node.last_load


def _is_relay_eligible(node: Node, settings: OptimizationSettings) -> bool:
    battery_ok = (not node.is_battery_powered) or node.battery_level >= settings.min_battery
    return battery_ok and node.trust_score >= settings.min_trust


def _relay_score(node: Node) -> float:
    plugged_bonus = 2.2 if not node.is_battery_powered else 0.0
    battery_bonus = 1.8 * node.battery_level
    trust_bonus = 1.6 * node.trust_score
    range_bonus = min(2.1, node.transfer_range_m / 1_800.0)
    speed_bonus = min(1.8, node.transfer_speed_kbps / 12.0)
    fairness_penalty = 0.9 * node.relay_history
    load_penalty = 0.5 * node.last_load
    return plugged_bonus + battery_bonus + trust_bonus + range_bonus + speed_bonus - fairness_penalty - load_penalty

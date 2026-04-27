from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np

from .geo import SOFIA_BOUNDS, SOFIA_CENTER, SOFIA_ZONES, effective_link_margin, offset_lat_lon
from .linear_program import OptimizationSettings, optimize_mesh
from .mqtt_runtime import MQTTMeshRuntime
from .models import AntennaShape, LinkMetric, Node, OptimizationResult, Packet, PowerMode, Snapshot, TrafficClass


class SimulationConfig:
    def __init__(
        self,
        node_count: int = 108,
        seed: int = 12,
        epoch_ticks: int = 8,
        heatmap_resolution: int = 84,
    ) -> None:
        self.node_count = node_count
        self.seed = seed
        self.epoch_ticks = epoch_ticks
        self.heatmap_resolution = heatmap_resolution


class SofiaMeshSimulation:
    def __init__(self, node_count: int = 108, seed: int = 12) -> None:
        self.config = SimulationConfig(node_count=node_count, seed=seed)
        self.optimizer_settings = OptimizationSettings()
        self.random = random.Random(seed)
        self.tick = 0
        self.epoch = 0
        self.packet_id = 0
        self.gateway = self._make_gateway()
        self.nodes: list[Node] = []
        self.node_by_id: dict[int, Node] = {}
        self.links: dict[tuple[int, int], LinkMetric] = {}
        self.optimization: OptimizationResult | None = None
        self.tree_adjacency: dict[int, set[int]] = defaultdict(set)
        self.event_log: deque[str] = deque(maxlen=32)
        self.delivery_window: deque[int] = deque(maxlen=180)
        self.latency_window: deque[float] = deque(maxlen=180)
        self.total_sent = 0
        self.total_delivered = 0
        self.total_dropped = 0
        self.runtime: MQTTMeshRuntime | None = None
        self.last_snapshot: Snapshot | None = None
        self.regenerate(node_count=node_count, seed=seed)

    def regenerate(self, node_count: int | None = None, seed: int | None = None) -> Snapshot:
        self.shutdown()
        if node_count is not None:
            self.config.node_count = node_count
        if seed is not None:
            self.config.seed = seed
        self.random = random.Random(self.config.seed)
        self.tick = 0
        self.epoch = 0
        self.packet_id = 0
        self.gateway = self._make_gateway()
        self.nodes = self._generate_nodes(self.config.node_count)
        self.node_by_id = {node.node_id: node for node in self.nodes}
        self.links = self._build_links()
        self.event_log.clear()
        self.delivery_window.clear()
        self.latency_window.clear()
        self.total_sent = 0
        self.total_delivered = 0
        self.total_dropped = 0
        self._run_optimizer(initial=True)
        self.runtime = MQTTMeshRuntime(self.nodes, self.gateway, self.links, self.config.seed)
        self.runtime.start()
        if self.optimization is None:
            raise RuntimeError("Optimization must exist before MQTT runtime startup.")
        self.runtime.update_topology(self.links, self.optimization, self.epoch)
        self.last_snapshot = self._make_snapshot([], 0)
        return self.last_snapshot

    def force_reoptimize(self) -> Snapshot:
        self._run_optimizer(initial=False)
        if self.runtime is not None and self.optimization is not None:
            self.runtime.update_topology(self.links, self.optimization, self.epoch)
        self.last_snapshot = self._make_snapshot([], 0)
        return self.last_snapshot

    def shutdown(self) -> None:
        if self.runtime is not None:
            self.runtime.shutdown()
            self.runtime = None

    def step(self) -> Snapshot:
        self.tick += 1
        if self.tick % self.config.epoch_ticks == 0:
            self.epoch += 1
            self._run_optimizer(initial=False)
            if self.runtime is not None and self.optimization is not None:
                self.runtime.update_topology(self.links, self.optimization, self.epoch)

        for node in self.nodes:
            node.last_load *= 0.72
            if node.is_battery_powered:
                node.battery_level = max(0.02, node.battery_level - 0.00022 - 0.00010 * node.last_load)
            else:
                node.battery_level = min(1.0, node.battery_level + 0.012)

        if self.runtime is None:
            raise RuntimeError("MQTT runtime is not initialized.")
        report = self.runtime.step(self.tick)
        self.total_sent = report.total_sent
        self.total_delivered = report.total_delivered
        self.total_dropped = report.total_dropped
        self.delivery_window.extend(report.delivery_results)
        self.latency_window.extend(report.final_latencies_ms)
        for message in report.log_messages:
            self._log(message)

        if self.optimization and any(
            self.node_by_id[relay_id].battery_level < self.optimizer_settings.min_battery + 0.04
            or self.node_by_id[relay_id].trust_score < self.optimizer_settings.min_trust
            for relay_id in self.optimization.selected_relays
        ):
            self._run_optimizer(initial=False)
            if self.runtime is not None and self.optimization is not None:
                self.runtime.update_topology(self.links, self.optimization, self.epoch)

        snapshot = self._make_snapshot(report.active_edges, report.route_failures)
        self.last_snapshot = snapshot
        return snapshot

    def _make_gateway(self) -> Node:
        return Node(
            node_id=0,
            name="Sofia Gateway",
            zone="Gateway",
            lat=SOFIA_CENTER[0],
            lon=SOFIA_CENTER[1],
            power_mode=PowerMode.PLUGGED,
            battery_level=1.0,
            trust_score=1.0,
            transfer_speed_kbps=72.0,
            transfer_range_m=9_000.0,
            range_falloff=1.1,
            antenna_shape=AntennaShape.CIRCLE,
            antenna_direction_deg=0.0,
            antenna_beam_width_deg=360.0,
            sensor_profile="gateway",
            traffic_bias=0.0,
            gateway_affinity=1.0,
            is_gateway=True,
        )

    def _generate_nodes(self, count: int) -> list[Node]:
        sensor_profiles = ["air", "parking", "flood", "weather", "traffic", "noise", "lighting"]
        speed_buckets = [1.2, 2.4, 4.8, 7.2, 12.0, 18.0]
        nodes: list[Node] = []
        for index in range(1, count + 1):
            zone_name, zone_lat, zone_lon = SOFIA_ZONES[(index - 1) % len(SOFIA_ZONES)]
            north_offset = self.random.gauss(0.0, 620.0)
            east_offset = self.random.gauss(0.0, 760.0)
            lat, lon = offset_lat_lon(zone_lat, zone_lon, north_offset, east_offset)
            lat = min(max(lat, SOFIA_BOUNDS["min_lat"]), SOFIA_BOUNDS["max_lat"])
            lon = min(max(lon, SOFIA_BOUNDS["min_lon"]), SOFIA_BOUNDS["max_lon"])
            plugged = self.random.random() < 0.26
            antenna_shape = AntennaShape.SECTOR if self.random.random() < 0.24 else AntennaShape.CIRCLE
            if antenna_shape == AntennaShape.SECTOR:
                base_range = self.random.uniform(3_600.0, 7_000.0)
                beam_width = self.random.choice([90.0, 120.0, 150.0])
            else:
                base_range = self.random.uniform(2_500.0, 5_200.0)
                beam_width = 360.0
            node = Node(
                node_id=index,
                name=f"Node {index:03d}",
                zone=zone_name,
                lat=lat,
                lon=lon,
                power_mode=PowerMode.PLUGGED if plugged else PowerMode.BATTERY,
                battery_level=1.0 if plugged else self.random.uniform(0.36, 0.98),
                trust_score=self.random.uniform(0.58, 0.98),
                transfer_speed_kbps=self.random.choice(speed_buckets) * self.random.uniform(0.82, 1.15),
                transfer_range_m=base_range,
                range_falloff=self.random.uniform(1.05, 1.55),
                antenna_shape=antenna_shape,
                antenna_direction_deg=self.random.uniform(0.0, 360.0),
                antenna_beam_width_deg=beam_width,
                sensor_profile=self.random.choice(sensor_profiles),
                traffic_bias=self.random.uniform(0.45, 1.95),
                gateway_affinity=self.random.uniform(0.55, 1.0),
            )
            nodes.append(node)
        return nodes

    def _build_links(self) -> dict[tuple[int, int], LinkMetric]:
        participants = [self.gateway, *self.nodes]
        links: dict[tuple[int, int], LinkMetric] = {}
        for sender in participants:
            for receiver in participants:
                if sender.node_id == receiver.node_id:
                    continue
                distance_m, usable_range_m, link_quality = effective_link_margin(sender, receiver)
                if usable_range_m <= 0.0:
                    continue
                if distance_m > usable_range_m * 1.05 or link_quality <= 0.05:
                    continue
                effective_speed = min(sender.transfer_speed_kbps, receiver.transfer_speed_kbps) * max(0.18, 0.24 + 0.76 * link_quality)
                reliability = min(
                    0.995,
                    max(
                        0.28,
                        0.58 + 0.28 * link_quality + 0.06 * sender.trust_score + 0.06 * receiver.trust_score - 0.02 * sender.last_load,
                    ),
                )
                energy_cost = (distance_m / max(usable_range_m, 1.0)) * (1.18 if sender.is_battery_powered else 0.74)
                latency_ms = 110.0 + 210.0 * (1.0 - link_quality) + 850.0 / max(1.0, effective_speed)
                links[(sender.node_id, receiver.node_id)] = LinkMetric(
                    source_id=sender.node_id,
                    target_id=receiver.node_id,
                    distance_m=distance_m,
                    usable_range_m=usable_range_m,
                    link_quality=link_quality,
                    effective_speed_kbps=effective_speed,
                    reliability=reliability,
                    energy_cost=energy_cost,
                    latency_ms=latency_ms,
                )
        return links

    def _run_optimizer(self, initial: bool) -> None:
        self.links = self._build_links()
        self.optimization = optimize_mesh(self.nodes, self.gateway, self.links, self.optimizer_settings)
        self.tree_adjacency = defaultdict(set)
        for node in self.nodes:
            node.selected_as_relay = node.node_id in self.optimization.selected_relays
            node.relay_history = node.relay_history * 0.88 + (1.0 if node.selected_as_relay else 0.0)
        for child_id, parent_id in self.optimization.parent_by_node.items():
            self.tree_adjacency[child_id].add(parent_id)
            self.tree_adjacency[parent_id].add(child_id)
        relay_count = len(self.optimization.selected_relays)
        mode = "Initialized" if initial else f"Epoch {self.epoch}"
        solver_state = f"{self.optimization.status}{' fallback' if self.optimization.used_fallback else ''}"
        self._log(f"{mode}: {relay_count} relays active, solver {solver_state}.")
        for note in self.optimization.notes[:3]:
            self._log(note)

    def _generate_packets(self) -> list[Packet]:
        packets: list[Packet] = []
        for node in self.nodes:
            if node.battery_level <= 0.06:
                continue
            chance = 0.018 + 0.026 * node.traffic_bias + (0.008 if node.selected_as_relay else 0.0)
            if self.random.random() > min(0.22, chance):
                continue
            roll = self.random.random()
            if node.battery_level < 0.18 and roll < 0.08:
                traffic_class = TrafficClass.ALERT
                destination_id = self.gateway.node_id
                payload = 28
            elif roll < 0.14:
                traffic_class = TrafficClass.QUERY
                destination_id = self.gateway.node_id
                payload = 36
            elif roll < 0.32:
                traffic_class = TrafficClass.PEER
                destination_id = self._pick_peer_destination(node)
                payload = self.random.randint(20, 52)
            else:
                traffic_class = TrafficClass.TELEMETRY
                destination_id = self.gateway.node_id
                payload = self.random.randint(14, 34)
            self.packet_id += 1
            packets.append(
                Packet(
                    packet_id=self.packet_id,
                    source_id=node.node_id,
                    destination_id=destination_id,
                    traffic_class=traffic_class,
                    payload_bytes=payload,
                    created_tick=self.tick,
                    community=node.zone,
                )
            )
        return packets[:18]

    def _pick_peer_destination(self, node: Node) -> int:
        same_zone = [candidate.node_id for candidate in self.nodes if candidate.node_id != node.node_id and candidate.zone == node.zone]
        if same_zone and self.random.random() < 0.72:
            return self.random.choice(same_zone)
        others = [candidate.node_id for candidate in self.nodes if candidate.node_id != node.node_id]
        return self.random.choice(others)

    def _route(self, source_id: int, destination_id: int) -> list[int] | None:
        if source_id == destination_id:
            return [source_id]
        if source_id not in self.tree_adjacency:
            return None
        queue: deque[tuple[int, list[int]]] = deque([(source_id, [source_id])])
        visited = {source_id}
        while queue:
            current_id, path = queue.popleft()
            for neighbor_id in self.tree_adjacency.get(current_id, set()):
                if neighbor_id in visited:
                    continue
                next_path = [*path, neighbor_id]
                if neighbor_id == destination_id:
                    return next_path
                visited.add(neighbor_id)
                queue.append((neighbor_id, next_path))
        return None

    def _transmit_packet(
        self,
        packet: Packet,
        route: list[int],
        active_edges_counter: dict[tuple[int, int], float],
    ) -> bool:
        source = self.node_by_id.get(packet.source_id)
        if source is None:
            return False
        self.total_sent += 1
        source.emitted_packets += 1
        total_latency = 0.0
        for hop_index in range(len(route) - 1):
            sender_id = route[hop_index]
            receiver_id = route[hop_index + 1]
            metric = self.links.get((sender_id, receiver_id))
            if metric is None:
                self.total_dropped += 1
                self.delivery_window.append(0)
                source.dropped_packets += 1
                if sender_id in self.node_by_id:
                    self._update_trust(self.node_by_id[sender_id], success=False)
                self._log(f"Hop miss N{sender_id}->N{receiver_id} for packet {packet.packet_id}.")
                return False

            sender = self.gateway if sender_id == 0 else self.node_by_id[sender_id]
            success_prob = max(0.12, min(0.995, metric.reliability - 0.02 * sender.last_load))
            if self.random.random() > success_prob:
                self.total_dropped += 1
                self.delivery_window.append(0)
                source.dropped_packets += 1
                self._update_trust(sender, success=False)
                self._log(f"Drop on N{sender_id}->N{receiver_id} ({packet.traffic_class.value}).")
                return False

            self._apply_tx_cost(sender, metric, packet.payload_bytes)
            self._update_trust(sender, success=True)
            if sender_id != packet.source_id and sender_id != 0:
                sender.forwarded_packets += 1
            sender.last_load = min(3.4, sender.last_load + 0.10 + packet.payload_bytes / 160.0)
            if receiver_id != 0:
                self.node_by_id[receiver_id].last_load = min(3.4, self.node_by_id[receiver_id].last_load + 0.04 + packet.payload_bytes / 400.0)
            active_edges_counter[(sender_id, receiver_id)] += 1.0 + (0.55 if packet.traffic_class in {TrafficClass.ALERT, TrafficClass.QUERY} else 0.0)
            total_latency += metric.latency_ms * (1.0 + 0.10 * sender.last_load)

        source.delivered_packets += 1
        source.last_latency_ms = total_latency
        self.total_delivered += 1
        self.delivery_window.append(1)
        self.latency_window.append(total_latency)
        if packet.traffic_class == TrafficClass.ALERT:
            self._log(f"Alert uplink from N{packet.source_id} reached the gateway in {total_latency:.0f} ms.")
        return True

    def _spawn_response(self, packet: Packet) -> Packet:
        self.packet_id += 1
        return Packet(
            packet_id=self.packet_id,
            source_id=self.gateway.node_id,
            destination_id=packet.source_id,
            traffic_class=TrafficClass.RESPONSE,
            payload_bytes=42,
            created_tick=self.tick,
            community=packet.community,
        )

    def _apply_tx_cost(self, sender: Node, metric: LinkMetric, payload_bytes: int) -> None:
        if sender.is_gateway:
            return
        tx_cost = 0.00018 + 0.00048 * metric.energy_cost + 0.00012 * (payload_bytes / 32.0)
        if sender.selected_as_relay:
            tx_cost *= 1.15
        if sender.is_battery_powered:
            sender.battery_level = max(0.02, sender.battery_level - tx_cost)
        else:
            sender.battery_level = min(1.0, sender.battery_level + 0.008)

    def _update_trust(self, node: Node, success: bool) -> None:
        if node.is_gateway:
            return
        observed = 0.97 if success else 0.18
        alpha = 0.82
        node.trust_score = max(0.12, min(0.995, alpha * node.trust_score + (1.0 - alpha) * observed))

    def _make_snapshot(self, active_edges: list[tuple[int, int, float]], route_failures: int) -> Snapshot:
        if self.optimization is None:
            raise RuntimeError("Simulation snapshot requested before optimization completed.")

        heatmap = self._build_heatmap(active_edges)
        avg_battery = float(np.mean([node.battery_level for node in self.nodes])) if self.nodes else 0.0
        avg_trust = float(np.mean([node.trust_score for node in self.nodes])) if self.nodes else 0.0
        recent_pdr = (sum(self.delivery_window) / len(self.delivery_window)) if self.delivery_window else 0.0
        avg_latency = (sum(self.latency_window) / len(self.latency_window)) if self.latency_window else 0.0
        relay_load = sum(self.node_by_id[relay_id].last_load for relay_id in self.optimization.selected_relays)
        metrics = {
            "nodes": len(self.nodes),
            "relays": len(self.optimization.selected_relays),
            "pdr": recent_pdr,
            "avg_latency_ms": avg_latency,
            "avg_battery": avg_battery,
            "avg_trust": avg_trust,
            "total_sent": self.total_sent,
            "total_delivered": self.total_delivered,
            "total_dropped": self.total_dropped,
            "relay_load": relay_load,
            "solver_status": self.optimization.status,
            "objective": self.optimization.objective_value,
        }
        return Snapshot(
            tick=self.tick,
            epoch=self.epoch,
            nodes=self.nodes[:],
            gateway=self.gateway,
            optimization=self.optimization,
            active_edges=active_edges,
            heatmap_rgba=heatmap,
            metrics=metrics,
            event_log=list(self.event_log),
            route_failures=route_failures,
        )

    def _build_heatmap(self, active_edges: list[tuple[int, int, float]]) -> np.ndarray:
        resolution = self.config.heatmap_resolution
        grid = np.zeros((resolution, resolution), dtype=float)
        lat_span = SOFIA_BOUNDS["max_lat"] - SOFIA_BOUNDS["min_lat"]
        lon_span = SOFIA_BOUNDS["max_lon"] - SOFIA_BOUNDS["min_lon"]

        def deposit(lat: float, lon: float, weight: float) -> None:
            x = int((lon - SOFIA_BOUNDS["min_lon"]) / lon_span * (resolution - 1))
            y = int((lat - SOFIA_BOUNDS["min_lat"]) / lat_span * (resolution - 1))
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    gx = min(max(x + dx, 0), resolution - 1)
                    gy = min(max(y + dy, 0), resolution - 1)
                    distance = dx * dx + dy * dy
                    grid[gy, gx] += weight * np.exp(-distance / 3.4)

        for node in self.nodes:
            weight = 0.35 + node.last_load + (0.9 if node.selected_as_relay else 0.0)
            if node.is_battery_powered:
                weight += max(0.0, 0.30 - node.battery_level)
            deposit(node.lat, node.lon, weight)

        for source_id, target_id, weight in active_edges:
            source = self.gateway if source_id == 0 else self.node_by_id[source_id]
            target = self.gateway if target_id == 0 else self.node_by_id[target_id]
            deposit(source.lat, source.lon, 0.55 * weight)
            deposit(target.lat, target.lon, 0.55 * weight)

        if grid.max() > 0:
            grid /= grid.max()
        red = np.clip(grid * 255.0, 0, 255)
        orange = np.clip(grid * 220.0 + 28.0, 0, 255)
        blue = np.clip(120.0 - grid * 100.0, 10, 140)
        alpha = np.clip(grid * 175.0, 0, 175)
        rgba = np.dstack((red, orange, blue, alpha)).astype(np.uint8)
        return np.flipud(rgba)

    def _log(self, message: str) -> None:
        self.event_log.appendleft(message)

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PowerMode(str, Enum):
    BATTERY = "battery"
    PLUGGED = "plugged"


class AntennaShape(str, Enum):
    CIRCLE = "circle"
    SECTOR = "sector"


class TrafficClass(str, Enum):
    TELEMETRY = "telemetry"
    PEER = "peer"
    QUERY = "query"
    RESPONSE = "response"
    ALERT = "alert"


@dataclass(slots=True)
class Node:
    node_id: int
    name: str
    zone: str
    lat: float
    lon: float
    power_mode: PowerMode
    battery_level: float
    trust_score: float
    transfer_speed_kbps: float
    transfer_range_m: float
    range_falloff: float
    antenna_shape: AntennaShape
    antenna_direction_deg: float
    antenna_beam_width_deg: float
    sensor_profile: str
    traffic_bias: float
    gateway_affinity: float
    relay_history: float = 0.0
    last_load: float = 0.0
    last_latency_ms: float = 0.0
    emitted_packets: int = 0
    forwarded_packets: int = 0
    dropped_packets: int = 0
    delivered_packets: int = 0
    selected_as_relay: bool = False
    is_gateway: bool = False

    @property
    def is_battery_powered(self) -> bool:
        return self.power_mode == PowerMode.BATTERY


@dataclass(slots=True)
class LinkMetric:
    source_id: int
    target_id: int
    distance_m: float
    usable_range_m: float
    link_quality: float
    effective_speed_kbps: float
    reliability: float
    energy_cost: float
    latency_ms: float


@dataclass(slots=True)
class Packet:
    packet_id: int
    source_id: int
    destination_id: int
    traffic_class: TrafficClass
    payload_bytes: int
    created_tick: int
    community: str


@dataclass(slots=True)
class OptimizationResult:
    selected_relays: set[int]
    parent_by_node: dict[int, int]
    assignment_by_node: dict[int, int]
    zone_targets: dict[str, int]
    zone_relays: dict[str, list[int]]
    objective_value: float
    status: str
    used_fallback: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Snapshot:
    tick: int
    epoch: int
    nodes: list[Node]
    gateway: Node
    optimization: OptimizationResult
    active_edges: list[tuple[int, int, float]]
    heatmap_rgba: object
    metrics: dict[str, float | int | str]
    event_log: list[str]
    route_failures: int

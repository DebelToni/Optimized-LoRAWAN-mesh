from __future__ import annotations

import asyncio
import json
import random
import socket
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Any, cast

from amqtt.broker import Broker
from amqtt.client import MQTTClient
from amqtt.contexts import BrokerConfig, ListenerConfig

from .models import LinkMetric, Node, OptimizationResult, PowerMode, TrafficClass


MQTT_QOS = 0
TOPIC_TICK = "mesh/control/tick"
TOPIC_TX_PREFIX = "mesh/tx"
TOPIC_RX_PREFIX = "mesh/rx"
TOPIC_NODE_CONTROL_PREFIX = "mesh/control/node"


@dataclass(slots=True)
class RuntimeStepReport:
    active_edges: list[tuple[int, int, float]]
    route_failures: int
    delivery_results: list[int]
    final_latencies_ms: list[float]
    total_sent: int
    total_delivered: int
    total_dropped: int
    log_messages: list[str]


@dataclass(slots=True)
class _Flight:
    signature: tuple[int, int, int]
    end_time: float
    rx_power_dbm: float


@dataclass(slots=True)
class _TickAccumulator:
    active_edges_counter: dict[tuple[int, int], float] = field(default_factory=lambda: defaultdict(float))
    route_failures: int = 0
    delivery_results: list[int] = field(default_factory=list)
    final_latencies_ms: list[float] = field(default_factory=list)
    log_messages: list[str] = field(default_factory=list)


class MQTTMeshRuntime:
    def __init__(
        self,
        nodes: list[Node],
        gateway: Node,
        links: dict[tuple[int, int], LinkMetric],
        seed: int,
    ) -> None:
        self.nodes = nodes
        self.gateway = gateway
        self.links = links
        self.seed = seed
        self.uri = f"mqtt://127.0.0.1:{self._reserve_port()}/"
        self.host, self.port = self._parse_uri(self.uri)
        self.node_by_id = {node.node_id: node for node in nodes}
        self._loop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._started = False
        self._packet_id = 0
        self._totals = {"sent": 0, "delivered": 0, "dropped": 0}
        self._broker: Broker | None = None
        self._admin_client: MQTTClient | None = None
        self._medium: _LoRaMedium | None = None
        self._agents: dict[int, _NodeAgent] = {}

    def start(self) -> None:
        if self._started:
            return
        self._thread = threading.Thread(target=self._run_loop, name="mqtt-mesh-runtime", daemon=True)
        self._thread.start()
        self._submit(self._async_start()).result(timeout=20)
        self._started = True

    def shutdown(self) -> None:
        if not self._started:
            return
        self._submit(self._async_shutdown()).result(timeout=20)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._started = False

    def update_topology(
        self,
        links: dict[tuple[int, int], LinkMetric],
        optimization: OptimizationResult,
        epoch: int,
    ) -> None:
        self.links = links
        self._submit(self._async_update_topology(links, optimization, epoch)).result(timeout=10)

    def step(self, tick: int) -> RuntimeStepReport:
        return self._submit(self._async_step(tick)).result(timeout=20)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    async def _async_start(self) -> None:
        broker_config = BrokerConfig(
            listeners={"default": ListenerConfig(type="tcp", bind=f"{self.host}:{self.port}")},
            plugins={
                "amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True},
                "amqtt.plugins.sys.broker.BrokerSysPlugin": {"sys_interval": 0},
            },
        )
        self._broker = Broker(broker_config)
        await self._broker.start()

        self._admin_client = MQTTClient()
        await self._admin_client.connect(self.uri)

        self._medium = _LoRaMedium(self, self.uri, self.links, self.seed)
        await self._medium.start()

        self._agents = {
            self.gateway.node_id: _NodeAgent(self, self.gateway, self.uri, self.seed + 100_000),
        }
        for node in self.nodes:
            self._agents[node.node_id] = _NodeAgent(self, node, self.uri, self.seed + node.node_id)
        await asyncio.gather(*(agent.start() for agent in self._agents.values()))

    async def _async_shutdown(self) -> None:
        await asyncio.gather(*(agent.stop() for agent in self._agents.values()), return_exceptions=True)
        self._agents.clear()
        if self._medium is not None:
            await self._medium.stop()
            self._medium = None
        if self._admin_client is not None:
            await self._admin_client.disconnect()
            self._admin_client = None
        if self._broker is not None:
            await self._broker.shutdown()
            self._broker = None

    async def _async_update_topology(
        self,
        links: dict[tuple[int, int], LinkMetric],
        optimization: OptimizationResult,
        epoch: int,
    ) -> None:
        if self._medium is None:
            return
        await self._medium.set_topology(links, optimization)
        for node in [self.gateway, *self.nodes]:
            payload = {
                "epoch": epoch,
                "parent_id": optimization.parent_by_node.get(node.node_id),
                "selected_as_relay": node.node_id in optimization.selected_relays,
                "power_mode": node.power_mode.value,
            }
            await self._publish_json(f"{TOPIC_NODE_CONTROL_PREFIX}/{node.node_id}", payload)
        await asyncio.sleep(0.03)

    async def _async_step(self, tick: int) -> RuntimeStepReport:
        if self._medium is None:
            raise RuntimeError("MQTT runtime medium is not initialized.")
        self._medium.begin_tick(tick)
        await self._publish_json(TOPIC_TICK, {"tick": tick})
        await asyncio.sleep(0.05)
        await self._medium.flush(0.10)
        return self._medium.build_report(self._totals)

    async def _publish_json(self, topic: str, payload: dict[str, Any]) -> None:
        if self._admin_client is None:
            return
        await self._admin_client.publish(topic, json.dumps(payload).encode("utf-8"), qos=MQTT_QOS)

    def next_packet_id(self) -> int:
        self._packet_id += 1
        return self._packet_id

    def choose_peer_destination(self, node: Node, rng: random.Random) -> int:
        same_zone = [candidate.node_id for candidate in self.nodes if candidate.node_id != node.node_id and candidate.zone == node.zone]
        if same_zone and rng.random() < 0.72:
            return rng.choice(same_zone)
        others = [candidate.node_id for candidate in self.nodes if candidate.node_id != node.node_id]
        return rng.choice(others)

    def build_app_packet(self, node: Node, tick: int, rng: random.Random) -> dict[str, Any] | None:
        if node.battery_level <= 0.06:
            return None
        chance = 0.018 + 0.026 * node.traffic_bias + (0.008 if node.selected_as_relay else 0.0)
        if rng.random() > min(0.22, chance):
            return None
        roll = rng.random()
        if node.battery_level < 0.18 and roll < 0.08:
            traffic_class = TrafficClass.ALERT
            destination_id = self.gateway.node_id
            payload_bytes = 28
        elif roll < 0.14:
            traffic_class = TrafficClass.QUERY
            destination_id = self.gateway.node_id
            payload_bytes = 36
        elif roll < 0.32:
            traffic_class = TrafficClass.PEER
            destination_id = self.choose_peer_destination(node, rng)
            payload_bytes = rng.randint(20, 52)
        else:
            traffic_class = TrafficClass.TELEMETRY
            destination_id = self.gateway.node_id
            payload_bytes = rng.randint(14, 34)
        return {
            "packet_id": self.next_packet_id(),
            "origin_id": node.node_id,
            "current_sender_id": node.node_id,
            "final_destination_id": destination_id,
            "traffic_class": traffic_class.value,
            "payload_bytes": payload_bytes,
            "created_tick": tick,
            "community": node.zone,
            "route": [],
            "hop_index": 0,
            "accumulated_latency_ms": 0.0,
        }

    def build_response_packet(self, incoming_packet: dict[str, Any], tick: int) -> dict[str, Any]:
        return {
            "packet_id": self.next_packet_id(),
            "origin_id": self.gateway.node_id,
            "current_sender_id": self.gateway.node_id,
            "final_destination_id": incoming_packet["origin_id"],
            "traffic_class": TrafficClass.RESPONSE.value,
            "payload_bytes": 42,
            "created_tick": tick,
            "community": incoming_packet.get("community", "gateway"),
            "route": [],
            "hop_index": 0,
            "accumulated_latency_ms": 0.0,
        }

    @staticmethod
    def _reserve_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, int]:
        without_scheme = uri.removeprefix("mqtt://").removesuffix("/")
        host, port = without_scheme.split(":", 1)
        return host, int(port)


class _NodeAgent:
    def __init__(self, runtime: MQTTMeshRuntime, node: Node, uri: str, seed: int) -> None:
        self.runtime = runtime
        self.node = node
        self.uri = uri
        self.rng = random.Random(seed)
        self.client = MQTTClient()
        self.parent_id: int | None = None
        self.selected_as_relay = node.selected_as_relay
        self.current_epoch = 0
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        await self.client.connect(self.uri)
        await self.client.subscribe(
            [
                (TOPIC_TICK, MQTT_QOS),
                (f"{TOPIC_RX_PREFIX}/{self.node.node_id}", MQTT_QOS),
                (f"{TOPIC_NODE_CONTROL_PREFIX}/{self.node.node_id}", MQTT_QOS),
            ]
        )
        self._running = True
        self._task = asyncio.create_task(self._listen(), name=f"mqtt-node-{self.node.node_id}")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.client.disconnect()

    async def _listen(self) -> None:
        while self._running:
            try:
                message = await self.client.deliver_message(timeout_duration=0.25)
            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                continue
            if message is None:
                continue
            topic = message.topic
            payload = json.loads(bytes(message.data).decode("utf-8"))
            if topic == TOPIC_TICK:
                await self._handle_tick(payload)
            elif topic == f"{TOPIC_NODE_CONTROL_PREFIX}/{self.node.node_id}":
                self.parent_id = payload.get("parent_id")
                self.selected_as_relay = bool(payload.get("selected_as_relay", False))
                self.current_epoch = int(payload.get("epoch", 0))
            elif topic == f"{TOPIC_RX_PREFIX}/{self.node.node_id}":
                await self._handle_incoming(payload)

    async def _handle_tick(self, payload: dict[str, Any]) -> None:
        if self.node.is_gateway:
            return
        packet = self.runtime.build_app_packet(self.node, int(payload.get("tick", 0)), self.rng)
        if packet is not None:
            await self._publish_tx(packet)

    async def _handle_incoming(self, packet: dict[str, Any]) -> None:
        medium = cast(_LoRaMedium | None, self.runtime._medium)
        if medium is None:
            return
        final_destination_id = int(packet["final_destination_id"])
        if final_destination_id == self.node.node_id:
            medium.record_final_delivery(self.node, packet)
            if self.node.is_gateway and packet["traffic_class"] == TrafficClass.QUERY.value:
                await self._publish_tx(self.runtime.build_response_packet(packet, int(packet.get("created_tick", 0))))
            return

        if not (self.node.is_gateway or self.selected_as_relay):
            medium.finalize_drop(packet, self.node.node_id, f"N{self.node.node_id} cannot forward this packet.")
            return

        packet["current_sender_id"] = self.node.node_id
        await self._publish_tx(packet)

    async def _publish_tx(self, packet: dict[str, Any]) -> None:
        await self.client.publish(
            f"{TOPIC_TX_PREFIX}/{self.node.node_id}",
            json.dumps(packet).encode("utf-8"),
            qos=MQTT_QOS,
        )


class _LoRaMedium:
    def __init__(self, runtime: MQTTMeshRuntime, uri: str, links: dict[tuple[int, int], LinkMetric], seed: int) -> None:
        self.runtime = runtime
        self.uri = uri
        self.links = links
        self.random = random.Random(seed + 777)
        self.client = MQTTClient()
        self.optimization: OptimizationResult | None = None
        self.tree_adjacency: dict[int, set[int]] = defaultdict(set)
        self._listen_task: asyncio.Task[None] | None = None
        self._running = False
        self._pending: set[asyncio.Task[None]] = set()
        self._tick_data = _TickAccumulator()
        self._inflight: list[_Flight] = []
        self._next_tx_allowed: dict[int, float] = defaultdict(float)
        self._current_tick = 0

    async def start(self) -> None:
        await self.client.connect(self.uri)
        await self.client.subscribe([(f"{TOPIC_TX_PREFIX}/+", MQTT_QOS)])
        self._running = True
        self._listen_task = asyncio.create_task(self._listen(), name="lora-medium")

    async def stop(self) -> None:
        self._running = False
        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        await self.flush(0.1)
        await self.client.disconnect()

    async def set_topology(self, links: dict[tuple[int, int], LinkMetric], optimization: OptimizationResult) -> None:
        self.links = links
        self.optimization = optimization
        self.tree_adjacency = defaultdict(set)
        for child_id, parent_id in optimization.parent_by_node.items():
            self.tree_adjacency[child_id].add(parent_id)
            self.tree_adjacency[parent_id].add(child_id)

    def begin_tick(self, tick: int) -> None:
        self._current_tick = tick
        self._tick_data = _TickAccumulator()

    async def flush(self, timeout_s: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while self._pending and loop.time() < deadline:
            wait_timeout = max(0.01, deadline - loop.time())
            done, pending = await asyncio.wait(self._pending, timeout=wait_timeout)
            self._pending = set(pending)
            for task in done:
                task.result()

    def build_report(self, totals: dict[str, int]) -> RuntimeStepReport:
        active_edges = [
            (source_id, target_id, weight)
            for (source_id, target_id), weight in sorted(self._tick_data.active_edges_counter.items(), key=lambda item: item[1], reverse=True)
        ]
        return RuntimeStepReport(
            active_edges=active_edges,
            route_failures=self._tick_data.route_failures,
            delivery_results=self._tick_data.delivery_results[:],
            final_latencies_ms=self._tick_data.final_latencies_ms[:],
            total_sent=totals["sent"],
            total_delivered=totals["delivered"],
            total_dropped=totals["dropped"],
            log_messages=self._tick_data.log_messages[:],
        )

    async def _listen(self) -> None:
        while self._running:
            try:
                message = await self.client.deliver_message(timeout_duration=0.25)
            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                continue
            if message is None:
                continue
            packet = json.loads(bytes(message.data).decode("utf-8"))
            await self._handle_tx(packet)

    async def _handle_tx(self, packet: dict[str, Any]) -> None:
        sender_id = int(packet["current_sender_id"])
        if not packet.get("route"):
            route = self._route(int(packet["origin_id"]), int(packet["final_destination_id"]))
            if route is None:
                self.finalize_drop(packet, sender_id, f"Route miss from N{packet['origin_id']} to N{packet['final_destination_id']}.", route_failure=True)
                return
            packet["route"] = route
            packet["hop_index"] = 0

        route = [int(node_id) for node_id in packet["route"]]
        hop_index = int(packet.get("hop_index", 0))
        if hop_index >= len(route) - 1 or route[hop_index] != sender_id:
            self.finalize_drop(packet, sender_id, f"Invalid route state on packet {packet['packet_id']}.")
            return

        next_hop = route[hop_index + 1]
        metric = self.links.get((sender_id, next_hop))
        if metric is None:
            self.finalize_drop(packet, sender_id, f"Hop miss N{sender_id}->N{next_hop} on packet {packet['packet_id']}.")
            return

        now = asyncio.get_running_loop().time()
        if hop_index == 0:
            self.runtime._totals["sent"] += 1
            if sender_id in self.runtime.node_by_id:
                self.runtime.node_by_id[sender_id].emitted_packets += 1

        if sender_id != 0 and now < self._next_tx_allowed[sender_id]:
            self.finalize_drop(packet, sender_id, f"Duty-cycle block at N{sender_id}.")
            return

        radio = self._radio_profile(metric, int(packet["payload_bytes"]))
        channel = int(radio["channel"])
        sf = int(radio["sf"])
        rx_power_dbm = float(radio["rx_power_dbm"])
        self._prune_inflight(now)
        if self._collides(now, next_hop, channel, sf, rx_power_dbm):
            self.finalize_drop(packet, sender_id, f"Collision on channel {radio['channel']} to N{next_hop}.")
            return

        if sender_id != 0:
            self._next_tx_allowed[sender_id] = now + radio["toa_scaled_s"] * 15.0

        self._inflight.append(
            _Flight(
                signature=(next_hop, int(radio["channel"]), int(radio["sf"])),
                end_time=now + radio["toa_scaled_s"],
                rx_power_dbm=float(radio["rx_power_dbm"]),
            )
        )

        self._apply_tx_success(sender_id, next_hop, metric, int(packet["payload_bytes"]), packet["traffic_class"], hop_index == 0, int(packet["origin_id"]))

        packet["accumulated_latency_ms"] = float(packet.get("accumulated_latency_ms", 0.0)) + float(radio["total_latency_ms"])
        packet["hop_index"] = hop_index + 1

        delivery_task = asyncio.create_task(self._deliver_after(packet, next_hop, metric, radio), name=f"deliver-{packet['packet_id']}-{next_hop}")
        self._pending.add(delivery_task)
        delivery_task.add_done_callback(self._pending.discard)

    async def _deliver_after(
        self,
        packet: dict[str, Any],
        next_hop: int,
        metric: LinkMetric,
        radio: dict[str, float | int],
    ) -> None:
        total_delay_s = float(radio["total_delay_scaled_s"])
        await asyncio.sleep(total_delay_s)
        sender_id = int(packet["current_sender_id"])
        sender = self.runtime.gateway if sender_id == 0 else self.runtime.node_by_id[sender_id]
        success_prob = max(0.18, min(0.995, metric.reliability + 0.02 * (int(radio["sf"]) - 7) - 0.015 * sender.last_load))
        if self.random.random() > success_prob:
            self.finalize_drop(packet, sender_id, f"LoRa link loss on N{sender_id}->N{next_hop} ({packet['traffic_class']}).")
            return

        await self.client.publish(
            f"{TOPIC_RX_PREFIX}/{next_hop}",
            json.dumps(packet).encode("utf-8"),
            qos=MQTT_QOS,
        )

    def record_final_delivery(self, receiver: Node, packet: dict[str, Any]) -> None:
        origin_id = int(packet["origin_id"])
        latency_ms = float(packet.get("accumulated_latency_ms", 0.0))
        self.runtime._totals["delivered"] += 1
        self._tick_data.delivery_results.append(1)
        self._tick_data.final_latencies_ms.append(latency_ms)
        if origin_id in self.runtime.node_by_id:
            origin = self.runtime.node_by_id[origin_id]
            origin.delivered_packets += 1
            origin.last_latency_ms = latency_ms
        if packet["traffic_class"] == TrafficClass.ALERT.value:
            self.log(f"Alert uplink from N{origin_id} reached the gateway in {latency_ms:.0f} ms.")

    def finalize_drop(self, packet: dict[str, Any], sender_id: int, reason: str, route_failure: bool = False) -> None:
        self.runtime._totals["dropped"] += 1
        self._tick_data.delivery_results.append(0)
        origin_id = int(packet["origin_id"])
        if origin_id in self.runtime.node_by_id:
            self.runtime.node_by_id[origin_id].dropped_packets += 1
        if sender_id in self.runtime.node_by_id:
            self._update_trust(self.runtime.node_by_id[sender_id], success=False)
        if route_failure:
            self._tick_data.route_failures += 1
        self.log(reason)

    def _apply_tx_success(
        self,
        sender_id: int,
        receiver_id: int,
        metric: LinkMetric,
        payload_bytes: int,
        traffic_class: str,
        first_hop: bool,
        origin_id: int,
    ) -> None:
        sender = self.runtime.gateway if sender_id == 0 else self.runtime.node_by_id[sender_id]
        self._update_trust(sender, success=True)
        if sender_id != 0 and not first_hop:
            sender.forwarded_packets += 1
        if sender_id != 0:
            tx_cost = 0.00018 + 0.00052 * metric.energy_cost + 0.00011 * (payload_bytes / 28.0)
            if sender.selected_as_relay:
                tx_cost *= 1.15
            if sender.power_mode == PowerMode.BATTERY:
                sender.battery_level = max(0.02, sender.battery_level - tx_cost)
            else:
                sender.battery_level = min(1.0, sender.battery_level + 0.006)
        sender.last_load = min(3.8, sender.last_load + 0.08 + payload_bytes / 150.0)
        if receiver_id in self.runtime.node_by_id:
            receiver = self.runtime.node_by_id[receiver_id]
            receiver.last_load = min(3.8, receiver.last_load + 0.03 + payload_bytes / 380.0)
        edge_weight = 1.0 + (0.55 if traffic_class in {TrafficClass.ALERT.value, TrafficClass.QUERY.value} else 0.0)
        self._tick_data.active_edges_counter[(sender_id, receiver_id)] += edge_weight

    def _route(self, source_id: int, destination_id: int) -> list[int] | None:
        if source_id == destination_id:
            return [source_id]
        queue: asyncio.Queue[tuple[int, list[int]]] = asyncio.Queue()
        queue.put_nowait((source_id, [source_id]))
        visited = {source_id}
        while not queue.empty():
            current_id, path = queue.get_nowait()
            for neighbor_id in self.tree_adjacency.get(current_id, set()):
                if neighbor_id in visited:
                    continue
                next_path = [*path, neighbor_id]
                if neighbor_id == destination_id:
                    return next_path
                visited.add(neighbor_id)
                queue.put_nowait((neighbor_id, next_path))
        return None

    def _radio_profile(self, metric: LinkMetric, payload_bytes: int) -> dict[str, float | int]:
        link_quality = metric.link_quality
        if link_quality >= 0.82:
            sf = 7
        elif link_quality >= 0.68:
            sf = 8
        elif link_quality >= 0.54:
            sf = 9
        elif link_quality >= 0.40:
            sf = 10
        elif link_quality >= 0.26:
            sf = 11
        else:
            sf = 12
        bw_khz = 250 if metric.effective_speed_kbps >= 12.0 else 125
        cr = 1 if link_quality >= 0.6 else 4
        symbol_ms = (2**sf) / float(bw_khz)
        payload_symbols = 8 + max(
            ceil((8 * payload_bytes - 4 * sf + 28 + 16 - 20) / max(4 * (sf - 2), 4)) * (cr + 4),
            0,
        )
        toa_ms = (8 + 4.25 + payload_symbols) * symbol_ms
        rx_power_dbm = -118.0 + 28.0 * link_quality - 0.0024 * metric.distance_m
        channel = (payload_bytes + int(metric.distance_m) + sf) % 3
        total_latency_ms = toa_ms + metric.latency_ms
        return {
            "sf": sf,
            "bw_khz": bw_khz,
            "cr": cr,
            "channel": channel,
            "toa_ms": toa_ms,
            "toa_scaled_s": max(0.0007, toa_ms / 1_000.0 * 0.025),
            "total_latency_ms": total_latency_ms,
            "total_delay_scaled_s": max(0.0010, total_latency_ms / 1_000.0 * 0.035),
            "rx_power_dbm": rx_power_dbm,
        }

    def _collides(self, now: float, next_hop: int, channel: int, sf: int, rx_power_dbm: float) -> bool:
        for flight in self._inflight:
            if flight.end_time <= now:
                continue
            same_signature = flight.signature == (next_hop, channel, sf)
            if same_signature and abs(flight.rx_power_dbm - rx_power_dbm) < 6.0:
                return True
        return False

    def _prune_inflight(self, now: float) -> None:
        self._inflight = [flight for flight in self._inflight if flight.end_time > now]

    def _update_trust(self, node: Node, success: bool) -> None:
        if node.is_gateway:
            return
        observed = 0.97 if success else 0.18
        alpha = 0.84
        node.trust_score = max(0.12, min(0.995, alpha * node.trust_score + (1.0 - alpha) * observed))

    def log(self, message: str) -> None:
        self._tick_data.log_messages.append(message)

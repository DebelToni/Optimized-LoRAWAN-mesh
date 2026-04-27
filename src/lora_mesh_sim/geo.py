from __future__ import annotations

import math
from typing import Iterable

from .models import AntennaShape, Node


SOFIA_CENTER = (42.6977, 23.3219)
SOFIA_BOUNDS = {
    "min_lat": 42.625,
    "max_lat": 42.755,
    "min_lon": 23.220,
    "max_lon": 23.460,
}

SOFIA_ZONES: tuple[tuple[str, float, float], ...] = (
    ("Lyulin", 42.7166, 23.2575),
    ("Nadezhda", 42.7330, 23.3015),
    ("Center", 42.6977, 23.3219),
    ("Lozenets", 42.6755, 23.3302),
    ("Studentski", 42.6508, 23.3523),
    ("Mladost", 42.6468, 23.3838),
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def offset_lat_lon(base_lat: float, base_lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    lat = base_lat + north_m / 111_320.0
    lon = base_lon + east_m / (111_320.0 * math.cos(math.radians(base_lat)))
    return lat, lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi_2)
    x = math.cos(phi_1) * math.sin(phi_2) - math.sin(phi_1) * math.cos(phi_2) * math.cos(delta_lambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angular_delta_deg(angle_a: float, angle_b: float) -> float:
    return abs(((angle_a - angle_b + 180.0) % 360.0) - 180.0)


def antenna_gain(node: Node, bearing_to_target: float) -> float:
    if node.antenna_shape == AntennaShape.CIRCLE:
        return 1.0
    delta = angular_delta_deg(node.antenna_direction_deg, bearing_to_target)
    half_beam = max(1.0, node.antenna_beam_width_deg / 2.0)
    if delta >= half_beam:
        return 0.22
    normalized = 1.0 - (delta / half_beam)
    return 0.55 + 0.45 * normalized * normalized


def effective_link_margin(sender: Node, receiver: Node) -> tuple[float, float, float]:
    distance = haversine_m(sender.lat, sender.lon, receiver.lat, receiver.lon)
    forward_bearing = bearing_deg(sender.lat, sender.lon, receiver.lat, receiver.lon)
    reverse_bearing = (forward_bearing + 180.0) % 360.0
    sender_gain = antenna_gain(sender, forward_bearing)
    receiver_gain = antenna_gain(receiver, reverse_bearing)
    sender_range = sender.transfer_range_m * max(0.22, sender_gain)
    receiver_range = receiver.transfer_range_m * max(0.45, receiver_gain)
    usable_range = min(sender_range, receiver_range)
    if usable_range <= 0.0:
        return distance, usable_range, 0.0
    normalized_distance = clamp(distance / usable_range, 0.0, 2.0)
    quality = clamp(1.0 - normalized_distance**sender.range_falloff, 0.0, 1.0)
    return distance, usable_range, quality


def coverage_polygon(node: Node, steps: int = 40) -> list[tuple[float, float]]:
    if node.antenna_shape == AntennaShape.CIRCLE:
        headings = [index * 360.0 / steps for index in range(steps)]
    else:
        half_beam = node.antenna_beam_width_deg / 2.0
        headings = [
            node.antenna_direction_deg - half_beam + index * node.antenna_beam_width_deg / max(1, steps - 1)
            for index in range(steps)
        ]
        headings = [node.antenna_direction_deg] + headings
    points = [(node.lat, node.lon)] if node.antenna_shape == AntennaShape.SECTOR else []
    for heading in headings:
        rad = math.radians(heading)
        north_m = math.cos(rad) * node.transfer_range_m
        east_m = math.sin(rad) * node.transfer_range_m
        points.append(offset_lat_lon(node.lat, node.lon, north_m, east_m))
    if node.antenna_shape == AntennaShape.SECTOR:
        points.append((node.lat, node.lon))
    elif points:
        points.append(points[0])
    return points


def centroid(points: Iterable[tuple[float, float]]) -> tuple[float, float]:
    pts = list(points)
    lat = sum(point[0] for point in pts) / len(pts)
    lon = sum(point[1] for point in pts) / len(pts)
    return lat, lon

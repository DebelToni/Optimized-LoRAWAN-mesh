from __future__ import annotations

import argparse
import sys
from collections import deque
from typing import cast

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph_gis import MapWidget
from pyqtgraph_gis.utils import latlon_to_web_mercator, vectorized_wgs84_to_wm

from .geo import SOFIA_BOUNDS, coverage_polygon, haversine_m
from .models import AntennaShape, Node, PowerMode, Snapshot
from .simulation import SofiaMeshSimulation


pg.setConfigOptions(useOpenGL=False, antialias=True)


class PolygonOverlayItem(pg.GraphicsObject):
    def __init__(self, pen: QtGui.QPen, brush: QtGui.QBrush | None = None, parent: QtWidgets.QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self._pen = pen
        self._brush = brush or QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush)
        self._path = QtGui.QPainterPath()
        self._bounding_rect = QtCore.QRectF()

    def set_polygons(self, polygons: list[list[tuple[float, float]]]) -> None:
        path = QtGui.QPainterPath()
        for polygon in polygons:
            if len(polygon) < 2:
                continue
            qpolygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in polygon])
            sub_path = QtGui.QPainterPath()
            sub_path.addPolygon(qpolygon)
            path.addPath(sub_path)
        self.prepareGeometryChange()
        self._path = path
        self._bounding_rect = path.boundingRect()
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        return self._bounding_rect

    def paint(
        self,
        painter: QtGui.QPainter | None,
        option: QtWidgets.QStyleOptionGraphicsItem | None,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        if painter is None:
            return
        painter.setPen(self._pen)
        painter.setBrush(self._brush)
        painter.drawPath(self._path)


class MeshWindow(QtWidgets.QMainWindow):
    def __init__(self, node_count: int = 108, seed: int = 12, interval_ms: int = 850, auto_start: bool = True) -> None:
        super().__init__()
        self.simulation = SofiaMeshSimulation(node_count=node_count, seed=seed)
        self.snapshot = self.simulation.last_snapshot
        self.selected_node_id = self._default_selected_node_id()
        self.history_limit = 180
        self._history_last_tick: int | None = None
        self._history_tick: deque[int] = deque(maxlen=self.history_limit)
        self._history_pdr: deque[float] = deque(maxlen=self.history_limit)
        self._history_latency: deque[float] = deque(maxlen=self.history_limit)
        self._history_battery: deque[float] = deque(maxlen=self.history_limit)
        self._history_trust: deque[float] = deque(maxlen=self.history_limit)
        self._history_sent: deque[float] = deque(maxlen=self.history_limit)
        self._history_delivered: deque[float] = deque(maxlen=self.history_limit)
        self._history_dropped: deque[float] = deque(maxlen=self.history_limit)
        self._history_relays: deque[float] = deque(maxlen=self.history_limit)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self._advance_once)
        self._build_ui(interval_ms)
        self._apply_theme()
        self._refresh(self.snapshot)
        if auto_start:
            self.timer.start()

    def _build_ui(self, interval_ms: int) -> None:
        self.setWindowTitle("Optimized LoRaWAN Mesh - Sofia Simulation")
        self.resize(1580, 940)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_map_panel())
        splitter.addWidget(self._build_sidebar(interval_ms))
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1150, 430])
        self.setCentralWidget(splitter)

        self.coord_label = QtWidgets.QLabel("42.6977, 23.3219")
        self.summary_label = QtWidgets.QLabel("MILP relay selection live")
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.addPermanentWidget(self.summary_label)
            status_bar.addPermanentWidget(self.coord_label)

    def _build_map_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.map_widget = MapWidget(
            tile_server_url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            headers={"User-Agent": "Optimized-LoRaWAN-Mesh-Demo/1.0 (educational simulator)"},  # type: ignore[arg-type]
            attribution_text='Map data <a href="https://www.openstreetmap.org/copyright">© OpenStreetMap contributors</a>',
        )
        self.map_widget.sigMapClicked.connect(self._on_map_clicked)
        self.map_widget.sigMouseMoved.connect(self._on_map_hover)
        self.map_widget.setBackground("#07131d")
        plot_item = self.map_widget.getPlotItem()
        if plot_item is not None:
            plot_item.setMenuEnabled(False)
        self.map_widget.showGrid(x=False, y=False)

        min_x, min_y = latlon_to_web_mercator(SOFIA_BOUNDS["min_lat"], SOFIA_BOUNDS["min_lon"])
        max_x, max_y = latlon_to_web_mercator(SOFIA_BOUNDS["max_lat"], SOFIA_BOUNDS["max_lon"])
        view_box = self.map_widget.getViewBox()
        if view_box is not None:
            view_box.setRange(QtCore.QRectF(min_x, min_y, max_x - min_x, max_y - min_y), padding=0.02)

        self.backbone_item = pg.PlotCurveItem(pen=pg.mkPen("#3bc8ff", width=2.0))
        self.active_links_item = pg.PlotCurveItem(pen=pg.mkPen("#ff9350", width=3.0))
        self.node_item = pg.ScatterPlotItem(pxMode=True)
        self.selection_item = pg.ScatterPlotItem(pxMode=True)
        self.selection_item.setZValue(40)

        ambient_pen = QtGui.QPen(QtGui.QColor(75, 193, 255, 50), 1.0)
        ambient_brush = QtGui.QBrush(QtGui.QColor(44, 139, 192, 18))
        relay_pen = QtGui.QPen(QtGui.QColor(255, 102, 180, 105), 1.35)
        relay_brush = QtGui.QBrush(QtGui.QColor(255, 102, 180, 28))
        focus_pen = QtGui.QPen(QtGui.QColor(255, 237, 129, 180), 2.0)
        focus_brush = QtGui.QBrush(QtGui.QColor(255, 237, 129, 32))

        self.ambient_ranges_item = PolygonOverlayItem(ambient_pen, ambient_brush)
        self.relay_ranges_item = PolygonOverlayItem(relay_pen, relay_brush)
        self.focus_range_item = PolygonOverlayItem(focus_pen, focus_brush)
        self.ambient_ranges_item.setZValue(4)
        self.relay_ranges_item.setZValue(5)
        self.focus_range_item.setZValue(6)

        self.map_widget.addItem(self.backbone_item)
        self.map_widget.addItem(self.active_links_item)
        self.map_widget.addItem(self.node_item)
        self.map_widget.addItem(self.selection_item)
        self.map_widget.addItem(self.ambient_ranges_item)
        self.map_widget.addItem(self.relay_ranges_item)
        self.map_widget.addItem(self.focus_range_item)

        blank_heatmap = self._empty_heatmap()
        self.heatmap_item = self.map_widget.image(
            blank_heatmap,
            SOFIA_BOUNDS["min_lat"],
            SOFIA_BOUNDS["min_lon"],
            SOFIA_BOUNDS["max_lat"],
            SOFIA_BOUNDS["max_lon"],
            opacity=0.72,
        )
        self.heatmap_item.setZValue(3)

        title_bar = QtWidgets.QFrame()
        title_bar.setObjectName("MapHeader")
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(14, 10, 14, 10)
        title_layout.setSpacing(10)
        title = QtWidgets.QLabel("Sofia Mesh Live View")
        title.setObjectName("MapTitle")
        subtitle = QtWidgets.QLabel("Gateway backbone, congestion heatmap, adaptive relay zones")
        subtitle.setObjectName("MapSubtitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        title_layout.addStretch(1)

        layout.addWidget(title_bar)
        layout.addWidget(self.map_widget)
        return panel

    def closeEvent(self, a0: QtGui.QCloseEvent | None) -> None:
        self.timer.stop()
        self.simulation.shutdown()
        threadpool = getattr(self.map_widget, "threadpool", None)
        if threadpool is not None:
            threadpool.waitForDone(3_000)
        super().closeEvent(a0)

    def _build_sidebar(self, interval_ms: int) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        layout.addWidget(self._build_controls(interval_ms))
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_telemetry_tab(), "Telemetry")
        tabs.addTab(self._build_graphs_tab(), "Graphs")
        layout.addWidget(tabs, 1)
        return panel

    def _build_telemetry_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_metrics_group())
        layout.addWidget(self._build_selected_group())
        layout.addWidget(self._build_relays_group())
        layout.addWidget(self._build_events_group(), 1)
        return page

    def _build_graphs_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.graph_summary_label = QtWidgets.QLabel("Live history for the current simulation run")
        self.graph_summary_label.setWordWrap(True)
        layout.addWidget(self.graph_summary_label)

        self.reliability_plot = self._make_plot_widget("Reliability", "Packet delivery and relay count")
        self.latency_plot = self._make_plot_widget("Latency", "Average delivery latency")
        self.energy_plot = self._make_plot_widget("Energy / Trust", "Average battery and trust")
        self.volume_plot = self._make_plot_widget("Traffic Volume", "Cumulative sent, delivered, dropped")

        self.reliability_pdr_curve = self.reliability_plot.plot(pen=pg.mkPen("#41d37d", width=2.2), name="PDR %")
        self.reliability_relay_curve = self.reliability_plot.plot(pen=pg.mkPen("#ff5fa2", width=1.8), name="Relays")
        self.latency_curve = self.latency_plot.plot(pen=pg.mkPen("#ffb347", width=2.2), name="Latency ms")
        self.energy_battery_curve = self.energy_plot.plot(pen=pg.mkPen("#59c2ff", width=2.0), name="Battery %")
        self.energy_trust_curve = self.energy_plot.plot(pen=pg.mkPen("#c792ff", width=2.0), name="Trust %")
        self.volume_sent_curve = self.volume_plot.plot(pen=pg.mkPen("#59c2ff", width=1.8), name="Sent")
        self.volume_delivered_curve = self.volume_plot.plot(pen=pg.mkPen("#41d37d", width=1.8), name="Delivered")
        self.volume_dropped_curve = self.volume_plot.plot(pen=pg.mkPen("#ff6b6b", width=1.8), name="Dropped")

        layout.addWidget(self.reliability_plot, 1)
        layout.addWidget(self.latency_plot, 1)
        layout.addWidget(self.energy_plot, 1)
        layout.addWidget(self.volume_plot, 1)
        return page

    def _make_plot_widget(self, title: str, subtitle: str) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setBackground("#0f1d29")
        plot.showGrid(x=True, y=True, alpha=0.18)
        plot.setMinimumHeight(150)
        plot.addLegend(offset=(8, 8))
        plot.setTitle(f"{title}<br><span style='color:#8fb0ca; font-size:10pt'>{subtitle}</span>")
        plot.getAxis("left").setTextPen(pg.mkPen("#b4cde4"))
        plot.getAxis("bottom").setTextPen(pg.mkPen("#b4cde4"))
        plot.getAxis("left").setPen(pg.mkPen("#37526b"))
        plot.getAxis("bottom").setPen(pg.mkPen("#37526b"))
        return plot

    def _build_controls(self, interval_ms: int) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Scenario")
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.node_count_spin = QtWidgets.QSpinBox()
        self.node_count_spin.setRange(40, 180)
        self.node_count_spin.setValue(self.simulation.config.node_count)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(1, 999_999)
        self.seed_spin.setValue(self.simulation.config.seed)

        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(180, 2500)
        self.interval_spin.setSuffix(" ms")
        self.interval_spin.setValue(interval_ms)
        self.interval_spin.valueChanged.connect(self._update_interval)

        self.run_button = QtWidgets.QPushButton("Pause")
        self.run_button.clicked.connect(self._toggle_running)
        self.step_button = QtWidgets.QPushButton("Single Tick")
        self.step_button.clicked.connect(self._advance_once)
        self.optimize_button = QtWidgets.QPushButton("Re-optimize")
        self.optimize_button.clicked.connect(self._force_optimize)
        self.regen_button = QtWidgets.QPushButton("Regenerate")
        self.regen_button.clicked.connect(self._regenerate)

        self.show_heatmap_check = QtWidgets.QCheckBox("Hotspots")
        self.show_heatmap_check.setChecked(True)
        self.show_heatmap_check.toggled.connect(lambda _: self._refresh(self.snapshot))

        self.show_ranges_check = QtWidgets.QCheckBox("Ranges")
        self.show_ranges_check.setChecked(True)
        self.show_ranges_check.toggled.connect(lambda _: self._refresh(self.snapshot))

        self.show_links_check = QtWidgets.QCheckBox("Links")
        self.show_links_check.setChecked(True)
        self.show_links_check.toggled.connect(lambda _: self._refresh(self.snapshot))

        grid.addWidget(QtWidgets.QLabel("Nodes"), 0, 0)
        grid.addWidget(self.node_count_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Seed"), 0, 2)
        grid.addWidget(self.seed_spin, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Tick interval"), 1, 0)
        grid.addWidget(self.interval_spin, 1, 1)
        grid.addWidget(self.run_button, 2, 0, 1, 2)
        grid.addWidget(self.step_button, 2, 2, 1, 2)
        grid.addWidget(self.optimize_button, 3, 0, 1, 2)
        grid.addWidget(self.regen_button, 3, 2, 1, 2)
        grid.addWidget(self.show_heatmap_check, 4, 0)
        grid.addWidget(self.show_ranges_check, 4, 1)
        grid.addWidget(self.show_links_check, 4, 2)
        return group

    def _build_metrics_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Telemetry")
        form = QtWidgets.QFormLayout(group)
        self.metric_labels: dict[str, QtWidgets.QLabel] = {}
        for key in [
            "Epoch",
            "Nodes / Relays",
            "Delivery Ratio",
            "Average Latency",
            "Average Battery",
            "Average Trust",
            "Relay Load",
            "Packets",
            "Solver",
        ]:
            label = QtWidgets.QLabel("-")
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self.metric_labels[key] = label
            form.addRow(f"{key}:", label)
        return group

    def _build_selected_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Selected Node")
        form = QtWidgets.QFormLayout(group)
        self.selected_labels: dict[str, QtWidgets.QLabel] = {}
        for key in ["Name", "Zone", "Role", "Power", "Battery", "Trust", "Link Speed", "Range", "Antenna", "Profile"]:
            label = QtWidgets.QLabel("-")
            label.setWordWrap(True)
            self.selected_labels[key] = label
            form.addRow(f"{key}:", label)
        return group

    def _build_relays_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Top Relay Load")
        layout = QtWidgets.QVBoxLayout(group)
        self.relay_table = QtWidgets.QTableWidget(0, 5)
        self.relay_table.setHorizontalHeaderLabels(["Node", "Zone", "Load", "Battery", "Trust"])
        vertical_header = self.relay_table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        self.relay_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.relay_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        header = self.relay_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.relay_table)
        return group

    def _build_events_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Event Stream")
        layout = QtWidgets.QVBoxLayout(group)
        self.event_list = QtWidgets.QListWidget()
        self.event_list.setAlternatingRowColors(True)
        layout.addWidget(self.event_list)
        return group

    def _apply_theme(self) -> None:
        self.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#0a1621"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#101d2a"))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#0d1824"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#e6f1fb"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#e6f1fb"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#132435"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#e6f1fb"))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#1d9bf0"))
        self.setPalette(palette)
        self.setStyleSheet(
            """
            QWidget {
                background: #0a1621;
                color: #e6f1fb;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #203548;
                border-radius: 10px;
                margin-top: 8px;
                padding-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                background: #153048;
                border: 1px solid #244867;
                border-radius: 8px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background: #1a3b59;
            }
            QSpinBox, QListWidget, QTableWidget {
                background: #0f1d29;
                border: 1px solid #264055;
                border-radius: 8px;
                padding: 4px;
            }
            QCheckBox {
                spacing: 6px;
            }
            QTabWidget::pane {
                border: 1px solid #203548;
                border-radius: 10px;
                top: -1px;
            }
            QTabBar::tab {
                background: #102131;
                border: 1px solid #203548;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #18324a;
            }
            #MapHeader {
                background: #0d1e2d;
                border-bottom: 1px solid #1f3649;
            }
            #MapTitle {
                font-size: 17px;
                font-weight: 700;
            }
            #MapSubtitle {
                color: #8fb0ca;
            }
            """
        )

    def _toggle_running(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.run_button.setText("Resume")
        else:
            self.timer.start()
            self.run_button.setText("Pause")

    def _update_interval(self, value: int) -> None:
        self.timer.setInterval(value)

    def _regenerate(self) -> None:
        self.snapshot = self.simulation.regenerate(node_count=self.node_count_spin.value(), seed=self.seed_spin.value())
        self.selected_node_id = self._default_selected_node_id()
        self._reset_history()
        self._refresh(self.snapshot)

    def _force_optimize(self) -> None:
        self.snapshot = self.simulation.force_reoptimize()
        self._refresh(self.snapshot)

    def _advance_once(self) -> None:
        self.snapshot = self.simulation.step()
        self._refresh(self.snapshot)

    def _refresh(self, snapshot: Snapshot | None) -> None:
        if snapshot is None:
            return
        self.snapshot = snapshot
        self._record_history(snapshot)
        if self.selected_node_id not in {node.node_id for node in snapshot.nodes} and self.selected_node_id != 0:
            self.selected_node_id = self._default_selected_node_id()
        self._render_heatmap(snapshot)
        self._render_ranges(snapshot)
        self._render_links(snapshot)
        self._render_nodes(snapshot)
        self._update_metrics(snapshot)
        self._update_selected(snapshot)
        self._update_relay_table(snapshot)
        self._update_events(snapshot)
        self._update_graphs(snapshot)
        self.summary_label.setText(
            f"Epoch {snapshot.epoch} | {snapshot.metrics['relays']} relays | {snapshot.metrics['solver_status']} | route fails {snapshot.route_failures}"
        )

    def _render_heatmap(self, snapshot: Snapshot) -> None:
        heatmap_item = cast(object, self.heatmap_item)
        if heatmap_item is None:
            return
        image_item = cast(pg.ImageItem, heatmap_item)
        image_item.setVisible(self.show_heatmap_check.isChecked())
        image_item.setImage(cast(np.ndarray, snapshot.heatmap_rgba), autoLevels=False)

    def _render_links(self, snapshot: Snapshot) -> None:
        if not self.show_links_check.isChecked():
            self.backbone_item.setData([], [])
            self.active_links_item.setData([], [])
            return
        backbone_edges = [(child_id, parent_id, 1.0) for child_id, parent_id in snapshot.optimization.parent_by_node.items()]
        active_edges = snapshot.active_edges
        bx, by = self._segment_arrays(backbone_edges, snapshot)
        ax, ay = self._segment_arrays(active_edges, snapshot)
        self.backbone_item.setData(x=bx, y=by)
        self.active_links_item.setData(x=ax, y=ay)

    def _render_nodes(self, snapshot: Snapshot) -> None:
        nodes = [snapshot.gateway, *snapshot.nodes]
        lats = [node.lat for node in nodes]
        lons = [node.lon for node in nodes]
        xs, ys = vectorized_wgs84_to_wm(lats, lons)
        spots: list[dict[str, object]] = []
        selected_x = []
        selected_y = []
        for index, node in enumerate(nodes):
            symbol = "star" if node.is_gateway else "o"
            size = 12
            brush = pg.mkBrush("#4dc96b") if node.is_gateway else pg.mkBrush("#59c2ff")
            pen = pg.mkPen("#0f2737", width=1.2)
            if not node.is_gateway and node.power_mode == PowerMode.PLUGGED:
                symbol = "s"
                brush = pg.mkBrush("#f5b849")
            if not node.is_gateway and node.antenna_shape == AntennaShape.SECTOR:
                symbol = "t1" if node.power_mode == PowerMode.BATTERY else "d"
            if not node.is_gateway and node.selected_as_relay:
                size = 16
                pen = pg.mkPen("#ff5fa2", width=2.0)
                brush = pg.mkBrush("#ff82b8" if node.power_mode == PowerMode.PLUGGED else "#f05ee6")
            if node.node_id == self.selected_node_id:
                selected_x = [xs[index]]
                selected_y = [ys[index]]
            spots.append(
                {
                    "pos": (float(xs[index]), float(ys[index])),
                    "data": node.node_id,
                    "symbol": symbol,
                    "size": 18 if node.is_gateway else size,
                    "brush": brush,
                    "pen": pen,
                }
            )
        self.node_item.setData(spots=spots)
        if selected_x:
            self.selection_item.setData(
                x=selected_x,
                y=selected_y,
                symbol="o",
                size=24,
                brush=pg.mkBrush(0, 0, 0, 0),
                pen=pg.mkPen("#ffe98a", width=2.4),
            )
        else:
            self.selection_item.setData([], [])

    def _render_ranges(self, snapshot: Snapshot) -> None:
        if not self.show_ranges_check.isChecked():
            self.ambient_ranges_item.set_polygons([])
            self.relay_ranges_item.set_polygons([])
            self.focus_range_item.set_polygons([])
            return

        ambient_polygons: list[list[tuple[float, float]]] = []
        relay_polygons: list[list[tuple[float, float]]] = []
        focus_polygons: list[list[tuple[float, float]]] = []
        for node in snapshot.nodes:
            polygon = self._polygon_to_wm(coverage_polygon(node, steps=26 if node.antenna_shape == AntennaShape.CIRCLE else 18))
            if node.node_id == self.selected_node_id:
                focus_polygons.append(polygon)
            if node.selected_as_relay:
                relay_polygons.append(polygon)
            else:
                ambient_polygons.append(polygon)
        self.ambient_ranges_item.set_polygons(ambient_polygons)
        self.relay_ranges_item.set_polygons(relay_polygons)
        self.focus_range_item.set_polygons(focus_polygons)

    def _update_metrics(self, snapshot: Snapshot) -> None:
        metrics = snapshot.metrics
        self.metric_labels["Epoch"].setText(f"{snapshot.epoch} / tick {snapshot.tick}")
        self.metric_labels["Nodes / Relays"].setText(f"{metrics['nodes']} / {metrics['relays']}")
        self.metric_labels["Delivery Ratio"].setText(f"{float(metrics['pdr']) * 100.0:5.1f}%")
        self.metric_labels["Average Latency"].setText(f"{float(metrics['avg_latency_ms']):.0f} ms")
        self.metric_labels["Average Battery"].setText(f"{float(metrics['avg_battery']) * 100.0:5.1f}%")
        self.metric_labels["Average Trust"].setText(f"{float(metrics['avg_trust']) * 100.0:5.1f}%")
        self.metric_labels["Relay Load"].setText(f"{float(metrics['relay_load']):.2f}")
        self.metric_labels["Packets"].setText(
            f"{metrics['total_delivered']}/{metrics['total_sent']} delivered, {metrics['total_dropped']} dropped"
        )
        solver_text = snapshot.optimization.status
        if snapshot.optimization.used_fallback:
            solver_text += " (heuristic fallback)"
        self.metric_labels["Solver"].setText(solver_text)

    def _update_selected(self, snapshot: Snapshot) -> None:
        node = self._selected_node(snapshot)
        role = "Gateway" if node.is_gateway else ("Relay" if node.selected_as_relay else "Edge Node")
        power = "Plugged" if node.power_mode == PowerMode.PLUGGED else "Battery"
        antenna = "Omni" if node.antenna_shape == AntennaShape.CIRCLE else f"Sector {node.antenna_beam_width_deg:.0f}°"
        self.selected_labels["Name"].setText(node.name)
        self.selected_labels["Zone"].setText(node.zone)
        self.selected_labels["Role"].setText(role)
        self.selected_labels["Power"].setText(power)
        self.selected_labels["Battery"].setText(f"{node.battery_level * 100.0:5.1f}%")
        self.selected_labels["Trust"].setText(f"{node.trust_score * 100.0:5.1f}%")
        self.selected_labels["Link Speed"].setText(f"{node.transfer_speed_kbps:4.1f} kbps")
        self.selected_labels["Range"].setText(f"{node.transfer_range_m:,.0f} m")
        self.selected_labels["Antenna"].setText(antenna)
        self.selected_labels["Profile"].setText(node.sensor_profile)

    def _update_relay_table(self, snapshot: Snapshot) -> None:
        relays = [node for node in snapshot.nodes if node.selected_as_relay]
        relays.sort(key=lambda item: item.last_load, reverse=True)
        relays = relays[:6]
        self.relay_table.setRowCount(len(relays))
        for row, node in enumerate(relays):
            values = [
                node.name,
                node.zone,
                f"{node.last_load:.2f}",
                f"{node.battery_level * 100.0:5.1f}%",
                f"{node.trust_score * 100.0:5.1f}%",
            ]
            for column, value in enumerate(values):
                self.relay_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def _update_events(self, snapshot: Snapshot) -> None:
        self.event_list.clear()
        for message in snapshot.event_log[:20]:
            self.event_list.addItem(message)

    def _reset_history(self) -> None:
        self._history_last_tick = None
        for history in (
            self._history_tick,
            self._history_pdr,
            self._history_latency,
            self._history_battery,
            self._history_trust,
            self._history_sent,
            self._history_delivered,
            self._history_dropped,
            self._history_relays,
        ):
            history.clear()

    def _record_history(self, snapshot: Snapshot) -> None:
        if self._history_last_tick == snapshot.tick:
            return
        metrics = snapshot.metrics
        self._history_last_tick = snapshot.tick
        self._history_tick.append(snapshot.tick)
        self._history_pdr.append(float(metrics["pdr"]) * 100.0)
        self._history_latency.append(float(metrics["avg_latency_ms"]))
        self._history_battery.append(float(metrics["avg_battery"]) * 100.0)
        self._history_trust.append(float(metrics["avg_trust"]) * 100.0)
        self._history_sent.append(float(metrics["total_sent"]))
        self._history_delivered.append(float(metrics["total_delivered"]))
        self._history_dropped.append(float(metrics["total_dropped"]))
        self._history_relays.append(float(metrics["relays"]))

    def _update_graphs(self, snapshot: Snapshot) -> None:
        x = list(self._history_tick)
        if not x:
            return
        self.reliability_pdr_curve.setData(x=x, y=list(self._history_pdr))
        self.reliability_relay_curve.setData(x=x, y=list(self._history_relays))
        self.latency_curve.setData(x=x, y=list(self._history_latency))
        self.energy_battery_curve.setData(x=x, y=list(self._history_battery))
        self.energy_trust_curve.setData(x=x, y=list(self._history_trust))
        self.volume_sent_curve.setData(x=x, y=list(self._history_sent))
        self.volume_delivered_curve.setData(x=x, y=list(self._history_delivered))
        self.volume_dropped_curve.setData(x=x, y=list(self._history_dropped))
        self.graph_summary_label.setText(
            f"Tick {snapshot.tick}: PDR {float(snapshot.metrics['pdr']) * 100.0:4.1f}% | "
            f"Latency {float(snapshot.metrics['avg_latency_ms']):.0f} ms | "
            f"Battery {float(snapshot.metrics['avg_battery']) * 100.0:4.1f}% | "
            f"Trust {float(snapshot.metrics['avg_trust']) * 100.0:4.1f}%"
        )

    def _selected_node(self, snapshot: Snapshot) -> Node:
        if self.selected_node_id == 0:
            return snapshot.gateway
        for node in snapshot.nodes:
            if node.node_id == self.selected_node_id:
                return node
        return snapshot.gateway

    def _default_selected_node_id(self) -> int:
        snapshot = self.simulation.last_snapshot
        if snapshot and snapshot.optimization.selected_relays:
            return min(snapshot.optimization.selected_relays)
        return 0

    def _on_map_hover(self, lat: float, lon: float) -> None:
        self.coord_label.setText(f"{lat:.5f}, {lon:.5f}")

    def _on_map_clicked(self, lat: float, lon: float) -> None:
        snapshot = self.snapshot
        if snapshot is None:
            return
        nearest_id = 0
        nearest_distance = haversine_m(lat, lon, snapshot.gateway.lat, snapshot.gateway.lon)
        for node in snapshot.nodes:
            distance = haversine_m(lat, lon, node.lat, node.lon)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_id = node.node_id
        if nearest_distance <= 1_600.0:
            self.selected_node_id = nearest_id
            self._refresh(snapshot)

    def _segment_arrays(self, edges: list[tuple[int, int, float]], snapshot: Snapshot) -> tuple[list[float], list[float]]:
        lat_points: list[float] = []
        lon_points: list[float] = []
        lookup = {node.node_id: node for node in snapshot.nodes}
        for source_id, target_id, _weight in edges:
            source = snapshot.gateway if source_id == 0 else lookup[source_id]
            target = snapshot.gateway if target_id == 0 else lookup[target_id]
            lat_points.extend([source.lat, target.lat, float("nan")])
            lon_points.extend([source.lon, target.lon, float("nan")])
        if not lat_points:
            return [], []
        xs, ys = vectorized_wgs84_to_wm(lat_points, lon_points)
        return xs.tolist(), ys.tolist()

    def _polygon_to_wm(self, polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
        lats = [point[0] for point in polygon]
        lons = [point[1] for point in polygon]
        xs, ys = vectorized_wgs84_to_wm(lats, lons)
        return list(zip(xs.tolist(), ys.tolist(), strict=True))

    def _empty_heatmap(self) -> np.ndarray:
        if self.snapshot is not None:
            return cast(np.ndarray, self.snapshot.heatmap_rgba)
        resolution = self.simulation.config.heatmap_resolution
        return np.zeros((resolution, resolution, 4), dtype=np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sofia LoRaWAN mesh simulator")
    parser.add_argument("--nodes", type=int, default=108)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--interval-ms", type=int, default=850)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--ticks", type=int, default=4)
    args = parser.parse_args(argv)

    if args.smoke_test:
        MapWidget.update_map_tiles = lambda self: None  # type: ignore[method-assign]

    app = QtWidgets.QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = MeshWindow(node_count=args.nodes, seed=args.seed, interval_ms=args.interval_ms, auto_start=not args.smoke_test)

    if args.smoke_test:
        window.show()
        app.processEvents()
        for _ in range(max(1, args.ticks)):
            window._advance_once()
            app.processEvents()
        window.close()
        app.processEvents()
        return 0

    window.show()
    return app.exec()

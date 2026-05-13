from __future__ import annotations

import base64
import os
import sys

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QByteArray, QMimeData, QTimer, Signal
from PySide6.QtGui import QColor, QDrag, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidgetItem,
    QListWidget,
    QMainWindow,
    QMenu,
    QSplitter,
    QSplashScreen,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from assembly_tree import AssemblyTreePanel, MIME_BRANCH, MIME_DATASET
from grim_dataset import RcsGrid
from grim_cut_dataset_mixin import DatasetOpsMixin
from grim_cut_plot_mixin import PlotOpsMixin
from plot_models import PlotContext

BLUE_PALETTE = {
    "is_dark": True,
    "win_bg": "#0f172a",
    "panel_bg": "#0b1222",
    "text": "#dbeafe",
    "head_bg": "#172554",
    "border": "#1e3a8a",
    "hover": "#1d4ed8",
    "checked_bg": "#2563eb",
    "checked_border": "#3b82f6",
    "grid": "#475569",
    "fg": "#dbeafe",
}
SPLASH_DURATION_MS = 4000


def _branch_arrow_uri(points: str, fill: str) -> str:
    """Return a base64 SVG data-URI for a small polygon arrow (used in QSS branch rules)."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">'
        f'<polygon points="{points}" fill="{fill}"/>'
        f'</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def build_qss(palette: dict[str, str]) -> str:
    arrow_right = _branch_arrow_uri("2,1 6,4 2,7", palette["text"])   # collapsed
    arrow_down  = _branch_arrow_uri("1,2 7,2 4,6", palette["text"])   # expanded
    return f"""
    QMainWindow {{ background: {palette['win_bg']}; }}
    QFrame {{ background: {palette['panel_bg']}; border: 1px solid {palette['border']}; border-radius: 8px; }}
    QFrame#paramSeparator {{
        background: {palette['border']}; min-width: 2px; max-width: 2px; border: none; border-radius: 0px;
    }}
    QGroupBox {{ color: {palette['text']}; border: 1px solid {palette['border']}; border-radius: 8px; margin-top: 10px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
    QLabel {{ color: {palette['text']}; }}
    QTableWidget {{
        background: {palette['panel_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']}; gridline-color: {palette['grid']};
    }}
    QHeaderView::section {{ background: {palette['head_bg']}; color: {palette['text']}; border: none; padding: 6px; }}
    QTabWidget::pane {{ border: 1px solid {palette['border']}; background: {palette['panel_bg']}; }}
    QTabBar::tab {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; border-bottom: 0; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
    QTabBar::tab:selected {{ background: {palette['head_bg']}; color: {palette['text']}; border-color: {palette['checked_border']}; }}
    QTabBar::tab:hover {{ background: {palette['hover']}; }}
    QListWidget {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; }}
    QTreeWidget {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; }}
    QTreeWidget::item {{ border-bottom: 1px solid {palette['grid']}; padding: 3px 4px; }}
    QTreeWidget::item:selected {{ background: {palette['checked_bg']}; color: white; }}
    QTreeWidget::branch {{ background: {palette['panel_bg']}; }}
    QTreeWidget::branch:has-children:!open {{ image: url("{arrow_right}"); }}
    QTreeWidget::branch:has-children:open  {{ image: url("{arrow_down}"); }}
    QTreeWidget#assemblyTree::branch:has-children {{ image: none; }}
    QListWidget::item {{ border-bottom: 1px solid {palette['grid']}; padding: 4px 6px; }}
    QListWidget QLineEdit {{
        background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']};
        padding: 2px 4px; min-height: 20px; font-size: 12px;
    }}
    QListWidget::item:selected {{
        background: {palette['checked_bg']}; color: white; border-bottom: 1px solid {palette['grid']};
    }}
    QToolButton, QDoubleSpinBox, QCheckBox, QLineEdit, QComboBox {{
        background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']};
        border-radius: 6px; padding: 6px;
    }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {palette['border']};
        border-radius: 3px;
        background: {palette['panel_bg']};
    }}
    QCheckBox::indicator:checked {{
        background: {palette['checked_bg']};
        border-color: {palette['checked_border']};
    }}
    QToolButton:hover {{ border-color: {palette['hover']}; }}
    QToolButton:checked {{ background: {palette['checked_bg']}; color: white; border-color: {palette['checked_border']}; }}
    QComboBox QAbstractItemView {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; }}
    QTableWidget::item:selected {{ background: {palette['checked_bg']}; color: white; }}
    QLabel#hoverReadout {{
        background: {palette['head_bg']}; color: {palette['text']}; border: 1px solid {palette['border']};
        border-radius: 4px; padding: 2px 6px; font-family: "Consolas","Courier New",monospace; font-size: 11px;
    }}
    """


def _extract_supported_drop_paths(mime: QMimeData) -> list[str]:
    if not mime.hasUrls():
        return []
    paths: list[str] = []
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        path = url.toLocalFile()
        if path.lower().endswith((".grim", ".csv", ".txt", ".out", ".pio", ".cmplx_di")):
            paths.append(path)
    return paths


class DatasetTable(QTableWidget):
    files_dropped = Signal(list)
    # branch_name: str, list of (name: str, grid: RcsGrid | None) tuples
    assembly_branch_dropped = Signal(str, list)
    rows_reordered = Signal()

    def __init__(self, rows: int, columns: int, parent: QWidget | None = None) -> None:
        super().__init__(rows, columns, parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self._pending_drag_data: tuple | None = None  # (name, RcsGrid|None)
        self._pending_drag_rows: list[int] = []

    def startDrag(self, _) -> None:
        rows = sorted({item.row() for item in self.selectedItems()})
        if not rows:
            return
        entries = []
        for row in rows:
            name_item = self.item(row, 0)
            if name_item is not None:
                entries.append((name_item.text(), name_item.data(Qt.UserRole)))
        if not entries:
            return
        self._pending_drag_data = entries  # list of (name, RcsGrid|None)
        self._pending_drag_rows = rows
        mime = QMimeData()
        mime.setData(MIME_DATASET, QByteArray(entries[0][0].encode("utf-8")))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction | Qt.MoveAction)
        self._pending_drag_data = None
        self._pending_drag_rows = []

    def dragEnterEvent(self, event) -> None:
        mime = event.mimeData()
        if event.source() is self:
            event.acceptProposedAction()
        elif mime.hasUrls() or mime.hasFormat(MIME_BRANCH):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        mime = event.mimeData()
        if event.source() is self:
            event.acceptProposedAction()
        elif mime.hasUrls() or mime.hasFormat(MIME_BRANCH):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        mime = event.mimeData()
        if event.source() is self and self._pending_drag_rows:
            self._reorder_to_drop(event)
            event.acceptProposedAction()
            return
        if mime.hasFormat(MIME_BRANCH):
            src = event.source()
            if hasattr(src, "_pending_branch_data") and src._pending_branch_data:
                branch_name = bytes(mime.data(MIME_BRANCH)).decode("utf-8")
                self.assembly_branch_dropped.emit(branch_name, src._pending_branch_data)
            event.acceptProposedAction()
        elif mime.hasUrls():
            paths = [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]
            if paths:
                self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def _reorder_to_drop(self, event) -> None:
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        drop_index = self.indexAt(pos)
        if drop_index.isValid():
            target_row = drop_index.row()
            if self.dropIndicatorPosition() == QAbstractItemView.BelowItem:
                target_row += 1
        else:
            target_row = self.rowCount()

        src_rows = sorted(set(self._pending_drag_rows))
        if not src_rows:
            return
        # No-op if dropping onto the same contiguous range.
        if src_rows[0] <= target_row <= src_rows[-1] + 1 and src_rows == list(range(src_rows[0], src_rows[-1] + 1)):
            return

        col_count = self.columnCount()
        # Snapshot rows to move (items only; row indices change as we remove).
        snapshots: list[list[QTableWidgetItem | None]] = []
        for r in src_rows:
            snapshots.append([self.takeItem(r, c) for c in range(col_count)])

        # Remove source rows bottom-up; adjust target for rows removed above it.
        for r in reversed(src_rows):
            self.removeRow(r)
            if r < target_row:
                target_row -= 1

        # Insert at target in original order.
        for offset, row_items in enumerate(snapshots):
            insert_at = target_row + offset
            self.insertRow(insert_at)
            for c, item in enumerate(row_items):
                if item is not None:
                    self.setItem(insert_at, c, item)

        self.clearSelection()
        if snapshots:
            self.setCurrentCell(target_row, 0)
            selection = self.selectionModel()
            for offset in range(len(snapshots)):
                idx = self.model().index(target_row + offset, 0)
                selection.select(
                    idx,
                    selection.Select | selection.Rows,
                )
        self.rows_reordered.emit()


class ClickableLabel(QLabel):
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
        else:
            super().mouseDoubleClickEvent(event)


class GrimCutWindow(DatasetOpsMixin, PlotOpsMixin, QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.palette = BLUE_PALETTE

        self.setWindowTitle("GRIM Cut")
        self.resize(1550, 900)

        right = QWidget()
        self.setCentralWidget(right)

        # ---------- Main panel ----------
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.main_tabs = QTabWidget()
        right_layout.addWidget(self.main_tabs, 1)
        self._tab_key_for_index: dict[int, str] = {}
        self._plot_splitters: dict[str, QSplitter] = {}
        self._plot_contexts: dict[str, PlotContext] = {}
        self._plot_controls_by_tab: dict[str, dict[str, QToolButton]] = {}
        self._plot_ops_index_by_tab: dict[str, int] = {}
        self._active_plot_tab = "plotting"

        self.tab_simple_plots = QWidget()
        simple_layout = QVBoxLayout(self.tab_simple_plots)
        simple_layout.setContentsMargins(10, 10, 10, 10)
        simple_layout.setSpacing(0)

        plot_splitter = QSplitter(Qt.Horizontal)
        simple_layout.addWidget(plot_splitter, 1)
        self._plot_splitters["plotting"] = plot_splitter

        left_panel = QWidget()
        right_panel = QWidget()
        plot_splitter.addWidget(left_panel)
        plot_splitter.addWidget(right_panel)
        plot_splitter.setStretchFactor(0, 1)
        plot_splitter.setStretchFactor(1, 0)

        self._plot_contexts["plotting"] = self._build_plot_left_context(left_panel)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        ops_group = QGroupBox("Dataset Operations")
        ops_vbox = QVBoxLayout(ops_group)
        ops_vbox.setSpacing(4)

        ops_row1 = QHBoxLayout()
        for label, attr in (
            ("Coherent +", "btn_coherent_add"),
            ("Coherent -", "btn_coherent_sub"),
            ("Coherent ÷", "btn_coherent_div"),
            ("Incoherent +", "btn_incoherent_add"),
            ("Incoherent -", "btn_incoherent_sub"),
            ("Slice", "btn_slice"),
            ("Stats", "btn_stats"),
            ("Join", "btn_join"),
            ("Overlap", "btn_overlap"),
        ):
            btn = QToolButton(text=label)
            setattr(self, attr, btn)
            ops_row1.addWidget(btn)
        ops_row1.addStretch(1)

        ops_row2 = QHBoxLayout()
        for label, attr in (
            ("Align", "btn_align"),
            ("Mirror", "btn_mirror"),
            ("Shift", "btn_shift"),
            ("Round", "btn_round"),
            ("Offset", "btn_offset"),
            ("Medianize", "btn_medianize"),
            ("Duplicate", "btn_duplicate"),
        ):
            btn = QToolButton(text=label)
            setattr(self, attr, btn)
            ops_row2.addWidget(btn)
        ops_row2.addStretch(1)

        ops_row3 = QHBoxLayout()
        for label, attr in (
            ("El->Az360", "btn_el_to_az360"),
            ("Swap El/Az", "btn_swap_el_az"),
            ("→ dBke", "btn_to_dbke"),
            ("→ dBsm", "btn_to_dbsm"),
            ("Conic ↔ GC", "btn_conic_gc"),
        ):
            btn = QToolButton(text=label)
            setattr(self, attr, btn)
            ops_row3.addWidget(btn)
        ops_row3.addStretch(1)

        ops_vbox.addLayout(ops_row1)
        ops_vbox.addLayout(ops_row2)
        ops_vbox.addLayout(ops_row3)
        right_layout.addWidget(ops_group)

        plot_group = QGroupBox("Plot Operations")
        plot_container = QVBoxLayout(plot_group)
        self.plot_ops_stack = QStackedWidget()

        def _add_plot_ops_page(
            tab_key: str,
            row1_specs: tuple[tuple[str, str], ...],
            row2_specs: tuple[tuple[str, str], ...],
        ) -> None:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(6)
            row1 = QHBoxLayout()
            row2 = QHBoxLayout()
            controls: dict[str, QToolButton] = {}

            def _make_plot_button(label: str, role: str) -> QToolButton:
                btn = QToolButton(text=label)
                if role in ("hold", "auto_plot", "pbp", "phase", "zoom_box"):
                    btn.setCheckable(True)
                controls[role] = btn
                return btn

            for label, role in row1_specs:
                row1.addWidget(_make_plot_button(label, role))
            row1.addStretch(1)

            for label, role in row2_specs:
                row2.addWidget(_make_plot_button(label, role))
            row2.addStretch(1)

            page_layout.addLayout(row1)
            page_layout.addLayout(row2)
            self._plot_controls_by_tab[tab_key] = controls
            self._plot_ops_index_by_tab[tab_key] = self.plot_ops_stack.addWidget(page)

        _add_plot_ops_page(
            "plotting",
            (
                ("Hold", "hold"),
                ("Clear", "clear"),
                ("Azimuth (Rect)", "azimuth_rect"),
                ("Azimuth (Polar)", "azimuth_polar"),
                ("Frequency", "frequency"),
                ("Elevation Sweep", "elevation_sweep"),
                ("Waterfall", "waterfall"),
                ("Compare", "compare"),
            ),
            (
                ("Fit X", "fit_x"),
                ("Fit Y", "fit_y"),
                ("Fit Both", "fit_both"),
                ("Zoom Box", "zoom_box"),
                ("Auto Plot", "auto_plot"),
                ("PbP", "pbp"),
                ("Phase", "phase"),
            ),
        )
        _add_plot_ops_page(
            "isar",
            (
                ("Hold", "hold"),
                ("Clear", "clear"),
                ("ISAR Image", "isar_image"),
                ("Az. vs D.R.", "az_vs_range"),
            ),
            (
                ("Fit X", "fit_x"),
                ("Fit Y", "fit_y"),
                ("Fit Both", "fit_both"),
                ("Zoom Box", "zoom_box"),
                ("Auto Plot", "auto_plot"),
            ),
        )
        plot_container.addWidget(self.plot_ops_stack)
        right_layout.addWidget(plot_group)

        dataset_group = QGroupBox("Datasets")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setSpacing(6)

        dataset_actions = QHBoxLayout()
        self.btn_dataset_save = QToolButton(text="Save")
        self.btn_dataset_save_all = QToolButton(text="Save All")
        self.btn_dataset_delete = QToolButton(text="Delete")
        dataset_actions.addWidget(self.btn_dataset_save)
        dataset_actions.addWidget(self.btn_dataset_save_all)
        dataset_actions.addWidget(self.btn_dataset_delete)
        dataset_actions.addStretch(1)
        dataset_layout.addLayout(dataset_actions)

        self.table = DatasetTable(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "File", "History"])
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        dataset_layout.addWidget(self.table, 1)

        params_group = QGroupBox("Available Parameters (selected dataset)")
        params_layout = QGridLayout(params_group)
        params_layout.setColumnStretch(0, 1)
        params_layout.setColumnStretch(2, 1)
        params_layout.setColumnStretch(4, 1)
        params_layout.setColumnStretch(6, 1)

        def _make_param_separator() -> QFrame:
            sep = QFrame()
            sep.setObjectName("paramSeparator")
            sep.setFixedWidth(2)
            return sep

        sep1 = _make_param_separator()
        sep2 = _make_param_separator()
        sep3 = _make_param_separator()
        self.list_pol = QListWidget()
        self.list_freq = QListWidget()
        self.list_elev = QListWidget()
        self.list_az = QListWidget()
        for widget in (self.list_pol, self.list_freq, self.list_elev, self.list_az):
            widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
            widget.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        lbl_pol = ClickableLabel("Polarization")
        lbl_freq = ClickableLabel("Frequency (GHz)")
        lbl_elev = ClickableLabel("Elevation (deg)")
        lbl_az = ClickableLabel("Azimuth (deg)")
        params_layout.addWidget(lbl_pol, 0, 0)
        params_layout.addWidget(sep1, 0, 1, 2, 1)
        params_layout.addWidget(lbl_freq, 0, 2)
        params_layout.addWidget(sep2, 0, 3, 2, 1)
        params_layout.addWidget(lbl_elev, 0, 4)
        params_layout.addWidget(sep3, 0, 5, 2, 1)
        params_layout.addWidget(lbl_az, 0, 6)
        params_layout.addWidget(self.list_pol, 1, 0)
        params_layout.addWidget(self.list_freq, 1, 2)
        params_layout.addWidget(self.list_elev, 1, 4)
        params_layout.addWidget(self.list_az, 1, 6)

        side_widget = QWidget()
        side_layout = QVBoxLayout(side_widget)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)
        side_layout.addWidget(params_group, 1)

        dataset_split = QHBoxLayout()
        dataset_split.setContentsMargins(0, 0, 0, 0)
        dataset_split.setSpacing(8)
        dataset_split.addWidget(dataset_group, 1)
        dataset_split.addWidget(side_widget, 1)
        right_layout.addLayout(dataset_split, 1)

        self._shared_right_panel = right_panel

        self.main_tabs.addTab(self.tab_simple_plots, "Plotting")
        self._tab_key_for_index[self.main_tabs.count() - 1] = "plotting"

        self.tab_isar = QWidget()
        isar_layout = QVBoxLayout(self.tab_isar)
        isar_layout.setContentsMargins(10, 10, 10, 10)
        isar_layout.setSpacing(0)

        isar_splitter = QSplitter(Qt.Horizontal)
        isar_layout.addWidget(isar_splitter, 1)
        self._plot_splitters["isar"] = isar_splitter

        isar_left_panel = QWidget()
        isar_splitter.addWidget(isar_left_panel)
        isar_splitter.setStretchFactor(0, 1)

        isar_context = self._build_plot_left_context(isar_left_panel)
        self._plot_contexts["isar"] = isar_context

        self.main_tabs.addTab(self.tab_isar, "ISAR")
        self._tab_key_for_index[self.main_tabs.count() - 1] = "isar"

        self.status = self.statusBar()
        self.status.showMessage("Ready")

        self.active_dataset: RcsGrid | None = None
        self._dataset_selection_order: list[int] = []
        self.last_plot_mode: str | None = None
        self.btn_phase = None
        self.btn_zoom_box = None
        self.pbp_fill_mode = "gray"
        self.pbp_fill_gray = "#7a7a7a"
        self.pbp_heatmap_samples = 80

        self.setStyleSheet(build_qss(BLUE_PALETTE))
        self.table.files_dropped.connect(self._handle_files_dropped)
        self.table.assembly_branch_dropped.connect(self._on_assembly_branch_dropped)
        self.table.rows_reordered.connect(self._on_dataset_rows_reordered)
        for context in self._plot_contexts.values():
            context.assembly_tree_panel.files_to_load.connect(self._handle_files_dropped)
            context.assembly_tree_panel.platform_built.connect(self._on_platform_built)
        self.table.itemSelectionChanged.connect(self._on_dataset_selection_changed)
        self.table.customContextMenuRequested.connect(self._on_dataset_context_menu)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self._on_dataset_header_double_clicked)
        for context in self._plot_contexts.values():
            context.plot_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
            context.plot_canvas.customContextMenuRequested.connect(self._on_plot_context_menu)
            context.plot_canvas.mpl_connect("scroll_event", self._on_plot_scroll_zoom)
            context.plot_canvas.mpl_connect("button_press_event", self._on_plot_mouse_press)
            context.plot_canvas.mpl_connect("motion_notify_event", self._on_plot_mouse_move)
            context.plot_canvas.mpl_connect("button_release_event", self._on_plot_mouse_release)
        self.list_pol.itemSelectionChanged.connect(self._on_polarization_selection_changed)
        self.list_freq.itemSelectionChanged.connect(self._on_param_selection_changed)
        self.list_elev.itemSelectionChanged.connect(self._on_param_selection_changed)
        self.list_az.itemSelectionChanged.connect(self._on_param_selection_changed)
        self._connect_param_list(self.list_pol, "polarization")
        self._connect_param_list(self.list_freq, "frequency")
        self._connect_param_list(self.list_elev, "elevation")
        self._connect_param_list(self.list_az, "azimuth")
        lbl_pol.doubleClicked.connect(lambda: self.list_pol.selectAll())
        lbl_freq.doubleClicked.connect(lambda: self.list_freq.selectAll())
        lbl_elev.doubleClicked.connect(lambda: self.list_elev.selectAll())
        lbl_az.doubleClicked.connect(lambda: self.list_az.selectAll())

        for controls in self._plot_controls_by_tab.values():
            if "azimuth_rect" in controls:
                controls["azimuth_rect"].clicked.connect(self._plot_azimuth_rect)
            if "frequency" in controls:
                controls["frequency"].clicked.connect(self._plot_frequency)
            if "elevation_sweep" in controls:
                controls["elevation_sweep"].clicked.connect(self._plot_elevation_sweep)
            if "waterfall" in controls:
                controls["waterfall"].clicked.connect(self._plot_waterfall)
            if "compare" in controls:
                controls["compare"].clicked.connect(self._plot_compare)
            if "clear" in controls:
                controls["clear"].clicked.connect(self._clear_plot)
            if "fit_x" in controls:
                controls["fit_x"].clicked.connect(self._fit_x)
            if "fit_y" in controls:
                controls["fit_y"].clicked.connect(self._fit_y)
            if "pbp" in controls:
                controls["pbp"].toggled.connect(self._on_pbp_toggled)
            if "azimuth_polar" in controls:
                controls["azimuth_polar"].clicked.connect(self._plot_azimuth_polar)
            if "isar_image" in controls:
                controls["isar_image"].clicked.connect(self._plot_isar_image)
            if "az_vs_range" in controls:
                controls["az_vs_range"].clicked.connect(self._plot_az_vs_range)
            if "fit_both" in controls:
                controls["fit_both"].clicked.connect(self._fit_both)
            if "phase" in controls:
                controls["phase"].toggled.connect(self._on_phase_toggled)
            if "zoom_box" in controls:
                controls["zoom_box"].toggled.connect(self._on_zoom_box_toggled)

        self.btn_coherent_add.clicked.connect(self._coherent_add_selected)
        self.btn_coherent_sub.clicked.connect(self._coherent_sub_selected)
        self.btn_coherent_div.clicked.connect(self._coherent_div_selected)
        self.btn_incoherent_add.clicked.connect(self._incoherent_add_selected)
        self.btn_incoherent_sub.clicked.connect(self._incoherent_sub_selected)
        self.btn_slice.clicked.connect(self._slice_selected)
        self.btn_stats.clicked.connect(self._statistics_selected)
        self.btn_join.clicked.connect(self._join_selected_datasets)
        self.btn_overlap.clicked.connect(self._overlap_selected_datasets)
        self.btn_align.clicked.connect(self._align_selected)
        self.btn_mirror.clicked.connect(self._mirror_selected)
        self.btn_shift.clicked.connect(self._shift_selected)
        self.btn_round.clicked.connect(self._round_selected)
        self.btn_offset.clicked.connect(self._offset_selected)
        self.btn_medianize.clicked.connect(self._medianize_selected)
        self.btn_duplicate.clicked.connect(self._duplicate_selected)
        self.btn_el_to_az360.clicked.connect(self._elevation_to_azimuth_360_selected)
        self.btn_swap_el_az.clicked.connect(self._swap_elevation_azimuth_selected)
        self.btn_to_dbke.clicked.connect(self._convert_to_dbke_selected)
        self.btn_to_dbsm.clicked.connect(self._convert_to_dbsm_selected)
        self.btn_conic_gc.clicked.connect(self._convert_conic_gc_selected)
        self.btn_dataset_save.clicked.connect(self._save_selected_datasets)
        self.btn_dataset_save_all.clicked.connect(self._save_all_datasets)
        self.btn_dataset_delete.clicked.connect(self._delete_selected_datasets)
        for tab_key, context in self._plot_contexts.items():
            context.btn_assembly_tree.toggled.connect(context.assembly_tree_panel.setVisible)
            context.btn_settings.toggled.connect(context.settings_frame.setVisible)
            context.btn_export_plot.clicked.connect(self._export_plot)
            context.chk_plot_legend.toggled.connect(self._update_legend_visibility)
            context.btn_plot_bg.clicked.connect(lambda _=False, which="bg": self._choose_plot_color(which))
            context.btn_plot_grid.clicked.connect(
                lambda _=False, which="grid": self._choose_plot_color(which)
            )
            context.btn_plot_text.clicked.connect(
                lambda _=False, which="text": self._choose_plot_color(which)
            )
            context.combo_polar_zero.currentIndexChanged.connect(self._on_polar_zero_changed)
            context.btn_isar_apply.clicked.connect(self._on_isar_window_changed)
            # On the ISAR tab every settings-frame change runs a full back-
            # projection / FFT, which is too slow for per-keystroke spinbox
            # updates. Defer everything through the Apply button instead.
            # The plotting tab keeps live updates because its plots are cheap.
            if tab_key == "isar":
                continue
            context.spin_plot_xmin.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_xmax.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_ymin.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_ymax.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_xstep.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_ystep.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_zmin.valueChanged.connect(self._on_waterfall_style_changed)
            context.spin_plot_zmax.valueChanged.connect(self._on_waterfall_style_changed)
            context.spin_plot_zstep.valueChanged.connect(self._on_waterfall_style_changed)
            context.combo_plot_scale.currentIndexChanged.connect(self._on_plot_scale_changed)
            context.combo_colormap.currentTextChanged.connect(self._on_colormap_changed)
            context.chk_colorbar.toggled.connect(self._on_waterfall_style_changed)
            context.chk_colorbar_shared.toggled.connect(self._on_waterfall_style_changed)
            context.chk_plot_grid_visible.toggled.connect(self._apply_plot_theme)
            context.chk_colormap_invert.toggled.connect(self._on_colormap_changed)
            context.combo_isar_window.currentIndexChanged.connect(self._on_isar_window_changed)
            context.combo_isar_units.currentIndexChanged.connect(self._on_isar_window_changed)
            context.combo_isar_algorithm.currentIndexChanged.connect(self._on_isar_window_changed)
            context.chk_isar_az_interp.toggled.connect(self._on_isar_window_changed)
            context.spin_isar_az_min.valueChanged.connect(self._on_isar_window_changed)
            context.spin_isar_az_max.valueChanged.connect(self._on_isar_window_changed)
            context.spin_isar_az_step.valueChanged.connect(self._on_isar_window_changed)
            context.chk_isar_square.toggled.connect(self._on_isar_window_changed)

        self._activate_plot_tab("plotting")
        self.main_tabs.currentChanged.connect(self._on_main_tab_changed)
        self._update_plot_color_buttons()

    def dragEnterEvent(self, event) -> None:
        if _extract_supported_drop_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if _extract_supported_drop_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = _extract_supported_drop_paths(event.mimeData())
        if paths:
            self._handle_files_dropped(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def _build_plot_left_context(self, panel: QWidget) -> PlotContext:
        left_layout = QVBoxLayout(panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        topbar = QHBoxLayout()
        topbar.addStretch(1)
        btn_assembly_tree = QToolButton(text="Assembly Tree")
        btn_assembly_tree.setCheckable(True)
        btn_export_plot = QToolButton(text="Export Plot")
        btn_settings = QToolButton(text="Plot Settings")
        btn_settings.setCheckable(True)
        topbar.addWidget(btn_assembly_tree)
        topbar.addWidget(btn_export_plot)
        topbar.addWidget(btn_settings)
        left_layout.addLayout(topbar)

        settings_frame = QFrame()
        settings_frame.setFrameShape(QFrame.StyledPanel)
        settings_frame.setVisible(False)
        settings_layout = QGridLayout(settings_frame)
        settings_layout.setHorizontalSpacing(8)
        settings_layout.setVerticalSpacing(6)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)
        settings_layout.setColumnStretch(5, 1)

        row = 0
        settings_layout.addWidget(QLabel("Plot X Min"), row, 0)
        spin_plot_xmin = QDoubleSpinBox()
        spin_plot_xmin.setRange(-1e9, 1e9)
        spin_plot_xmin.setValue(-180.0)
        settings_layout.addWidget(spin_plot_xmin, row, 1)
        settings_layout.addWidget(QLabel("Plot X Max"), row, 2)
        spin_plot_xmax = QDoubleSpinBox()
        spin_plot_xmax.setRange(-1e9, 1e9)
        spin_plot_xmax.setValue(180.0)
        settings_layout.addWidget(spin_plot_xmax, row, 3)
        settings_layout.addWidget(QLabel("Plot X Step"), row, 4)
        spin_plot_xstep = QDoubleSpinBox()
        spin_plot_xstep.setRange(0.0, 1e9)
        spin_plot_xstep.setDecimals(6)
        spin_plot_xstep.setValue(0.0)
        settings_layout.addWidget(spin_plot_xstep, row, 5)
        row += 1

        settings_layout.addWidget(QLabel("Plot Y Min"), row, 0)
        spin_plot_ymin = QDoubleSpinBox()
        spin_plot_ymin.setRange(-1e9, 1e9)
        spin_plot_ymin.setValue(-80.0)
        settings_layout.addWidget(spin_plot_ymin, row, 1)
        settings_layout.addWidget(QLabel("Plot Y Max"), row, 2)
        spin_plot_ymax = QDoubleSpinBox()
        spin_plot_ymax.setRange(-1e9, 1e9)
        spin_plot_ymax.setValue(0.0)
        settings_layout.addWidget(spin_plot_ymax, row, 3)
        settings_layout.addWidget(QLabel("Plot Y Step"), row, 4)
        spin_plot_ystep = QDoubleSpinBox()
        spin_plot_ystep.setRange(0.0, 1e9)
        spin_plot_ystep.setDecimals(6)
        spin_plot_ystep.setValue(0.0)
        settings_layout.addWidget(spin_plot_ystep, row, 5)
        row += 1

        settings_layout.addWidget(QLabel("Plot Z Min"), row, 0)
        spin_plot_zmin = QDoubleSpinBox()
        spin_plot_zmin.setRange(-1e9, 1e9)
        spin_plot_zmin.setValue(0.0)
        settings_layout.addWidget(spin_plot_zmin, row, 1)
        settings_layout.addWidget(QLabel("Plot Z Max"), row, 2)
        spin_plot_zmax = QDoubleSpinBox()
        spin_plot_zmax.setRange(-1e9, 1e9)
        spin_plot_zmax.setValue(0.0)
        settings_layout.addWidget(spin_plot_zmax, row, 3)
        settings_layout.addWidget(QLabel("Plot Z Step"), row, 4)
        spin_plot_zstep = QDoubleSpinBox()
        spin_plot_zstep.setRange(0.0, 1e9)
        spin_plot_zstep.setDecimals(6)
        spin_plot_zstep.setValue(0.0)
        settings_layout.addWidget(spin_plot_zstep, row, 5)
        row += 1

        settings_layout.addWidget(QLabel("Plot Scale"), row, 0)
        combo_plot_scale = QComboBox()
        combo_plot_scale.addItem("dBsm", "dbsm")
        combo_plot_scale.addItem("Linear", "linear")
        default_index = combo_plot_scale.findData("dbsm")
        if default_index >= 0:
            combo_plot_scale.setCurrentIndex(default_index)
        settings_layout.addWidget(combo_plot_scale, row, 1, 1, 5)
        row += 1

        settings_layout.addWidget(QLabel("Polar 0° Direction"), row, 0)
        combo_polar_zero = QComboBox()
        polar_zero_options = [
            ("North", "N"),
            ("North East", "NE"),
            ("East", "E"),
            ("South East", "SE"),
            ("South", "S"),
            ("South West", "SW"),
            ("West", "W"),
            ("North West", "NW"),
        ]
        for label, loc in polar_zero_options:
            combo_polar_zero.addItem(label, loc)
        default_index = combo_polar_zero.findData("N")
        if default_index >= 0:
            combo_polar_zero.setCurrentIndex(default_index)
        settings_layout.addWidget(combo_polar_zero, row, 1, 1, 5)
        row += 1

        settings_layout.addWidget(QLabel("Colormap"), row, 0)
        combo_colormap = QComboBox()
        combo_colormap.addItems(
            ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"]
        )
        settings_layout.addWidget(combo_colormap, row, 1)
        chk_colorbar = QCheckBox("Show Colorbar")
        chk_colorbar.setChecked(True)
        settings_layout.addWidget(chk_colorbar, row, 2)
        chk_colorbar_shared = QCheckBox("Shared Colorbar")
        chk_colorbar_shared.setChecked(True)
        settings_layout.addWidget(chk_colorbar_shared, row, 3)
        row += 1

        settings_layout.addWidget(QLabel("ISAR Window"), row, 0)
        combo_isar_window = QComboBox()
        combo_isar_window.addItems([
            "Hanning",
            "Hamming",
            "Blackman",
            "Blackman-Harris",
            "Kaiser β=15",
            "Rectangular",
        ])
        settings_layout.addWidget(combo_isar_window, row, 1)
        settings_layout.addWidget(QLabel("ISAR Units"), row, 2)
        combo_isar_units = QComboBox()
        combo_isar_units.addItems(["m", "in", "ft"])
        settings_layout.addWidget(combo_isar_units, row, 3)
        row += 1

        settings_layout.addWidget(QLabel("ISAR Algorithm"), row, 0)
        combo_isar_algorithm = QComboBox()
        combo_isar_algorithm.addItems(["Back-Projection", "Polar Format"])
        combo_isar_algorithm.setToolTip(
            "Back-Projection: geometrically exact at any aperture (incl. 360°), slower.\n"
            "Polar Format: 2-D IFFT of the (θ, f) data treated as Cartesian k-space — "
            "fast, tolerates any aperture numerically, matches the absolute-dB convention "
            "of most FFT-based ISAR tools, but distorts geometry away from broadside."
        )
        settings_layout.addWidget(combo_isar_algorithm, row, 1, 1, 5)
        row += 1

        chk_isar_az_interp = QCheckBox("Interp Az")
        chk_isar_az_interp.setToolTip(
            "Resample azimuth onto a uniform grid before imaging. Periodic "
            "(360°-wrapping) when the source covers ≥359°; otherwise linear "
            "with zero-fill outside the source support."
        )
        settings_layout.addWidget(chk_isar_az_interp, row, 0)
        spin_isar_az_min = QDoubleSpinBox()
        spin_isar_az_min.setRange(-3600.0, 3600.0)
        spin_isar_az_min.setDecimals(4)
        spin_isar_az_min.setSingleStep(1.0)
        spin_isar_az_min.setValue(0.0)
        spin_isar_az_min.setToolTip("Lower azimuth limit (deg) for the uniform target grid.")
        settings_layout.addWidget(spin_isar_az_min, row, 1)
        spin_isar_az_max = QDoubleSpinBox()
        spin_isar_az_max.setRange(-3600.0, 3600.0)
        spin_isar_az_max.setDecimals(4)
        spin_isar_az_max.setSingleStep(1.0)
        spin_isar_az_max.setValue(360.0)
        spin_isar_az_max.setToolTip("Upper azimuth limit (deg) for the uniform target grid.")
        settings_layout.addWidget(spin_isar_az_max, row, 2)
        spin_isar_az_step = QDoubleSpinBox()
        spin_isar_az_step.setRange(1.0e-4, 90.0)
        spin_isar_az_step.setDecimals(4)
        spin_isar_az_step.setSingleStep(0.1)
        spin_isar_az_step.setValue(1.0)
        spin_isar_az_step.setToolTip("Azimuth step (deg) for the uniform target grid.")
        settings_layout.addWidget(spin_isar_az_step, row, 3)
        row += 1

        chk_isar_square = QCheckBox("Square Aspect")
        chk_isar_square.setToolTip(
            "Lock the image to equal cross-range / down-range scale and clip the "
            "visible window to a square centred on (0, 0). The square is sized to "
            "the smaller of the down-range / cross-range half-extents so the target "
            "fills the box and the geometry is undistorted. Off uses 'fill the axes' "
            "scaling, which packs more data on screen but stretches the geometry."
        )
        settings_layout.addWidget(chk_isar_square, row, 0, 1, 2)
        btn_isar_apply = QToolButton(text="Apply ISAR Settings")
        btn_isar_apply.setToolTip(
            "Re-render the ISAR image with the current settings. On the ISAR tab, "
            "settings changes are deferred until you click here so typing into "
            "spinboxes doesn't trigger a back-projection per keystroke."
        )
        settings_layout.addWidget(btn_isar_apply, row, 2, 1, 4)
        row += 1

        chk_plot_grid_visible = QCheckBox("Show Grid")
        chk_plot_grid_visible.setChecked(True)
        settings_layout.addWidget(chk_plot_grid_visible, row, 0)
        chk_colormap_invert = QCheckBox("Invert Colormap")
        chk_colormap_invert.setChecked(False)
        settings_layout.addWidget(chk_colormap_invert, row, 1)
        row += 1

        settings_layout.addWidget(QLabel("Plot Colors"), row, 0)
        btn_plot_bg = QToolButton(text="BG")
        btn_plot_grid = QToolButton(text="Grid")
        btn_plot_text = QToolButton(text="Text")
        settings_layout.addWidget(btn_plot_bg, row, 1)
        settings_layout.addWidget(btn_plot_grid, row, 2)
        settings_layout.addWidget(btn_plot_text, row, 3)
        row += 1

        chk_plot_legend = QCheckBox("Show Legend")
        chk_plot_legend.setChecked(True)
        settings_layout.addWidget(chk_plot_legend, row, 0, 1, 6)
        left_layout.addWidget(settings_frame)

        plot_frame = QFrame()
        plot_frame.setFrameShape(QFrame.StyledPanel)
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(20, 20, 20, 20)
        plot_layout.setSpacing(12)

        plot_figure = Figure(facecolor=BLUE_PALETTE["panel_bg"])
        plot_canvas = FigureCanvas(plot_figure)
        plot_canvas.setMinimumSize(320, 240)
        plot_canvas.setStyleSheet("background: transparent;")
        plot_ax = plot_figure.add_subplot(111)
        plot_ax.set_facecolor(BLUE_PALETTE["panel_bg"])
        plot_ax.grid(True, color=BLUE_PALETTE["grid"], alpha=0.35)
        plot_ax.tick_params(colors=BLUE_PALETTE["text"])
        plot_ax.xaxis.label.set_color(BLUE_PALETTE["text"])
        plot_ax.yaxis.label.set_color(BLUE_PALETTE["text"])
        for spine in plot_ax.spines.values():
            spine.set_color(BLUE_PALETTE["border"])
        plot_canvas.draw_idle()
        plot_layout.addWidget(plot_canvas, 1)
        hover_readout = QLabel("x: --   y: --")
        hover_readout.setObjectName("hoverReadout")
        hover_readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        plot_layout.addWidget(hover_readout, 0, Qt.AlignLeft)
        plot_canvas.mpl_connect(
            "motion_notify_event",
            lambda event, lbl=hover_readout: self._on_plot_hover(event, lbl),
        )
        plot_canvas.mpl_connect(
            "axes_leave_event",
            lambda event, lbl=hover_readout: self._reset_hover_readout(lbl),
        )
        plot_canvas.mpl_connect(
            "figure_leave_event",
            lambda event, lbl=hover_readout: self._reset_hover_readout(lbl),
        )

        assembly_tree_panel = AssemblyTreePanel()
        assembly_tree_panel.setVisible(False)

        inner_split = QSplitter(Qt.Horizontal)
        inner_split.addWidget(assembly_tree_panel)
        inner_split.addWidget(plot_frame)
        inner_split.setStretchFactor(0, 0)
        inner_split.setStretchFactor(1, 1)
        inner_split.setSizes([240, 9999])
        left_layout.addWidget(inner_split, 1)

        return PlotContext(
            btn_export_plot=btn_export_plot,
            btn_assembly_tree=btn_assembly_tree,
            btn_settings=btn_settings,
            settings_frame=settings_frame,
            assembly_tree_panel=assembly_tree_panel,
            spin_plot_xmin=spin_plot_xmin,
            spin_plot_xmax=spin_plot_xmax,
            spin_plot_xstep=spin_plot_xstep,
            spin_plot_ymin=spin_plot_ymin,
            spin_plot_ymax=spin_plot_ymax,
            spin_plot_ystep=spin_plot_ystep,
            spin_plot_zmin=spin_plot_zmin,
            spin_plot_zmax=spin_plot_zmax,
            spin_plot_zstep=spin_plot_zstep,
            combo_plot_scale=combo_plot_scale,
            combo_polar_zero=combo_polar_zero,
            combo_colormap=combo_colormap,
            chk_colorbar=chk_colorbar,
            chk_colorbar_shared=chk_colorbar_shared,
            chk_plot_grid_visible=chk_plot_grid_visible,
            chk_colormap_invert=chk_colormap_invert,
            combo_isar_window=combo_isar_window,
            combo_isar_units=combo_isar_units,
            combo_isar_algorithm=combo_isar_algorithm,
            chk_isar_az_interp=chk_isar_az_interp,
            spin_isar_az_min=spin_isar_az_min,
            spin_isar_az_max=spin_isar_az_max,
            spin_isar_az_step=spin_isar_az_step,
            chk_isar_square=chk_isar_square,
            btn_isar_apply=btn_isar_apply,
            btn_plot_bg=btn_plot_bg,
            btn_plot_grid=btn_plot_grid,
            btn_plot_text=btn_plot_text,
            chk_plot_legend=chk_plot_legend,
            hover_readout=hover_readout,
            plot_figure=plot_figure,
            plot_canvas=plot_canvas,
            plot_ax=plot_ax,
            plot_colorbars=[],
            plot_axes=None,
            plot_bg_color=None,
            plot_grid_color=None,
            plot_text_color=None,
            last_plot_mode=None,
        )

    def _move_shared_right_panel(self, tab_key: str) -> None:
        splitter = self._plot_splitters.get(tab_key)
        if splitter is None:
            return
        if splitter.indexOf(self._shared_right_panel) >= 0:
            return
        self._shared_right_panel.setParent(None)
        splitter.addWidget(self._shared_right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

    def _activate_plot_tab(self, tab_key: str) -> None:
        if tab_key not in self._plot_contexts:
            return
        previous = self._plot_contexts.get(self._active_plot_tab)
        if previous is not None:
            for field in PlotContext.__dataclass_fields__:
                if hasattr(self, field):
                    setattr(previous, field, getattr(self, field))

        self._active_plot_tab = tab_key
        self._move_shared_right_panel(tab_key)

        controls = self._plot_controls_by_tab[tab_key]
        self.plot_ops_stack.setCurrentIndex(self._plot_ops_index_by_tab[tab_key])
        self.btn_hold = controls.get("hold")
        self.btn_clear = controls.get("clear")
        self.btn_azimuth_rect = controls.get("azimuth_rect")
        self.btn_azimuth_polar = controls.get("azimuth_polar")
        self.btn_frequency = controls.get("frequency")
        self.btn_waterfall = controls.get("waterfall")
        self.btn_fit_x = controls.get("fit_x")
        self.btn_fit_y = controls.get("fit_y")
        self.btn_auto_plot = controls.get("auto_plot")
        self.btn_pbp = controls.get("pbp")
        self.btn_isar_image = controls.get("isar_image")
        self.btn_phase = controls.get("phase")
        self.btn_zoom_box = controls.get("zoom_box")

        context = self._plot_contexts[tab_key]
        for field in PlotContext.__dataclass_fields__:
            setattr(self, field, getattr(context, field))

    def _on_main_tab_changed(self, index: int) -> None:
        tab_key = self._tab_key_for_index.get(index)
        if tab_key is None:
            return
        self._activate_plot_tab(tab_key)
        self._update_plot_color_buttons()
        self.plot_canvas.draw_idle()

    def _connect_param_list(self, widget: QListWidget, axis_name: str) -> None:
        widget.itemChanged.connect(
            lambda item, axis=axis_name, lw=widget: self._on_param_item_changed(item, axis, lw)
        )

    def _on_assembly_branch_dropped(self, branch_name: str, leaf_data: list) -> None:
        """Build the dragged subtree honouring per-node add modes.

        We reach into the source AssemblyTree to recover the originating
        QTreeWidgetItem (the flat `leaf_data` list doesn't carry the
        coherent / incoherent structure). Falls back to the legacy flat
        coherent sum if the item isn't available — e.g. drag from a tree
        we can't introspect.
        """
        from assembly_tree import build_assembly_grid

        branch_item = None
        for context in self._plot_contexts.values():
            tree = getattr(context.assembly_tree_panel, "tree", None)
            if tree is None:
                continue
            candidate = getattr(tree, "_branch_drag_item", None)
            if candidate is not None and candidate.text(0) == branch_name:
                branch_item = candidate
                break

        if branch_item is not None:
            try:
                grid, history = build_assembly_grid(branch_item, axis_mode="intersect")
            except (ValueError, TypeError) as exc:
                self.status.showMessage(f"Assembly build failed: {exc}")
                return
            if grid is None:
                self.status.showMessage(
                    "Assembly branch: no loaded leaves in this subtree."
                )
                return
            self._add_dataset_row(grid, branch_name, history, file_name="")
            self.status.showMessage(f"Assembly built: {branch_name}")
            return

        # Legacy fallback: flat coherent sum of every dropped leaf.
        datasets = [(name, grid) for name, grid in leaf_data if isinstance(grid, RcsGrid)]
        skipped = len(leaf_data) - len(datasets)
        skip_msg = f" ({skipped} empty leaf(s) skipped)" if skipped else ""
        if not datasets:
            self.status.showMessage(
                "Assembly branch: no dataset data is stored in these leaves yet."
            )
            return
        if len(datasets) == 1:
            _, grid = datasets[0]
            self._add_dataset_row(grid, branch_name, f"Assembly (single): {branch_name}", file_name="")
            self.status.showMessage(f"Assembly: added {branch_name}{skip_msg}")
            return
        name_list = [n for n, _ in datasets]
        base = datasets[0][1]
        try:
            result = base.coherent_add_many(*[g for _, g in datasets[1:]])
        except (ValueError, TypeError) as exc:
            self.status.showMessage(f"Assembly coherent sum failed: {exc}")
            return
        history = "Assembly Coherent +: " + ", ".join(name_list)
        self._add_dataset_row(result, branch_name, history, file_name="")
        self.status.showMessage(f"Assembly coherent sum created: {branch_name}{skip_msg}")

    def _on_platform_built(self, name: str, grid, history: str) -> None:
        """Add a built-platform dataset to the table (signal from BuildDialog)."""
        if not isinstance(grid, RcsGrid):
            self.status.showMessage("Build platform: invalid grid returned.")
            return
        self._add_dataset_row(grid, name, history or f"Σ {name}", file_name="")
        self.status.showMessage(f"Built platform: {name}")


def main() -> int:
    app = QApplication(sys.argv)
    splash = None
    splash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GRIM.png")
    if os.path.exists(splash_path):
        splash_pixmap = QPixmap(splash_path)
        if not splash_pixmap.isNull():
            splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
            splash.show()
            app.processEvents()

    window = GrimCutWindow()
    window.show()
    if splash is not None:
        QTimer.singleShot(SPLASH_DURATION_MS, lambda: splash.finish(window))
    return app.exec()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())

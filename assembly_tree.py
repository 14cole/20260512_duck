from __future__ import annotations

import base64
import io
import json
import os

import numpy as np

from PySide6.QtCore import QByteArray, QMimeData, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDrag, QFont, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QRadioButton,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# MIME sent FROM the Datasets table TO the tree (single loaded dataset)
MIME_DATASET = "application/x-grim-dataset"

# MIME sent FROM the tree TO the Datasets table (branch drag marker)
MIME_BRANCH = "application/x-grim-branch"

# QTreeWidgetItem data roles
_ROLE_TYPE    = Qt.UserRole        # "root" | "branch" | "leaf"
_ROLE_NAME    = Qt.UserRole + 1    # dataset name string (leaves only)
_ROLE_GRID    = Qt.UserRole + 2    # RcsGrid object (leaves only, may be None)
_ROLE_MODE    = Qt.UserRole + 3    # "coh" | "incoh" (every non-root node)

_TYPE_ROOT   = "root"
_TYPE_BRANCH = "branch"
_TYPE_LEAF   = "leaf"

_MODE_COH    = "coh"
_MODE_INCOH  = "incoh"
_DEFAULT_MODE = _MODE_COH

_MODE_LABEL  = {_MODE_COH: "coh +", _MODE_INCOH: "inc +"}
_MODE_COLOUR = {_MODE_COH: QColor("#3b82f6"), _MODE_INCOH: QColor("#f59e0b")}


def _apply_mode_badge(item: QTreeWidgetItem, mode: str | None) -> None:
    """Paint the mode column for a node. Root nodes have no parent, so they
    have no add-mode — their second column stays blank."""
    if mode is None:
        item.setText(1, "")
        item.setForeground(1, QBrush())
        return
    item.setText(1, _MODE_LABEL.get(mode, ""))
    item.setForeground(1, QBrush(_MODE_COLOUR.get(mode, QColor("#9ca3af"))))
    font = item.font(1)
    font.setBold(True)
    item.setFont(1, font)


def _set_node_mode(item: QTreeWidgetItem, mode: str) -> None:
    """Persist a node's add-mode and refresh its badge. No-op for roots."""
    if item.data(0, _ROLE_TYPE) == _TYPE_ROOT:
        return
    if mode not in (_MODE_COH, _MODE_INCOH):
        mode = _DEFAULT_MODE
    item.setData(0, _ROLE_MODE, mode)
    _apply_mode_badge(item, mode)


def _node_mode(item: QTreeWidgetItem) -> str:
    """Read a non-root node's add-mode. Defaults to coherent."""
    raw = item.data(0, _ROLE_MODE)
    if raw in (_MODE_COH, _MODE_INCOH):
        return raw
    return _DEFAULT_MODE

# Icon pixel size
_ICON = 14


def _node_icon(node_type: str, expanded: bool = False, has_data: bool = True) -> QIcon:
    """
    Draw a small 14×14 icon that encodes both node type and state:
      Root (closed) — blue folder with tab
      Root (open)   — open folder (lighter, tab raised)
      Branch (closed) — teal rounded rect
      Branch (open)   — teal rounded rect, lighter fill
      Leaf (loaded)   — green circle
      Leaf (empty)    — grey circle  (also italicised/dimmed by _apply_leaf_style)
    """
    pix = QPixmap(_ICON, _ICON)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)

    if node_type == _TYPE_ROOT:
        body  = QColor("#3b82f6") if not expanded else QColor("#93c5fd")
        tab   = QColor("#1d4ed8") if not expanded else QColor("#60a5fa")
        p.setBrush(tab)
        p.drawRoundedRect(1, 1, 6, 4, 1, 1)      # folder tab
        p.setBrush(body)
        p.drawRoundedRect(1, 4, 12, 9, 2, 2)     # folder body

    elif node_type == _TYPE_BRANCH:
        color = QColor("#0891b2") if not expanded else QColor("#22d3ee")
        p.setBrush(color)
        p.drawRoundedRect(2, 2, 10, 10, 2, 2)

    else:  # leaf
        color = QColor("#34d399") if has_data else QColor("#6b7280")
        p.setBrush(color)
        p.drawEllipse(2, 2, 10, 10)

    p.end()
    return QIcon(pix)


# ─────────────────────────────────────────────────────────────────────────────
# RcsGrid ↔ base-64 helpers (mirrors RcsGrid.save / RcsGrid.load exactly,
# but works with BytesIO instead of a file path)
# ─────────────────────────────────────────────────────────────────────────────

def _grid_to_b64(grid) -> str:
    """Serialise an RcsGrid to a base-64 string."""
    buf = io.BytesIO()
    units_payload = json.dumps(grid.units) if grid.units else ""
    np.savez(
        buf,
        azimuths=grid.azimuths,
        elevations=grid.elevations,
        frequencies=grid.frequencies,
        polarizations=grid.polarizations,
        rcs_power=np.asarray(grid.rcs_power, dtype=np.float32),
        rcs_phase=np.asarray(grid.rcs_phase, dtype=np.float32),
        rcs_domain="power_phase",
        power_domain="linear_rcs",
        source_path=grid.source_path if grid.source_path is not None else "",
        history=grid.history if grid.history is not None else "",
        units=units_payload,
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_grid(b64: str):
    """Reconstruct an RcsGrid from a base-64 string."""
    from grim_dataset import RcsGrid
    buf = io.BytesIO(base64.b64decode(b64))
    data = np.load(buf, allow_pickle=False)

    units: dict = {}
    if "units" in data:
        raw = data["units"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str) and raw:
            try:
                units = json.loads(raw)
            except json.JSONDecodeError:
                units = {}

    source_path_raw = data["source_path"].item() if "source_path" in data else None
    source_path     = source_path_raw if source_path_raw else None
    history_raw     = data["history"].item() if "history" in data else None
    history         = history_raw if history_raw else None

    return RcsGrid(
        data["azimuths"],
        data["elevations"],
        data["frequencies"],
        data["polarizations"],
        rcs_power=data["rcs_power"],
        rcs_phase=data["rcs_phase"],
        rcs_domain="power_phase",
        source_path=source_path,
        history=history,
        units=units,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tree item serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _item_to_dict(item: QTreeWidgetItem) -> dict:
    node_type = item.data(0, _ROLE_TYPE)
    d: dict = {
        "name":     item.text(0),
        "type":     node_type,
        "children": [_item_to_dict(item.child(i)) for i in range(item.childCount())],
    }
    if node_type != _TYPE_ROOT:
        d["mode"] = _node_mode(item)
    if node_type == _TYPE_LEAF:
        d["dataset"] = item.data(0, _ROLE_NAME)
        grid = item.data(0, _ROLE_GRID)
        if grid is not None:
            try:
                d["data"] = _grid_to_b64(grid)
            except Exception:
                pass
    return d


def _dict_to_item(d: dict) -> QTreeWidgetItem:
    node_type = d.get("type", _TYPE_BRANCH)
    item = QTreeWidgetItem([d["name"]])
    item.setData(0, _ROLE_TYPE, node_type)
    _apply_flags(item, node_type)

    if node_type == _TYPE_LEAF:
        item.setData(0, _ROLE_NAME, d.get("dataset"))
        grid = None
        if "data" in d:
            try:
                grid = _b64_to_grid(d["data"])
            except Exception:
                grid = None
        item.setData(0, _ROLE_GRID, grid)
        _apply_leaf_style(item, grid is not None)
        item.setIcon(0, _node_icon(_TYPE_LEAF, has_data=(grid is not None)))
    else:
        item.setIcon(0, _node_icon(node_type, expanded=False))
        if node_type == _TYPE_ROOT:
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)

    if node_type == _TYPE_ROOT:
        _apply_mode_badge(item, None)
    else:
        _set_node_mode(item, d.get("mode", _DEFAULT_MODE))

    for child in d.get("children", []):
        item.addChild(_dict_to_item(child))
    return item


def _apply_flags(item: QTreeWidgetItem, node_type: str) -> None:
    base = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
    if node_type == _TYPE_LEAF:
        item.setFlags(base | Qt.ItemIsDragEnabled)
    else:
        item.setFlags(base | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled)


def _apply_leaf_style(item: QTreeWidgetItem, has_data: bool) -> None:
    """Italicise and dim leaves that have no RcsGrid loaded yet."""
    font = item.font(0)
    font.setItalic(not has_data)
    item.setFont(0, font)
    if has_data:
        item.setForeground(0, QBrush())          # default colour
    else:
        item.setForeground(0, QBrush(QColor("#888888")))
        item.setToolTip(0, "No dataset data loaded – drag from the Datasets table")


# ─────────────────────────────────────────────────────────────────────────────
# Tree widget
# ─────────────────────────────────────────────────────────────────────────────

class AssemblyTree(QTreeWidget):
    """
    Node hierarchy:   Root  →  Branch(es)  →  Leaf (stores RcsGrid data)

    Drop IN (onto a branch or root):
      • MIME_DATASET from the Datasets table → leaf; RcsGrid copied from
        DatasetTable._pending_drag_data set during startDrag
      • dataset file URLs from the file explorer (.grim/.csv/.txt/.out/.pio/.cmplx_di)
        → .grim creates leaves with loaded data;
        also emits files_to_load so the main window adds them to the table

    Drag OUT (branch → Datasets table):
      • Stores list[(name, RcsGrid|None)] in _pending_branch_data for the
        table to retrieve.  Tree is never modified.

    Internal drag:
      • Branches/roots: reparented manually (MIME_BRANCH used as marker).
      • Leaves: Qt InternalMove.
    """

    files_to_load = Signal(list)  # list of supported dataset file paths

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("assemblyTree")
        self.setColumnCount(2)
        self.setHeaderLabels(["Assembly", "Mode"])
        self.setColumnWidth(0, 220)
        self.setColumnWidth(1, 56)
        # InternalMove prevents Qt's C++ QAbstractItemView::dropEvent from
        # calling model()->dropMimeData() for external drops, which would
        # interfere with items we add manually in our Python override.
        # Our dragEnterEvent/dragMoveEvent/dropEvent overrides still accept
        # external MIME_DATASET and URL drops; internal leaf moves fall
        # through to super().dropEvent() which InternalMove handles correctly.
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.invisibleRootItem().setFlags(
            self.invisibleRootItem().flags() | Qt.ItemIsDropEnabled
        )
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self.itemExpanded.connect(self._on_item_expanded)
        self.itemCollapsed.connect(self._on_item_collapsed)
        self._branch_drag_item: QTreeWidgetItem | None = None
        self._pending_branch_data: list | None = None

    # ── branch indicators (plus/minus box) ───────────────────────────────────

    def drawBranches(self, painter: QPainter, rect, index) -> None:
        super().drawBranches(painter, rect, index)
        item = self.itemFromIndex(index)
        if item is None or item.childCount() == 0:
            return
        sz     = 9
        indent = self.indentation()
        x      = rect.right() - indent + (indent - sz) // 2
        y      = rect.top() + (rect.height() - sz) // 2
        mid_x  = x + sz // 2
        mid_y  = y + sz // 2
        color  = self.palette().color(QPalette.ColorRole.Text)
        bg     = self.palette().color(QPalette.ColorRole.Base)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(QPen(color, 1))
        painter.setBrush(bg)
        painter.drawRect(x, y, sz - 1, sz - 1)
        painter.drawLine(x + 2, mid_y, x + sz - 3, mid_y)
        if not item.isExpanded():
            painter.drawLine(mid_x, y + 2, mid_x, y + sz - 3)
        painter.restore()

    # ── outbound drag ────────────────────────────────────────────────────────

    def startDrag(self, supported_actions) -> None:
        item = self.currentItem()
        if item is None:
            return
        node_type = item.data(0, _ROLE_TYPE)
        if node_type in (_TYPE_ROOT, _TYPE_BRANCH):
            leaf_data = self._collect_leaf_data(item)
            self._pending_branch_data = leaf_data
            self._branch_drag_item    = item
            mime = QMimeData()
            mime.setData(MIME_BRANCH, QByteArray(item.text(0).encode("utf-8")))
            drag = QDrag(self)
            drag.setMimeData(mime)
            drag.exec(Qt.CopyAction | Qt.MoveAction)
            self._pending_branch_data = None
            self._branch_drag_item    = None
        else:
            super().startDrag(supported_actions)

    def _collect_leaf_data(self, item: QTreeWidgetItem) -> list[tuple[str, object]]:
        result: list[tuple[str, object]] = []
        for i in range(item.childCount()):
            child = item.child(i)
            if child.data(0, _ROLE_TYPE) == _TYPE_LEAF:
                name = child.data(0, _ROLE_NAME) or child.text(0)
                grid = child.data(0, _ROLE_GRID)
                result.append((name, grid))
            else:
                result.extend(self._collect_leaf_data(child))
        return result

    # ── inbound drag ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event) -> None:
        mime = event.mimeData()
        if mime.hasFormat(MIME_DATASET) or mime.hasFormat(MIME_BRANCH) or mime.hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        mime = event.mimeData()
        if mime.hasFormat(MIME_DATASET) or mime.hasFormat(MIME_BRANCH) or mime.hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        mime    = event.mimeData()
        vp_pos  = self.viewport().mapFrom(self, event.position().toPoint())
        target  = self.itemAt(vp_pos)

        # ── internal branch reparent ─────────────────────────────────────────
        if mime.hasFormat(MIME_BRANCH) and event.source() is self:
            item = self._branch_drag_item
            if item is None or item is target or _is_ancestor(target, item):
                event.ignore()
                return
            (item.parent() or self.invisibleRootItem()).removeChild(item)
            if target is not None and target.data(0, _ROLE_TYPE) in (_TYPE_ROOT, _TYPE_BRANCH):
                target.addChild(item)
                target.setExpanded(True)
            elif target is not None and target.data(0, _ROLE_TYPE) == _TYPE_LEAF:
                parent = target.parent() or self.invisibleRootItem()
                parent.addChild(item)
                parent.setExpanded(True)
            else:
                self.invisibleRootItem().addChild(item)
            event.acceptProposedAction()
            return

        # ── dataset dragged from the Datasets table ──────────────────────────
        if mime.hasFormat(MIME_DATASET) and event.source() is not self:
            src = event.source()
            entries: list[tuple[str, object]] = []
            if hasattr(src, "_pending_drag_data") and src._pending_drag_data:
                entries = src._pending_drag_data  # list of (name, RcsGrid|None)
            else:
                entries = [(bytes(mime.data(MIME_DATASET)).decode("utf-8"), None)]
            for name, grid in entries:
                _attach(self, self._make_leaf(name, grid), target)
            event.acceptProposedAction()
            return

        # ── dataset files dropped from the file explorer ─────────────────────
        if mime.hasUrls() and event.source() is not self:
            from grim_dataset import RcsGrid
            supported_paths = [
                u.toLocalFile() for u in mime.urls()
                if u.isLocalFile() and u.toLocalFile().lower().endswith(
                    (".grim", ".csv", ".txt", ".out", ".pio", ".cmplx_di")
                )
            ]
            grim_paths = [path for path in supported_paths if path.lower().endswith(".grim")]
            if grim_paths:
                for path in grim_paths:
                    name = os.path.splitext(os.path.basename(path))[0]
                    try:
                        grid = RcsGrid.load(path)
                    except Exception:
                        grid = None
                    leaf = self._make_leaf(name, grid)
                    _attach(self, leaf, target)
            if supported_paths:
                self.files_to_load.emit(supported_paths)
                event.acceptProposedAction()
                return

        super().dropEvent(event)

    # ── node factories ───────────────────────────────────────────────────────

    def _make_leaf(self, dataset_name: str, grid=None) -> QTreeWidgetItem:
        item = QTreeWidgetItem([dataset_name])
        item.setData(0, _ROLE_TYPE, _TYPE_LEAF)
        item.setData(0, _ROLE_NAME, dataset_name)
        item.setData(0, _ROLE_GRID, grid)
        _apply_flags(item, _TYPE_LEAF)
        _apply_leaf_style(item, grid is not None)
        item.setIcon(0, _node_icon(_TYPE_LEAF, has_data=(grid is not None)))
        _set_node_mode(item, _DEFAULT_MODE)
        return item

    def _make_node(
        self,
        name: str,
        node_type: str,
        parent: QTreeWidgetItem | None = None,
        edit: bool = True,
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([name])
        item.setData(0, _ROLE_TYPE, node_type)
        _apply_flags(item, node_type)
        item.setIcon(0, _node_icon(node_type, expanded=False))
        if node_type == _TYPE_ROOT:
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
        if node_type == _TYPE_ROOT:
            _apply_mode_badge(item, None)
        else:
            _set_node_mode(item, _DEFAULT_MODE)
        if parent is not None:
            parent.addChild(item)
            parent.setExpanded(True)
        else:
            self.invisibleRootItem().addChild(item)
        if edit:
            self.scrollToItem(item)
            self.editItem(item, 0)
        return item

    def _remove_item(self, item: QTreeWidgetItem) -> None:
        (item.parent() or self.invisibleRootItem()).removeChild(item)

    # ── expand / collapse icon updates ───────────────────────────────────────

    def _on_item_expanded(self, item: QTreeWidgetItem) -> None:
        node_type = item.data(0, _ROLE_TYPE)
        if node_type in (_TYPE_ROOT, _TYPE_BRANCH):
            item.setIcon(0, _node_icon(node_type, expanded=True))

    def _on_item_collapsed(self, item: QTreeWidgetItem) -> None:
        node_type = item.data(0, _ROLE_TYPE)
        if node_type in (_TYPE_ROOT, _TYPE_BRANCH):
            item.setIcon(0, _node_icon(node_type, expanded=False))

    # ── context menu ─────────────────────────────────────────────────────────

    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        menu = QMenu(self)
        act_root   = menu.addAction("Add Root")
        act_branch = menu.addAction("Add Branch")
        act_del    = menu.addAction("Delete")
        menu.addSeparator()
        act_expand   = menu.addAction("Expand")
        act_collapse = menu.addAction("Collapse")
        menu.addSeparator()
        act_rename = menu.addAction("Rename")

        # Per-node add-mode setters (only meaningful for non-root nodes).
        act_set_coh = None
        act_set_inc = None
        if item is not None and item.data(0, _ROLE_TYPE) != _TYPE_ROOT:
            menu.addSeparator()
            current = _node_mode(item)
            act_set_coh = menu.addAction("Set Add Mode: Coherent (+)")
            act_set_inc = menu.addAction("Set Add Mode: Incoherent (+)")
            act_set_coh.setCheckable(True)
            act_set_inc.setCheckable(True)
            act_set_coh.setChecked(current == _MODE_COH)
            act_set_inc.setChecked(current == _MODE_INCOH)

        chosen = menu.exec(self.viewport().mapToGlobal(pos))
        if chosen == act_root:
            self._make_node("New Root", _TYPE_ROOT, parent=None)
        elif chosen == act_branch:
            self._make_node("New Branch", _TYPE_BRANCH, parent=item)
        elif chosen == act_del and item is not None:
            self._remove_item(item)
        elif chosen == act_expand and item is not None:
            self.expandItem(item)
            for i in range(item.childCount()):
                self.expandItem(item.child(i))
        elif chosen == act_collapse and item is not None:
            self.collapseItem(item)
        elif chosen == act_rename and item is not None:
            self.editItem(item, 0)
        elif chosen == act_set_coh and item is not None:
            _set_node_mode(item, _MODE_COH)
        elif chosen == act_set_inc and item is not None:
            _set_node_mode(item, _MODE_INCOH)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _attach(
    tree: QTreeWidget, item: QTreeWidgetItem, target: QTreeWidgetItem | None
) -> None:
    if target is not None and target.data(0, _ROLE_TYPE) in (_TYPE_ROOT, _TYPE_BRANCH):
        target.addChild(item)
        target.setExpanded(True)
    elif target is not None and target.data(0, _ROLE_TYPE) == _TYPE_LEAF:
        # Drop on a leaf → insert into the leaf's parent container
        parent = target.parent() or tree.invisibleRootItem()
        parent.addChild(item)
        parent.setExpanded(True)
    else:
        tree.invisibleRootItem().addChild(item)


def _is_ancestor(candidate: QTreeWidgetItem | None, item: QTreeWidgetItem) -> bool:
    if candidate is None:
        return False
    p = item.parent()
    while p is not None:
        if p is candidate:
            return True
        p = p.parent()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Panel widget
# ─────────────────────────────────────────────────────────────────────────────

class AssemblyTreePanel(QWidget):
    files_to_load = Signal(list)
    # (platform_name, combined_RcsGrid, history_string)
    platform_built = Signal(str, object, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Assembly Tree"))

        row1 = QHBoxLayout()
        row1.setSpacing(4)
        self.btn_add_root   = QToolButton(text="+ Root")
        self.btn_add_branch = QToolButton(text="+ Branch")
        self.btn_delete     = QToolButton(text="Delete")
        row1.addWidget(self.btn_add_root)
        row1.addWidget(self.btn_add_branch)
        row1.addWidget(self.btn_delete)
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(4)
        self.btn_expand   = QToolButton(text="Expand")
        self.btn_collapse = QToolButton(text="Collapse")
        self.btn_save = QToolButton(text="Save .asy")
        self.btn_load = QToolButton(text="Load .asy")
        row2.addWidget(self.btn_expand)
        row2.addWidget(self.btn_collapse)
        row2.addStretch(1)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(4)
        row3.addWidget(self.btn_save)
        row3.addWidget(self.btn_load)
        row3.addStretch(1)
        layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.setSpacing(4)
        self.btn_build = QToolButton(text="Build Platform")
        self.btn_build.setToolTip(
            "Recursively combine the selected root/branch into a single dataset, "
            "honouring each node's coherent / incoherent add-mode and aligning "
            "axes across parts using the strategy chosen in the dialog."
        )
        row4.addWidget(self.btn_build)
        row4.addStretch(1)
        layout.addLayout(row4)

        self.tree = AssemblyTree()
        layout.addWidget(self.tree, 1)

        self.btn_add_root.clicked.connect(
            lambda: self.tree._make_node("New Root", _TYPE_ROOT)
        )
        self.btn_add_branch.clicked.connect(self._add_branch)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_expand.clicked.connect(self._expand_selected)
        self.btn_collapse.clicked.connect(self._collapse_selected)
        self.btn_save.clicked.connect(self._save)
        self.btn_load.clicked.connect(self._load)
        self.btn_build.clicked.connect(self._build)
        self.tree.files_to_load.connect(self.files_to_load)

    def _add_branch(self) -> None:
        parent = self.tree.currentItem()
        if parent is not None and parent.data(0, _ROLE_TYPE) == _TYPE_LEAF:
            parent = parent.parent()
        self.tree._make_node("New Branch", _TYPE_BRANCH, parent=parent)

    def _delete_selected(self) -> None:
        item = self.tree.currentItem()
        if item is not None:
            self.tree._remove_item(item)

    def _expand_selected(self) -> None:
        item = self.tree.currentItem()
        if item is not None:
            self.tree.expandItem(item)
            for i in range(item.childCount()):
                self.tree.expandItem(item.child(i))

    def _collapse_selected(self) -> None:
        item = self.tree.currentItem()
        if item is not None:
            self.tree.collapseItem(item)

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Assembly Tree", "assembly.asy", "Assembly Files (*.asy)"
        )
        if not path:
            return
        if not path.lower().endswith(".asy"):
            path += ".asy"
        root  = self.tree.invisibleRootItem()
        nodes = [_item_to_dict(root.child(i)) for i in range(root.childCount())]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "tree": nodes}, f, indent=2)

    def _build(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            roots = [
                self.tree.invisibleRootItem().child(i)
                for i in range(self.tree.invisibleRootItem().childCount())
            ]
            if len(roots) == 1:
                item = roots[0]
            else:
                self._notify("Select a root or branch to build first.")
                return
        if item.data(0, _ROLE_TYPE) not in (_TYPE_ROOT, _TYPE_BRANCH):
            self._notify("Select a root or branch (not a leaf) to build.")
            return

        dlg = BuildDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        axis_mode = dlg.axis_mode()

        try:
            grid, history = build_assembly_grid(item, axis_mode=axis_mode)
        except Exception as exc:
            self._notify(f"Build failed: {exc}")
            return
        if grid is None:
            self._notify("Build produced no data (subtree has no loaded leaves).")
            return
        self.platform_built.emit(item.text(0), grid, history)

    def _notify(self, text: str) -> None:
        # Surface a transient error through the main window's status bar if
        # we can find it; otherwise just print so the user has *something*.
        try:
            self.window().status.showMessage(text)
        except Exception:
            print(text)

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Assembly Tree", "", "Assembly Files (*.asy)"
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tree.clear()
        for node_dict in data.get("tree", []):
            self.tree.invisibleRootItem().addChild(_dict_to_item(node_dict))
        self.tree.expandAll()


# ─────────────────────────────────────────────────────────────────────────────
# Build dialog + recursive accumulator
# ─────────────────────────────────────────────────────────────────────────────


class BuildDialog(QDialog):
    """Choose how to handle axis mismatch when summing parts in a subtree."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build Platform")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Axis alignment across parts (each leaf's grid may differ "
            "in azimuth / elevation / frequency / polarization):"
        ))
        self._btn_group = QButtonGroup(self)
        self._radio_intersect = QRadioButton(
            "Intersect — keep only axis values present in every part (no interpolation, lossless)."
        )
        self._radio_interp = QRadioButton(
            "Interpolate — bilinear-interpolate each part onto a common grid (no extrapolation)."
        )
        self._radio_strict = QRadioButton(
            "Strict — require every part to share exactly the same axes (error if any differs)."
        )
        self._radio_intersect.setChecked(True)
        self._btn_group.addButton(self._radio_intersect, 0)
        self._btn_group.addButton(self._radio_interp, 1)
        self._btn_group.addButton(self._radio_strict, 2)
        layout.addWidget(self._radio_intersect)
        layout.addWidget(self._radio_interp)
        layout.addWidget(self._radio_strict)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def axis_mode(self) -> str:
        if self._radio_interp.isChecked():
            return "interp"
        if self._radio_strict.isChecked():
            return "strict"
        return "intersect"


def _intersect_numeric_axes(arrays, tol: float = 1e-6) -> np.ndarray:
    """Intersect a list of numeric 1-D arrays element-wise, with absolute
    tolerance for floating-point matches. Preserves the first array's
    ordering.
    """
    if not arrays:
        return np.asarray([], dtype=float)
    out = np.asarray(arrays[0], dtype=float)
    for arr in arrays[1:]:
        arr = np.asarray(arr, dtype=float)
        keep = []
        for v in out:
            if np.any(np.isclose(arr, v, atol=tol, rtol=0.0)):
                keep.append(v)
        out = np.asarray(keep, dtype=float)
        if out.size == 0:
            break
    return out


def _intersect_categorical_axes(arrays) -> np.ndarray:
    """Intersection of categorical (string) axes preserving the first axis order."""
    if not arrays:
        return np.asarray([], dtype=object)
    first = list(arrays[0])
    others = [set(arr.tolist() if hasattr(arr, "tolist") else list(arr)) for arr in arrays[1:]]
    keep = [v for v in first if all(v in s for s in others)]
    return np.asarray(keep)


def _interp_target_axes(grids) -> tuple:
    """For interp mode: target axes = the first grid's values restricted to the
    common axis range across all grids, plus the categorical intersection of
    polarizations. This keeps every output sample within every part's support
    so RcsGrid.align_to('interp') never has to extrapolate.
    """
    ref = grids[0]

    def _clipped(axis_name: str) -> np.ndarray:
        ref_axis = np.asarray(getattr(ref, axis_name), dtype=float)
        if ref_axis.size == 0:
            return ref_axis
        lo = max(np.asarray(getattr(g, axis_name), dtype=float).min() for g in grids)
        hi = min(np.asarray(getattr(g, axis_name), dtype=float).max() for g in grids)
        mask = (ref_axis >= lo - 1e-9) & (ref_axis <= hi + 1e-9)
        return ref_axis[mask]

    az = _clipped("azimuths")
    el = _clipped("elevations")
    f = _clipped("frequencies")
    pol = _intersect_categorical_axes([g.polarizations for g in grids])
    return az, el, f, pol


def _axes_only_grid(az, el, f, pol):
    """Construct a stub RcsGrid carrying only axis arrays — used as the target
    passed to RcsGrid.align_to() so we can align every part to the same set
    of axes without inventing a synthetic data array each time."""
    from grim_dataset import RcsGrid

    shape = (len(az), len(el), len(f), len(pol))
    zero = np.zeros(shape, dtype=np.float32)
    return RcsGrid(
        np.asarray(az), np.asarray(el), np.asarray(f), np.asarray(pol),
        rcs=None, rcs_power=zero, rcs_phase=zero,
        rcs_domain="power_phase",
    )


def _align_grids_for_assembly(grids, axis_mode: str) -> list:
    """Align every grid in `grids` to a shared set of axes per `axis_mode`.

    strict: validate every grid has identical axes; no resampling.
    intersect: take the pairwise intersection of every axis; no interpolation.
    interp: build a common grid clipped to the overlapping range; bilinear
            (linear-per-axis) interpolation onto it.
    """
    if not grids:
        return []
    if axis_mode == "strict":
        ref = grids[0]
        for g in grids[1:]:
            ref._assert_compatible(g)
        return list(grids)

    if axis_mode == "intersect":
        az = _intersect_numeric_axes([g.azimuths for g in grids])
        el = _intersect_numeric_axes([g.elevations for g in grids])
        f = _intersect_numeric_axes([g.frequencies for g in grids])
        pol = _intersect_categorical_axes([g.polarizations for g in grids])
        if az.size == 0 or el.size == 0 or f.size == 0 or pol.size == 0:
            raise ValueError(
                "intersect: parts have no common azimuth/elevation/frequency/polarization values"
            )
        target = _axes_only_grid(az, el, f, pol)
        return [g.align_to(target, mode="intersect") for g in grids]

    if axis_mode == "interp":
        az, el, f, pol = _interp_target_axes(grids)
        if az.size == 0 or el.size == 0 or f.size == 0 or pol.size == 0:
            raise ValueError(
                "interp: parts have no overlapping axis range across all parts"
            )
        target = _axes_only_grid(az, el, f, pol)
        return [g.align_to(target, mode="interp") for g in grids]

    raise ValueError(f"unknown axis_mode {axis_mode!r}")


def _combine_children(coh_grids, incoh_grids, ref):
    """Coherent-field + incoherent-power combination of pre-aligned children.

    coh_grids contribute by complex sum   →  C_coh = Σ rcs
    incoh_grids contribute by power sum  →  P_incoh = Σ rcs_power
    output power   = |C_coh|² + P_incoh
    output phase   = arg(C_coh)   (NaN if there are no coherent contributors)
    """
    from grim_dataset import RcsGrid

    C_coh = None
    if coh_grids:
        C_coh = np.array(coh_grids[0].rcs, copy=True)
        for g in coh_grids[1:]:
            C_coh = C_coh + g.rcs

    P_incoh = None
    if incoh_grids:
        P_incoh = np.array(incoh_grids[0].rcs_power, copy=True)
        for g in incoh_grids[1:]:
            P_incoh = P_incoh + g.rcs_power

    if C_coh is not None and P_incoh is not None:
        total_power = (np.abs(C_coh) ** 2 + P_incoh).astype(np.float32)
        total_phase = np.angle(C_coh).astype(np.float32)
    elif C_coh is not None:
        total_power = (np.abs(C_coh) ** 2).astype(np.float32)
        total_phase = np.angle(C_coh).astype(np.float32)
    else:
        total_power = P_incoh.astype(np.float32)
        total_phase = np.full(total_power.shape, np.nan, dtype=np.float32)

    return RcsGrid(
        ref.azimuths, ref.elevations, ref.frequencies, ref.polarizations,
        rcs=None,
        rcs_power=total_power,
        rcs_phase=total_phase,
        rcs_domain="power_phase",
        units=dict(ref.units or {}),
    )


def build_assembly_grid(node: QTreeWidgetItem, *, axis_mode: str = "intersect"):
    """Recursively materialise an assembly subtree into a single RcsGrid.

    Leaves return their stored grid (or None if empty). Branches/roots gather
    every non-None child grid, separate them by add-mode, align all children
    to a common axis grid per `axis_mode`, and combine coherent+incoherent
    contributions as in `_combine_children`.

    Returns (grid, history_string). Both are None / "" if the subtree has
    no loaded data.
    """
    node_type = node.data(0, _ROLE_TYPE)
    if node_type == _TYPE_LEAF:
        grid = node.data(0, _ROLE_GRID)
        if grid is None:
            return None, ""
        return grid, node.text(0)

    coh: list = []
    incoh: list = []
    for i in range(node.childCount()):
        child = node.child(i)
        child_grid, child_history = build_assembly_grid(child, axis_mode=axis_mode)
        if child_grid is None:
            continue
        child_mode = _node_mode(child)
        bucket = coh if child_mode == _MODE_COH else incoh
        bucket.append((child_history, child_grid))

    if not coh and not incoh:
        return None, ""

    all_pairs = coh + incoh
    grids_in_order = [g for _, g in all_pairs]

    # Refuse to mix dBsm and dBke flavours — the linear values represent
    # physically different quantities (m² vs m) and adding them would be
    # silent nonsense.
    unit_set = {
        str((g.units or {}).get("rcs_log_unit", "dBsm")).strip().lower()
        for g in grids_in_order
    }
    if len(unit_set) > 1:
        raise ValueError(
            f"refusing to combine parts with mixed rcs_log_unit values {unit_set}"
        )

    aligned = _align_grids_for_assembly(grids_in_order, axis_mode=axis_mode)
    n_coh = len(coh)
    aligned_coh = aligned[:n_coh]
    aligned_incoh = aligned[n_coh:]
    ref = aligned[0]
    result = _combine_children(aligned_coh, aligned_incoh, ref)

    parts = []
    if coh:
        parts.append("coh[" + " + ".join(h for h, _ in coh) + "]")
    if incoh:
        parts.append("incoh[" + " + ".join(h for h, _ in incoh) + "]")
    history = f"Σ {node.text(0)} ({axis_mode}): " + " ⊕ ".join(parts)
    return result, history

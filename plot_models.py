from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QLabel,
    QToolButton,
    QWidget,
)



@dataclass
class PlotContext:
    btn_export_plot: QToolButton
    btn_assembly_tree: QToolButton
    btn_settings: QToolButton
    settings_frame: QFrame
    assembly_tree_panel: QWidget
    spin_plot_xmin: QDoubleSpinBox
    spin_plot_xmax: QDoubleSpinBox
    spin_plot_xstep: QDoubleSpinBox
    spin_plot_ymin: QDoubleSpinBox
    spin_plot_ymax: QDoubleSpinBox
    spin_plot_ystep: QDoubleSpinBox
    spin_plot_zmin: QDoubleSpinBox
    spin_plot_zmax: QDoubleSpinBox
    spin_plot_zstep: QDoubleSpinBox
    combo_plot_scale: QComboBox
    combo_polar_zero: QComboBox
    combo_colormap: QComboBox
    chk_colorbar: QCheckBox
    chk_colorbar_shared: QCheckBox
    chk_plot_grid_visible: QCheckBox
    chk_colormap_invert: QCheckBox
    combo_isar_window: QComboBox
    combo_isar_units: QComboBox
    combo_isar_algorithm: QComboBox
    chk_isar_az_interp: QCheckBox
    spin_isar_az_min: QDoubleSpinBox
    spin_isar_az_max: QDoubleSpinBox
    spin_isar_az_step: QDoubleSpinBox
    chk_isar_square: QCheckBox
    btn_isar_apply: QToolButton
    btn_plot_bg: QToolButton
    btn_plot_grid: QToolButton
    btn_plot_text: QToolButton
    chk_plot_legend: QCheckBox
    hover_readout: QLabel
    plot_figure: Figure
    plot_canvas: FigureCanvas
    plot_ax: Any
    plot_colorbars: list[Any]
    plot_axes: list[Any] | None
    plot_bg_color: str | None
    plot_grid_color: str | None
    plot_text_color: str | None
    last_plot_mode: str | None

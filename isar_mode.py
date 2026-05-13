from __future__ import annotations

import numpy as np


def _unit_to_hz_scale(unit: str) -> float:
    unit = unit.strip().lower()
    if unit == "hz":
        return 1.0
    if unit == "khz":
        return 1e3
    if unit == "mhz":
        return 1e6
    if unit == "ghz":
        return 1e9
    return 1e9


_LENGTH_UNIT_FACTORS = {
    "m": 1.0,
    "in": 1.0 / 0.0254,
    "ft": 1.0 / 0.3048,
}


def _length_unit(name: str | None) -> tuple[str, float]:
    key = (name or "m").strip().lower()
    return (key, _LENGTH_UNIT_FACTORS[key]) if key in _LENGTH_UNIT_FACTORS else ("m", 1.0)


def _resample_azimuth_to_target(
    source_deg: np.ndarray,
    samples: np.ndarray,
    target_deg: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample complex `samples` along `axis` from azimuth `source_deg` onto `target_deg`.

    Treats azimuth as a periodic (360°) axis when the source covers ≥359°,
    which is the case the reference ISAR program is built around — a full sweep
    sampled non-uniformly. For partial apertures, falls back to linear
    interpolation with zero-fill outside the source support so the missing
    arcs don't get aliased by a long way-round wrap.

    Real and imaginary parts are interpolated independently — interpolating
    magnitude or argument introduces phase-wrap artefacts.
    """
    source = np.asarray(source_deg, dtype=float)
    target = np.asarray(target_deg, dtype=float)
    span = float(source.max() - source.min())
    use_period = span >= 359.0

    # Sort source ascending — np.interp requires this. For periodic mode,
    # wrap into the canonical [src_min, src_min+360) before interpolation.
    order = np.argsort(source)
    source_sorted = source[order]
    samples_sorted = np.take(samples, order, axis=axis)

    samples_moved = np.moveaxis(samples_sorted, axis, -1)
    flat = samples_moved.reshape(-1, samples_moved.shape[-1])

    if use_period:
        # np.interp accepts `period` for periodic interpolation and reduces
        # any target value into the canonical interval automatically.
        out = np.empty((flat.shape[0], target.size), dtype=np.complex128)
        for i in range(flat.shape[0]):
            out[i, :] = (
                np.interp(target, source_sorted, flat[i, :].real, period=360.0)
                + 1j * np.interp(target, source_sorted, flat[i, :].imag, period=360.0)
            )
    else:
        # Linear interp with zero-fill outside [source_min, source_max].
        out = np.zeros((flat.shape[0], target.size), dtype=np.complex128)
        in_range = (target >= source_sorted[0]) & (target <= source_sorted[-1])
        if np.any(in_range):
            t_sub = target[in_range]
            for i in range(flat.shape[0]):
                out[i, in_range] = (
                    np.interp(t_sub, source_sorted, flat[i, :].real)
                    + 1j * np.interp(t_sub, source_sorted, flat[i, :].imag)
                )

    out = out.reshape(samples_moved.shape[:-1] + (target.size,))
    out = np.moveaxis(out, -1, axis)
    return target, out


def _resample_complex_uniform(
    values: np.ndarray,
    samples: np.ndarray,
    axis: int,
    rel_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Linearly resample complex `samples` onto a uniform grid along `axis`.

    Returns (uniform_values, resampled_samples, non_uniformity).
    `non_uniformity` is the relative spread of the original spacings —
    `(max_diff - min_diff) / median_diff`. When this is below `rel_tol`,
    the inputs are returned unchanged (no work done) and the value is
    reported so callers can warn the user.

    Real and imaginary components are interpolated separately — that's
    what np.interp expects, and for ISAR it's what you want anyway since
    interpolating |z| or arg(z) introduces phase-wrap artefacts.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return values, samples, 0.0
    diffs = np.diff(values)
    median_diff = float(np.median(diffs))
    if median_diff <= 0.0:
        return values, samples, 0.0
    non_uniformity = float(np.max(diffs) - np.min(diffs)) / median_diff
    if non_uniformity < rel_tol:
        return values, samples, non_uniformity

    target = np.linspace(values[0], values[-1], values.size)
    samples = np.moveaxis(samples, axis, -1)
    flat_in = samples.reshape(-1, samples.shape[-1])
    flat_out = np.empty_like(flat_in)
    for i in range(flat_in.shape[0]):
        flat_out[i, :] = (
            np.interp(target, values, flat_in[i, :].real)
            + 1j * np.interp(target, values, flat_in[i, :].imag)
        )
    out = flat_out.reshape(samples.shape)
    out = np.moveaxis(out, -1, axis)
    return target, out, non_uniformity


def _split_into_bands(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    bands: list[list[int]] = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            bands.append(current)
            current = [idx]
    bands.append(current)
    return bands




def _compute_band_backprojection(
    self,
    rcs_polar: np.ndarray,
    theta: np.ndarray,
    freq_hz: np.ndarray,
    n_x: int,
    n_y: int,
    half_x: float,
    half_y: float,
    unit_scale: float,
):
    """Time-domain back-projection ISAR. Geometrically correct at any aperture
    (including a full 360° sweep), unlike the decoupled FFT and PFA which both
    rely on a small-angle / paraxial approximation.

    Math: for a target rotating in the (x, y) plane (or equivalently, the
    radar looking at angles θ_n around the target), the round-trip path-length
    difference for a scatterer at (x, y) is 2·(x sin θ + y cos θ), and the
    measured backscatter follows S(θ, k) ∝ exp(+j · 2 k r). The matched-filter
    image at hypothesised pixel (x, y) is therefore

        I(x, y) = ∑_{n, m} S(θ_n, f_m) · exp(-j · 2 k_m · (x sin θ_n + y cos θ_n))

    (negative sign — conjugate of the data kernel, matching the +j convention
    used by the existing decoupled-FFT and PFA paths which both rely on
    np.fft.ifft). We factor the exponential as
    exp(-j · 2 k_m · x sin θ_n) · exp(-j · 2 k_m · y cos θ_n) and evaluate the
    inner sum-over-frequency as a single (n_x, n_f) @ (n_f, n_y) matrix
    multiply per angle, which is ~10×–100× faster than the naive triple loop
    while keeping memory bounded.

    Returns (image, x_range, y_range) with `image` indexed (n_x, n_y) so the
    caller's `imshow(image.T, extent=…, origin='lower')` puts cross-range on
    the horizontal axis exactly like the FFT-based paths.
    """
    n_az = theta.size
    n_freq = freq_hz.size
    c0 = 299_792_458.0
    k = 2.0 * np.pi * freq_hz / c0  # (n_freq,) wavenumbers in rad/m

    win_az = self._isar_window(n_az)
    win_freq = self._isar_window(n_freq)
    rcs_windowed = (rcs_polar * np.outer(win_az, win_freq)).astype(np.complex128)

    # Image grid in metres. Defaults from the caller pick a sensible FoV.
    x_grid = np.linspace(-half_x, half_x, n_x)
    y_grid = np.linspace(-half_y, half_y, n_y)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    image = np.zeros((n_x, n_y), dtype=np.complex128)

    # Per-angle inner loop. The frequency sum for angle θ_n is
    #
    #   I_n(x, y) = Σ_m S(θ_n, f_m) · u_n(x, m) · v_n(y, m)
    #             = (u_n diag(S_n)) @ v_n.T
    #
    # where u_n(x, m) = exp(j · 2 k_m x sin θ_n) and
    #       v_n(y, m) = exp(j · 2 k_m y cos θ_n).
    neg_two_k = -2.0 * k  # (n_freq,)
    for n in range(n_az):
        u = np.exp(1j * np.outer(x_grid * sin_t[n], neg_two_k))  # (n_x, n_freq)
        v = np.exp(1j * np.outer(y_grid * cos_t[n], neg_two_k))  # (n_y, n_freq)
        scaled = u * rcs_windowed[n][None, :]                     # (n_x, n_freq)
        image += scaled @ v.T                                     # (n_x, n_y)

    # Normalize so a unit-amplitude scatterer at the origin produces a unit-
    # magnitude peak (→ 0 dB after the dBsm conversion).  At origin every
    # phasor is 1, so the sum is Σ_n win_az[n] · Σ_m win_freq[m].
    peak_norm = float(np.sum(win_az) * np.sum(win_freq))
    if peak_norm > 0.0:
        image = image / peak_norm

    x_range = x_grid * unit_scale
    y_range = y_grid * unit_scale

    return image, x_range, y_range


def _decoupled_scene_half_extents(
    theta: np.ndarray, freq_hz: np.ndarray, df: float
) -> tuple[float, float]:
    """Return (half_x, half_y) in metres for the natural Cartesian scene
    extent given the polar sample spacings. Used to seed the back-projection
    image grid so the image FoV scales with the input aperture and bandwidth."""
    c0 = 299_792_458.0
    if theta.size >= 2:
        dtheta = float(np.mean(np.diff(theta)))
    else:
        dtheta = 0.0
    f_c = float(np.mean(freq_hz)) if freq_hz.size else 0.0
    if dtheta > 0.0 and f_c > 0.0:
        half_x = c0 / (4.0 * f_c * dtheta)
    else:
        half_x = 1.0
    n_freq = freq_hz.size
    if df > 0.0 and n_freq > 0:
        half_y = (c0 / 2.0) * (n_freq / 2.0) / (n_freq * df)
    else:
        half_y = 1.0
    return float(half_x), float(half_y)


def _compute_band(
    self,
    band_az_indices: list[int],
    freq_indices_sorted: list[int],
    elev_idx: int,
    pol_idx: int,
    freq_hz: np.ndarray,
    df: float,
    unit_scale: float,
    *,
    az_target_deg: np.ndarray | None = None,
):
    band_az_values = self.active_dataset.azimuths[band_az_indices]
    order = np.argsort(band_az_values)
    sorted_band_indices = [band_az_indices[i] for i in order]
    az_values = band_az_values[order].astype(float)

    rcs_slice = self.active_dataset.rcs[
        np.ix_(sorted_band_indices, [elev_idx], freq_indices_sorted, [pol_idx])
    ][:, 0, :, 0]
    phase_slice = self.active_dataset.rcs_phase[
        np.ix_(sorted_band_indices, [elev_idx], freq_indices_sorted, [pol_idx])
    ][:, 0, :, 0]
    if not np.any(np.isfinite(phase_slice)):
        return "ISAR imaging requires phase-aware samples; selected data has no finite rcs_phase."
    rcs_slice = np.where(np.isfinite(rcs_slice), rcs_slice, 0.0)

    if az_target_deg is not None:
        # User explicitly asked for a uniform azimuth grid. Use periodic
        # interpolation (or linear w/ zero-fill for partial apertures).
        az_uniform, rcs_slice = _resample_azimuth_to_target(
            az_values, rcs_slice, np.asarray(az_target_deg, dtype=float), axis=0
        )
        if az_uniform.size < 2:
            return "ISAR azimuth target grid must have ≥2 samples."
        az_nonuniformity = 0.0
    else:
        theta_native = np.deg2rad(az_values)
        if not np.all(np.isfinite(theta_native)) or np.any(np.diff(theta_native) <= 0):
            return "Azimuth samples must be strictly increasing within a band."
        # Auto-regularise non-uniform input. Back-projection doesn't strictly
        # need it, but the uniform spacing makes the scene-extent formulas
        # below well-defined.
        az_uniform, rcs_slice, az_nonuniformity = _resample_complex_uniform(
            az_values, rcs_slice, axis=0
        )

    freq_uniform, rcs_slice, fr_nonuniformity = _resample_complex_uniform(
        freq_hz, rcs_slice, axis=1
    )
    theta = np.deg2rad(az_uniform)
    if freq_uniform.size >= 2:
        df_eff = float(np.mean(np.diff(freq_uniform)))
    else:
        df_eff = df
    az_values = az_uniform

    # Back-projection on a square scene sized to the larger of the two
    # decoupled-FFT half-extents — full-sweep data is symmetric in cross-
    # range and range, and this keeps everything visible. Image pixels are
    # capped so the O(N_θ·N_x·N_y·N_f) sum stays interactive.
    half_x_m, half_y_m = _decoupled_scene_half_extents(theta, freq_uniform, df_eff)
    n_pix_default = int(np.clip(max(theta.size, freq_uniform.size), 128, 512))
    complex_image, x_range, y_range = _compute_band_backprojection(
        self,
        rcs_slice,
        theta,
        freq_uniform,
        n_pix_default,
        n_pix_default,
        half_x_m,
        half_y_m,
        unit_scale,
    )

    # Sanity-check the computed scene extent.
    if (
        not np.all(np.isfinite(x_range))
        or not np.all(np.isfinite(y_range))
        or float(np.max(np.abs(x_range))) > 1.0e4
        or float(np.max(np.abs(y_range))) > 1.0e4
    ):
        x_max_abs = float(np.max(np.abs(x_range))) if np.all(np.isfinite(x_range)) else float("inf")
        y_max_abs = float(np.max(np.abs(y_range))) if np.all(np.isfinite(y_range)) else float("inf")
        th_max_deg = float(np.rad2deg(np.max(np.abs(theta))))
        dth_deg = float(np.rad2deg(np.mean(np.diff(theta)))) if theta.size > 1 else 0.0
        f_min_ghz = float(np.min(freq_uniform)) / 1e9
        f_max_ghz = float(np.max(freq_uniform)) / 1e9
        return (
            f"ISAR produced a degenerate scene extent: "
            f"x≈±{x_max_abs:.1e}m, y≈±{y_max_abs:.1e}m. "
            f"Inputs: θ_max={th_max_deg:.4f}°, dθ={dth_deg:.6f}°, "
            f"f∈[{f_min_ghz:.3f}, {f_max_ghz:.3f}] GHz. "
            f"Likely a too-narrow azimuth selection or unit mismatch."
        )

    return {
        "az_values": az_values,
        "magnitude": np.abs(complex_image),
        "x_range": x_range,
        "y_range": y_range,
        "az_nonuniformity": az_nonuniformity,
        "freq_nonuniformity": fr_nonuniformity,
    }


def render(self) -> None:
    self.last_plot_mode = "isar_image"
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return

    az_indices = sorted(self._selected_indices(self.list_az))
    if not az_indices:
        self.status.showMessage("Select one or more azimuths to plot.")
        return
    freq_indices = sorted(self._selected_indices(self.list_freq))
    if not freq_indices:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    if len(freq_indices) < 2:
        self.status.showMessage("Select at least 2 frequency samples for ISAR imaging.")
        return

    pol_idx = self._single_selection_index(self.list_pol, "polarization")
    if pol_idx is None:
        return
    elev_idx = self._single_selection_index(self.list_elev, "elevation")
    if elev_idx is None:
        return

    az_interp_widget = getattr(self, "chk_isar_az_interp", None)
    az_interp_on = bool(az_interp_widget.isChecked()) if az_interp_widget is not None else False
    az_target_deg: np.ndarray | None = None
    if az_interp_on:
        az_min = float(self.spin_isar_az_min.value())
        az_max = float(self.spin_isar_az_max.value())
        az_step = float(self.spin_isar_az_step.value())
        if not np.isfinite(az_min) or not np.isfinite(az_max) or not np.isfinite(az_step):
            self.status.showMessage("ISAR azimuth interp: limits/step must be finite.")
            return
        if az_step <= 0.0:
            self.status.showMessage("ISAR azimuth interp: step must be positive.")
            return
        if az_max <= az_min:
            self.status.showMessage("ISAR azimuth interp: max must exceed min.")
            return
        # arange-with-half-step so the inclusive upper bound lands on the grid
        # when (max-min) is an integer multiple of step (the common case).
        az_target_deg = np.arange(az_min, az_max + az_step * 0.5, az_step, dtype=float)
        if az_target_deg.size < 2:
            self.status.showMessage("ISAR azimuth interp grid needs ≥2 samples.")
            return

    if az_interp_on:
        # Explicit resample collapses the multi-band view into a single image —
        # the periodic interpolator stitches selected sub-apertures together
        # exactly the way the reference program does for full-sweep mode.
        bands: list[list[int]] = [az_indices] if len(az_indices) >= 2 else []
    else:
        bands = _split_into_bands(az_indices)
        bands = [b for b in bands if len(b) >= 2]
    if not bands:
        self.status.showMessage(
            "Each azimuth band needs at least 2 contiguous samples for ISAR imaging."
        )
        return

    freq_values_full = self.active_dataset.frequencies[freq_indices]
    freq_order = np.argsort(freq_values_full)
    freq_indices_sorted = [freq_indices[i] for i in freq_order]
    freq_values = freq_values_full[freq_order].astype(float)
    if np.any(np.diff(freq_values) <= 0) or not np.all(np.isfinite(freq_values)):
        self.status.showMessage(
            "Frequency samples must be finite and strictly increasing for ISAR imaging."
        )
        return

    freq_unit = str(self.active_dataset.units.get("frequency", "ghz"))
    freq_hz = freq_values * _unit_to_hz_scale(freq_unit)
    df = float(np.mean(np.diff(freq_hz)))
    if df <= 0.0:
        self.status.showMessage("ISAR imaging requires increasing frequency samples.")
        return

    units_combo = getattr(self, "combo_isar_units", None)
    unit_name, unit_scale = _length_unit(units_combo.currentText() if units_combo else "m")

    band_results = []
    for band_az_indices in bands:
        result = _compute_band(
            self,
            band_az_indices,
            freq_indices_sorted,
            elev_idx,
            pol_idx,
            freq_hz,
            df,
            unit_scale,
            az_target_deg=az_target_deg,
        )
        if isinstance(result, str):
            self.status.showMessage(result)
            return
        band_results.append(result)

    # Convert linear magnitudes to display values (absolute dBsm — a unit-
    # amplitude scatterer reads 0 dB).
    for br in band_results:
        if self._plot_scale_is_linear():
            br["isar_display"] = br["magnitude"]
        else:
            br["isar_display"] = self.active_dataset.rcs_to_dbsm(br["magnitude"])

    n_bands = len(band_results)

    self._remove_colorbar()
    self.plot_figure.clear()
    if n_bands == 1:
        self.plot_ax = self.plot_figure.add_subplot(111)
        self.plot_axes = None
        active_axes = [self.plot_ax]
    else:
        ax_array = self.plot_figure.subplots(1, n_bands, sharey=True)
        if not isinstance(ax_array, np.ndarray):
            ax_array = np.array([ax_array])
        active_axes = list(ax_array.ravel())
        self.plot_axes = active_axes
        self.plot_ax = active_axes[0]
    self._style_plot_axes()

    cmap = self._effective_colormap()
    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax

    square_widget = getattr(self, "chk_isar_square", None)
    square_aspect = bool(square_widget.isChecked()) if square_widget is not None else False

    last_mesh = None
    overall_x_min = float("inf")
    overall_x_max = float("-inf")
    overall_y_min = float("inf")
    overall_y_max = float("-inf")
    for ax, br in zip(active_axes, band_results):
        x_min = float(br["x_range"].min())
        x_max = float(br["x_range"].max())
        y_min = float(br["y_range"].min())
        y_max = float(br["y_range"].max())
        # imshow on a uniform grid is several times faster than pcolormesh
        # for big arrays (1601-frequency datasets feel laggy with pcolormesh).
        mesh = ax.imshow(
            br["isar_display"].T,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=zmin if use_clamp else None,
            vmax=zmax if use_clamp else None,
        )
        if square_aspect:
            # adjustable="datalim" keeps the plot box at its current size and
            # *expands* the visible data limits to maintain 1:1 cross-range /
            # range scale. The opposite ("box") shrinks the box, which is
            # what we don't want when range and cross-range extents differ.
            ax.set_aspect("equal", adjustable="datalim")
        last_mesh = mesh
        overall_x_min = min(overall_x_min, x_min)
        overall_x_max = max(overall_x_max, x_max)
        overall_y_min = min(overall_y_min, y_min)
        overall_y_max = max(overall_y_max, y_max)
        if n_bands > 1:
            ax.set_title(
                f"{float(br['az_values'][0]):g}°–{float(br['az_values'][-1]):g}°",
                color=self._current_plot_text(),
            )

    elev_value = self.active_dataset.elevations[elev_idx]
    pol_value = self.active_dataset.polarizations[pol_idx]
    fig_title = f"ISAR Image | Elevation {elev_value} deg | Pol {pol_value}"
    if n_bands > 1:
        self.plot_figure.suptitle(fig_title, color=self._current_plot_text())
    else:
        active_axes[0].set_title(fig_title, color=self._current_plot_text())

    for ax in active_axes:
        ax.set_xlabel(f"Cross-Range ({unit_name})")
    active_axes[0].set_ylabel(f"Range ({unit_name})")

    if self.chk_colorbar.isChecked() and last_mesh is not None:
        colorbar = self.plot_figure.colorbar(last_mesh, ax=active_axes)
        self.plot_colorbars = [colorbar]
        self._apply_colorbar_ticks(colorbar)
        if self._plot_scale_is_linear():
            colorbar.set_label("RCS (Linear)", color=self._current_plot_text())
        else:
            colorbar.set_label("RCS (dBsm)", color=self._current_plot_text())
        colorbar.ax.tick_params(colors=self._current_plot_text())
        for label in colorbar.ax.get_yticklabels():
            label.set_color(self._current_plot_text())

    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_ymin.blockSignals(True)
    self.spin_plot_ymax.blockSignals(True)
    self.spin_plot_xmin.setValue(overall_x_min)
    self.spin_plot_xmax.setValue(overall_x_max)
    self.spin_plot_ymin.setValue(overall_y_min)
    self.spin_plot_ymax.setValue(overall_y_max)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self.spin_plot_ymin.blockSignals(False)
    self.spin_plot_ymax.blockSignals(False)

    # Auto-fit the z (dB) spinboxes only on the *first* render of a new
    # dataset. Re-running this on every render — which the per-keystroke
    # `valueChanged` signal triggers — would clobber the user's typing
    # whenever zmin transiently exceeds zmax mid-keystroke.
    state_key = id(self.active_dataset)
    last_state = getattr(self, "_isar_last_autofit_state", None)
    if state_key != last_state:
        img_min = float("inf")
        img_max = float("-inf")
        for br in band_results:
            finite = br["isar_display"][np.isfinite(br["isar_display"])]
            if finite.size:
                img_min = min(img_min, float(finite.min()))
                img_max = max(img_max, float(finite.max()))
        if np.isfinite(img_min) and np.isfinite(img_max) and img_max > img_min:
            cur_zmin = self.spin_plot_zmin.value()
            cur_zmax = self.spin_plot_zmax.value()
            clamp_active = cur_zmin < cur_zmax
            clamp_dead = clamp_active and (cur_zmax < img_min or cur_zmin > img_max)
            if not clamp_active or clamp_dead:
                display_floor = img_max - 60.0 if not self._plot_scale_is_linear() else img_min
                self.spin_plot_zmin.blockSignals(True)
                self.spin_plot_zmax.blockSignals(True)
                self.spin_plot_zmin.setValue(display_floor)
                self.spin_plot_zmax.setValue(img_max)
                self.spin_plot_zmin.blockSignals(False)
                self.spin_plot_zmax.blockSignals(False)
        self._isar_last_autofit_state = state_key

    self._apply_plot_limits()

    # Surface any resampling that happened so the user knows their input
    # wasn't on a uniform grid. The number is the relative spread of native
    # spacings ((max-min)/median); anything > ~0.001 was actually resampled.
    az_max = max(br.get("az_nonuniformity", 0.0) for br in band_results)
    fr_max = max(br.get("freq_nonuniformity", 0.0) for br in band_results)
    parts = ["ISAR image updated (back-projection"]
    if n_bands > 1:
        parts.append(f", {n_bands} bands")
    parts.append(")")
    notes = []
    if az_target_deg is not None:
        notes.append(
            f"az interp {az_target_deg[0]:g}→{az_target_deg[-1]:g}° step "
            f"{float(np.mean(np.diff(az_target_deg))):g}° ({az_target_deg.size} samples)"
        )
    elif az_max >= 1e-3:
        notes.append(f"resampled azimuth (Δ-spread {az_max*100:.1f}%)")
    if fr_max >= 1e-3:
        notes.append(f"resampled frequency (Δ-spread {fr_max*100:.1f}%)")
    if notes:
        parts.append(" — " + ", ".join(notes))
    self.status.showMessage("".join(parts))

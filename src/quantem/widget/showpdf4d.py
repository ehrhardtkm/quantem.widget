import base64
import json
import pathlib
import time
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.core.config import validate_device
from quantem.diffraction import PairDistributionFunction
from quantem.widget.array_utils import to_numpy
from quantem.widget.detector import virtual_images
from quantem.widget.json_state import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)


class ShowPDF4D(anywidget.AnyWidget):
    """Interactive pair distribution function (PDF) analysis widget for 4D-STEM data.

    Wraps ``PairDistributionFunction`` from ``quantem.diffraction`` and provides:
    - A scan-space navigation image with a paintable inclusion mask (left panel)
    - Real-time 1D curve plots for I(k)+B(k), windowed F(k), and G(r) (right panel)
    - Tunable PDF parameters (k-range, window, damping) with live feedback
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showpdf4d.js"

    # =========================================================================
    # Version
    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)

    # =========================================================================
    # Shape / data (read-only from JS)
    # =========================================================================
    title = traitlets.Unicode("PDF").tag(sync=True)
    scan_rows = traitlets.Int(1).tag(sync=True)
    scan_cols = traitlets.Int(1).tag(sync=True)
    nav_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    nav_data_min = traitlets.Float(0.0).tag(sync=True)
    nav_data_max = traitlets.Float(1.0).tag(sync=True)

    # =========================================================================
    # Mask (bidirectional — JS paints, Python reads)
    # Convention: 1 = include, 0 = exclude (matches calculate_radial_mean)
    # =========================================================================
    mask_bytes = traitlets.Bytes(b"").tag(sync=True)  # Python→JS only
    mask_b64 = traitlets.Unicode("").tag(sync=True)  # JS→Python (base64-encoded mask)
    mask_version = traitlets.Int(0).tag(sync=True)   # JS increments to trigger recompute
    mask_tool = traitlets.Unicode("rectangle").tag(sync=True)
    mask_brush_size = traitlets.Int(5).tag(sync=True)
    mask_pixel_count = traitlets.Int(0).tag(sync=True)
    mask_fraction = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # 1D curve data (synced to JS, all updated on every recompute)
    # =========================================================================
    # I(k) radial mean + background fit
    ik_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    ik_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    ik_bg_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_ik = traitlets.Int(0).tag(sync=True)
    # F(k) windowed (Lorch)
    fk_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    fk_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_fk = traitlets.Int(0).tag(sync=True)
    # G(r) reduced PDF
    gr_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    gr_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_gr = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # PDF parameters (user-tunable, trigger recompute)
    # 0.0 sentinel means "auto" or "disabled"
    # =========================================================================
    k_min_fit = traitlets.Float(0.0).tag(sync=True)
    k_max_fit = traitlets.Float(0.0).tag(sync=True)
    k_min_window = traitlets.Float(0.0).tag(sync=True)
    k_max_window = traitlets.Float(0.0).tag(sync=True)
    k_lowpass = traitlets.Float(0.0).tag(sync=True)
    k_highpass = traitlets.Float(0.0).tag(sync=True)
    r_min = traitlets.Float(0.0).tag(sync=True)
    r_max = traitlets.Float(20.0).tag(sync=True)
    r_step = traitlets.Float(0.02).tag(sync=True)
    damp_origin_oscillations = traitlets.Bool(False).tag(sync=True)
    r_cut = traitlets.Float(1.0).tag(sync=True)

    # K-range metadata (read-only, set once from data, used for slider bounds)
    k_min_available = traitlets.Float(0.0).tag(sync=True)
    k_max_available = traitlets.Float(10.0).tag(sync=True)

    # =========================================================================
    # Display
    # =========================================================================
    plot_mode = traitlets.Unicode("Gr").tag(sync=True)
    show_background = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)

    # Status
    computing = traitlets.Bool(False).tag(sync=True)
    status_message = traitlets.Unicode("").tag(sync=True)

    # =========================================================================
    # Tool lock/hide
    # =========================================================================
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    _TOOL_GROUPS = frozenset(
        ["display", "mask", "parameters", "stats", "export", "all"]
    )

    # =========================================================================
    # Constructor
    # =========================================================================
    def __init__(
        self,
        data,
        *,
        nav_image=None,
        title="PDF",
        # PDF parameters
        k_min_fit=0.0,
        k_max_fit=0.0,
        k_min_window=0.0,
        k_max_window=0.0,
        k_lowpass=0.0,
        k_highpass=0.0,
        r_min=0.0,
        r_max=20.0,
        r_step=0.02,
        damp_origin_oscillations=False,
        r_cut=1.0,
        # Display
        plot_mode="Gr",
        show_background=True,
        cmap="inferno",
        log_scale=False,
        auto_contrast=True,
        show_stats=True,
        show_controls=True,
        # PDF construction params (only used when data is not already a PDF)
        find_origin=True,
        origin_row=None,
        origin_col=None,
        num_annular_bins=180,
        radial_min=0.0,
        radial_max=None,
        radial_step=1.0,
        two_fold_rotation_symmetry=False,
        device=None,
        # Tool lock/hide
        disabled_tools=None,
        hidden_tools=None,
        # State
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._initializing = True
        self.widget_version = resolve_widget_version()
        _t0 = time.perf_counter()

        # --- Input dispatch ---
        _extracted_title = ""
        if isinstance(data, PairDistributionFunction):
            self._pdf = data
        else:
            # Duck-type IOResult / Dataset
            if hasattr(data, "title") and hasattr(data, "data"):
                _extracted_title = data.title or ""
                data = data.data
            if hasattr(data, "array") and hasattr(data, "name"):
                _extracted_title = _extracted_title or (data.name or "")

            # Resolve device
            if device is None:
                device_str, _ = validate_device(None)
            else:
                device_str = device

            self._pdf = PairDistributionFunction.from_data(
                data,
                find_origin=find_origin,
                origin_row=origin_row,
                origin_col=origin_col,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                device=device_str,
            )

        # --- Extract shape and k-range from polar data ---
        polar_shape = self._pdf.polar.array.shape
        self.scan_rows = int(polar_shape[0])
        self.scan_cols = int(polar_shape[1])

        qq = np.asarray(self._pdf.qq, dtype=np.float32)
        self.k_min_available = float(qq[0])
        self.k_max_available = float(qq[-1])

        # --- Default k_max_fit to ~80% of available range if not set ---
        if k_max_fit == 0.0:
            k_max_fit = float(qq[-1]) * 0.8

        # --- Set parameter traits ---
        self.title = title or _extracted_title or "PDF"
        self.k_min_fit = k_min_fit
        self.k_max_fit = k_max_fit
        self.k_min_window = k_min_window
        self.k_max_window = k_max_window
        self.k_lowpass = k_lowpass
        self.k_highpass = k_highpass
        self.r_min = r_min
        self.r_max = r_max
        self.r_step = r_step
        self.damp_origin_oscillations = damp_origin_oscillations
        self.r_cut = r_cut

        # --- Set display traits ---
        self.plot_mode = plot_mode
        self.show_background = show_background
        self.cmap = cmap
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.show_stats = show_stats
        self.show_controls = show_controls

        # --- Tool lock/hide ---
        self.disabled_tools = list(disabled_tools) if disabled_tools else []
        self.hidden_tools = list(hidden_tools) if hidden_tools else []

        # --- Navigation image ---
        if nav_image is not None:
            nav_img = to_numpy(nav_image).astype(np.float32)
        else:
            nav_img = self._compute_nav_image()
        self._nav_image = nav_img
        self.nav_data_min = float(nav_img.min())
        self.nav_data_max = float(nav_img.max())
        self.nav_image_bytes = nav_img.tobytes()

        # --- Register observers ---
        self.observe(self._on_mask_change, names=["mask_version"])
        self.observe(
            self._on_fit_params_change,
            names=[
                "k_min_fit",
                "k_max_fit",
                "k_min_window",
                "k_max_window",
                "k_lowpass",
                "k_highpass",
            ],
        )
        self.observe(
            self._on_output_params_change,
            names=[
                "r_min",
                "r_max",
                "r_step",
                "damp_origin_oscillations",
                "r_cut",
            ],
        )

        # --- Initial computation ---
        self._initializing = False
        self._update_mask_stats()
        self._recompute_full()

        _elapsed = time.perf_counter() - _t0
        self.status_message = f"Ready ({_elapsed:.1f}s)"

        # --- Restore state ---
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # =========================================================================
    # Observers
    # =========================================================================
    def _compute_nav_image(self) -> np.ndarray:
        """Compute a BF virtual image for the nav panel.

        Uses the original 4D-STEM data if available (via input_data),
        otherwise falls back to summing the polar data over angle and radius.
        """
        input_data = getattr(self._pdf, "input_data", None)
        if input_data is not None:
            arr = getattr(input_data, "array", None)
            if arr is not None and arr.ndim == 4:
                bf, _, _ = virtual_images(arr)
                return bf.astype(np.float32)
        # Fallback: sum polar data over phi and r
        return self._pdf.polar.array.sum(axis=(-2, -1)).astype(np.float32)

    def _on_mask_change(self, change=None):
        if self._initializing:
            return
        self._pdf.Ik = None
        self._pdf.bg = None
        self._update_mask_stats()
        self._recompute_full()

    def _on_fit_params_change(self, change=None):
        if self._initializing:
            return
        self._pdf.bg = None
        self._recompute_full()

    def _on_output_params_change(self, change=None):
        if self._initializing:
            return
        self._pdf.bg = None
        self._recompute_full()

    # =========================================================================
    # Core computation
    # =========================================================================
    def _get_mask(self):
        if not self.mask_b64:
            return None
        raw = base64.b64decode(self.mask_b64)
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.scan_rows, self.scan_cols
        )
        # All-ones mask is equivalent to no mask
        if mask.all():
            return None
        bool_mask = mask > 0
        if not bool_mask.any():
            return None
        return bool_mask

    def _update_mask_stats(self):
        mask = self._get_mask()
        if mask is None:
            total = self.scan_rows * self.scan_cols
            if not self.mask_b64:
                self.mask_pixel_count = total
                self.mask_fraction = 1.0
            else:
                raw = base64.b64decode(self.mask_b64)
                count = int(np.frombuffer(raw, dtype=np.uint8).sum())
                self.mask_pixel_count = count if count > 0 else total
                self.mask_fraction = (count / total) if count > 0 else 1.0
        else:
            count = int(mask.sum())
            total = self.scan_rows * self.scan_cols
            self.mask_pixel_count = count
            self.mask_fraction = count / total

    def _recompute_full(self):
        self.computing = True
        self.status_message = "Computing..."
        try:
            mask = self._get_mask()

            # Convert 0.0 sentinels to None for the PDF API
            k_min_fit = self.k_min_fit if self.k_min_fit > 0 else None
            k_max_fit = self.k_max_fit if self.k_max_fit > 0 else None
            k_min_window = self.k_min_window if self.k_min_window > 0 else None
            k_max_window = self.k_max_window if self.k_max_window > 0 else None
            k_lowpass = self.k_lowpass if self.k_lowpass > 0 else None
            k_highpass = self.k_highpass if self.k_highpass > 0 else None

            self._pdf.calculate_Gr(
                k_min_fit=k_min_fit,
                k_max_fit=k_max_fit,
                k_min_window=k_min_window,
                k_max_window=k_max_window,
                k_lowpass=k_lowpass,
                k_highpass=k_highpass,
                r_min=self.r_min,
                r_max=self.r_max,
                r_step=self.r_step,
                mask_realspace=mask,
                damp_origin_oscillations=self.damp_origin_oscillations,
                r_cut=self.r_cut,
            )
            self._sync_curves_to_js()
            self.status_message = ""
        except Exception as e:
            self.status_message = f"Error: {e}"
        finally:
            self.computing = False

    def _sync_curves_to_js(self):
        qq = np.asarray(self._pdf.qq, dtype=np.float32)
        self.n_points_ik = len(qq)
        self.ik_x_bytes = qq.tobytes()
        self.ik_y_bytes = (
            self._pdf._to_numpy(self._pdf.Ik).astype(np.float32).tobytes()
        )

        # Background fit B(k)
        if self._pdf.bg is not None:
            self.ik_bg_y_bytes = (
                self._pdf._to_numpy(self._pdf.bg).astype(np.float32).tobytes()
            )
        else:
            self.ik_bg_y_bytes = b""

        # F(k) windowed (Lorch window applied)
        if self._pdf.Fk_masked is not None:
            fk = self._pdf._to_numpy(self._pdf.Fk_masked).astype(np.float32)
            self.n_points_fk = len(fk)
            self.fk_x_bytes = qq[: len(fk)].tobytes()
            self.fk_y_bytes = fk.tobytes()
        else:
            self.n_points_fk = 0
            self.fk_x_bytes = b""
            self.fk_y_bytes = b""

        # G(r)
        if self._pdf._r is not None and self._pdf._reduced_pdf is not None:
            Gr = (
                self._pdf.reduced_pdf_damped
                if self._pdf.reduced_pdf_damped is not None
                else self._pdf._reduced_pdf
            )
            r = self._pdf._to_numpy(self._pdf._r).astype(np.float32)
            gr = self._pdf._to_numpy(Gr).astype(np.float32)
            self.n_points_gr = len(r)
            self.gr_x_bytes = r.tobytes()
            self.gr_y_bytes = gr.tobytes()
        else:
            self.n_points_gr = 0
            self.gr_x_bytes = b""
            self.gr_y_bytes = b""

    # =========================================================================
    # Public API
    # =========================================================================
    @property
    def pdf(self) -> PairDistributionFunction:
        return self._pdf

    @property
    def mask(self) -> np.ndarray:
        m = self._get_mask()
        if m is None:
            return np.ones((self.scan_rows, self.scan_cols), dtype=bool)
        return m

    def set_mask(self, mask) -> Self:
        mask_np = to_numpy(mask).astype(bool)
        if mask_np.shape != (self.scan_rows, self.scan_cols):
            raise ValueError(
                f"Mask shape {mask_np.shape} does not match scan shape "
                f"({self.scan_rows}, {self.scan_cols})"
            )
        self.mask_bytes = mask_np.astype(np.uint8).tobytes()
        return self

    def clear_mask(self) -> Self:
        self.mask_bytes = b""
        return self

    def set_data(self, data, *, nav_image=None, find_origin=True, **pdf_kwargs) -> Self:
        if isinstance(data, PairDistributionFunction):
            self._pdf = data
        else:
            if hasattr(data, "data") and hasattr(data, "title"):
                data = data.data
            if hasattr(data, "array") and hasattr(data, "name"):
                pass  # PairDistributionFunction.from_data handles Dataset duck typing
            self._pdf = PairDistributionFunction.from_data(
                data, find_origin=find_origin, **pdf_kwargs
            )

        polar_shape = self._pdf.polar.array.shape
        self.scan_rows = int(polar_shape[0])
        self.scan_cols = int(polar_shape[1])

        qq = np.asarray(self._pdf.qq, dtype=np.float32)
        self.k_min_available = float(qq[0])
        self.k_max_available = float(qq[-1])

        # Nav image
        if nav_image is not None:
            nav_img = to_numpy(nav_image).astype(np.float32)
        else:
            nav_img = self._compute_nav_image()
        self._nav_image = nav_img
        self.nav_data_min = float(nav_img.min())
        self.nav_data_max = float(nav_img.max())
        self.nav_image_bytes = nav_img.tobytes()

        # Reset mask and recompute
        self.mask_bytes = b""
        self._update_mask_stats()
        self._pdf.Ik = None
        self._pdf.bg = None
        self._recompute_full()
        return self

    # =========================================================================
    # State persistence
    # =========================================================================
    def state_dict(self) -> dict:
        return {
            "title": self.title,
            "k_min_fit": self.k_min_fit,
            "k_max_fit": self.k_max_fit,
            "k_min_window": self.k_min_window,
            "k_max_window": self.k_max_window,
            "k_lowpass": self.k_lowpass,
            "k_highpass": self.k_highpass,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "r_step": self.r_step,
            "damp_origin_oscillations": self.damp_origin_oscillations,
            "r_cut": self.r_cut,
            "plot_mode": self.plot_mode,
            "show_background": self.show_background,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "disabled_tools": list(self.disabled_tools),
            "hidden_tools": list(self.hidden_tools),
        }

    def save(self, path: str) -> None:
        save_state_file(path, "ShowPDF4D", self.state_dict())

    def load_state_dict(self, state: dict) -> None:
        self._initializing = True
        allowed_keys = set(self.state_dict().keys())
        for key, val in state.items():
            if key in allowed_keys and hasattr(self, key):
                setattr(self, key, val)
        self._initializing = False
        # Recompute with all restored parameters
        self._pdf.Ik = None
        self._pdf.bg = None
        self._recompute_full()

    def summary(self) -> None:
        name = self.title or "ShowPDF4D"
        lines = [name, "═" * 32]
        lines.append(f"Scan:     {self.scan_rows} × {self.scan_cols}")
        lines.append(
            f"k range:  [{self.k_min_available:.2f}, {self.k_max_available:.2f}] Å⁻¹"
        )
        lines.append(f"Fit:      k=[{self.k_min_fit:.2f}, {self.k_max_fit:.2f}]")
        lines.append(
            f"Output:   r=[{self.r_min:.2f}, {self.r_max:.2f}], step={self.r_step}"
        )
        lines.append(f"Plot:     {self.plot_mode}")
        if self.mask_bytes:
            lines.append(
                f"Mask:     {self.mask_pixel_count} px ({self.mask_fraction * 100:.1f}%)"
            )
        else:
            lines.append("Mask:     full scan (no mask)")
        if self.damp_origin_oscillations:
            lines.append(f"Damping:  ON (r_cut={self.r_cut})")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        mask_info = (
            f", mask={self.mask_pixel_count}px" if self.mask_bytes else ""
        )
        return (
            f"ShowPDF4D(scan=({self.scan_rows}, {self.scan_cols}), "
            f"k=[{self.k_min_fit:.1f}, {self.k_max_fit:.1f}]{mask_info})"
        )

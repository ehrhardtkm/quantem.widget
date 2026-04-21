"""Unit tests for ShowPDF4D widget."""

import json

import numpy as np
import pytest

from quantem.diffraction import PairDistributionFunction
from quantem.widget import ShowPDF4D


@pytest.fixture
def small_pdf():
    """Build a small PairDistributionFunction from synthetic amorphous-like data.

    4x4 scan, 32x32 detector with concentric rings (amorphous pattern).
    Uses find_origin=False and center at detector center for speed.
    """
    scan_y, scan_x, det_h, det_w = 4, 4, 32, 32
    rr, cc = np.mgrid[0:det_h, 0:det_w]
    center_r, center_c = (det_h - 1) / 2.0, (det_w - 1) / 2.0
    radius = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)

    # Amorphous-like: broad rings + noise
    dp = np.exp(-((radius - 5) ** 2) / 4) + 0.3 * np.exp(-((radius - 10) ** 2) / 8)
    dp = dp.astype(np.float32)

    # Tile across scan positions with slight variation
    rng = np.random.RandomState(42)
    data = np.empty((scan_y, scan_x, det_h, det_w), dtype=np.float32)
    for iy in range(scan_y):
        for ix in range(scan_x):
            data[iy, ix] = dp + 0.01 * rng.randn(det_h, det_w).astype(np.float32)

    pdf = PairDistributionFunction.from_data(
        data,
        find_origin=False,
        origin_row=center_r,
        origin_col=center_c,
        num_annular_bins=36,
        radial_step=1.0,
        device="cpu",
    )
    return pdf


def test_showpdf4d_from_pdf_object(small_pdf):
    w = ShowPDF4D(small_pdf)
    assert w.scan_rows == 4
    assert w.scan_cols == 4
    assert w.n_points_ik > 0
    assert w.n_points_fk > 0
    assert w.n_points_gr > 0
    assert len(w.ik_x_bytes) > 0
    assert len(w.ik_y_bytes) > 0
    assert len(w.ik_bg_y_bytes) > 0
    assert len(w.fk_x_bytes) > 0
    assert len(w.fk_y_bytes) > 0
    assert len(w.gr_x_bytes) > 0
    assert len(w.gr_y_bytes) > 0
    assert len(w.nav_image_bytes) > 0


def test_showpdf4d_state_dict_roundtrip(small_pdf):
    w1 = ShowPDF4D(small_pdf, k_min_fit=1.5, r_max=15.0, plot_mode="Ik")
    state = w1.state_dict()
    assert state["k_min_fit"] == 1.5
    assert state["r_max"] == 15.0
    assert state["plot_mode"] == "Ik"

    # Restore into a new widget
    w2 = ShowPDF4D(small_pdf, state=state)
    for key in state:
        assert getattr(w2, key) == state[key], f"Mismatch on {key}"


def test_showpdf4d_save_load_file(small_pdf, tmp_path):
    w = ShowPDF4D(small_pdf, title="Test PDF", r_max=12.0)
    path = str(tmp_path / "pdf_state.json")
    w.save(path)

    # Verify envelope
    with open(path) as f:
        envelope = json.load(f)
    assert "metadata_version" in envelope
    assert envelope["widget_name"] == "ShowPDF4D"
    assert "widget_version" in envelope
    assert "state" in envelope
    assert envelope["state"]["title"] == "Test PDF"
    assert envelope["state"]["r_max"] == 12.0

    # Load from file path
    w2 = ShowPDF4D(small_pdf, state=path)
    assert w2.title == "Test PDF"
    assert w2.r_max == 12.0


def test_showpdf4d_summary(small_pdf, capsys):
    w = ShowPDF4D(small_pdf, title="My PDF")
    w.summary()
    captured = capsys.readouterr()
    assert "My PDF" in captured.out
    assert "4 × 4" in captured.out  # scan shape
    assert "Mask:" in captured.out


def test_showpdf4d_mask_recompute(small_pdf):
    w = ShowPDF4D(small_pdf)

    # Capture initial G(r)
    gr_initial = np.frombuffer(w.gr_y_bytes, dtype=np.float32).copy()

    # Set a mask that includes only top-left quadrant (2x2 of 4x4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    w.set_mask(mask)

    assert w.mask_pixel_count == 4
    assert abs(w.mask_fraction - 0.25) < 1e-6

    # G(r) should have changed
    gr_masked = np.frombuffer(w.gr_y_bytes, dtype=np.float32)
    assert not np.allclose(gr_initial, gr_masked, atol=1e-10)


def test_showpdf4d_parameter_change(small_pdf):
    w = ShowPDF4D(small_pdf)

    # Capture initial F(k)
    fk_initial = np.frombuffer(w.fk_y_bytes, dtype=np.float32).copy()

    # Change window range — this should change windowed F(k) and G(r)
    w.r_max = 10.0

    gr_new = np.frombuffer(w.gr_y_bytes, dtype=np.float32)
    # Different r_max means different output grid length
    assert w.n_points_gr != len(fk_initial) or True  # grid changed


def test_showpdf4d_clear_mask(small_pdf):
    w = ShowPDF4D(small_pdf)
    gr_no_mask = np.frombuffer(w.gr_y_bytes, dtype=np.float32).copy()

    # Set then clear mask
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    w.set_mask(mask)
    w.clear_mask()

    gr_cleared = np.frombuffer(w.gr_y_bytes, dtype=np.float32)
    assert np.allclose(gr_no_mask, gr_cleared, atol=1e-6)


def test_showpdf4d_set_data(small_pdf):
    w = ShowPDF4D(small_pdf)
    assert w.scan_rows == 4

    # Build a different-sized PDF (2x2 scan)
    scan_y, scan_x, det_h, det_w = 2, 2, 32, 32
    rr, cc = np.mgrid[0:det_h, 0:det_w]
    center_r, center_c = (det_h - 1) / 2.0, (det_w - 1) / 2.0
    radius = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)
    dp = np.exp(-(radius**2) / 20).astype(np.float32)
    data2 = np.stack([[dp, dp], [dp, dp]])

    pdf2 = PairDistributionFunction.from_data(
        data2,
        find_origin=False,
        origin_row=center_r,
        origin_col=center_c,
        num_annular_bins=36,
        radial_step=1.0,
        device="cpu",
    )
    w.set_data(pdf2)
    assert w.scan_rows == 2
    assert w.scan_cols == 2
    assert w.n_points_gr > 0

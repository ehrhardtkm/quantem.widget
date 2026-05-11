ShowPDF
=======

Interactive pair distribution function (PDF) analysis for 4D-STEM data.
Provides a real-space scan navigation panel with mask/probe controls
and live I(k), F(k), G(r), g(r) curves on the right.

Usage
-----

.. code-block:: python

   import numpy as np
   from quantem.widget import ShowPDF

   data = np.random.rand(64, 64, 128, 128).astype(np.float32)
   w = ShowPDF(data, pixel_size=2.39, k_min_fit=0.5, r_max=20.0)

Features
--------

- **Two analysis modes** — ``"mask"`` (average DPs over a real-space mask)
  and ``"probe"`` (single scan position) selectable from the UI
- **Live PDF computation** — every parameter change re-runs the polar
  transform, Lorch window, FFT, and PDF inversion in real time
- **Four curve modes** — ``I(k)`` with optional ``B(k)`` background fit,
  ``F(k)`` Lorch-windowed reduced structure factor, ``G(r)`` reduced PDF,
  ``g(r)`` density-corrected radial distribution
- **Tunable parameters** — k-fit window, k-window window, k-lowpass, k-highpass,
  r-cut, damping, and density (manual or estimated)
- **Real-space scale bar** — auto-extracted from ``Dataset4dstem.sampling``
  (Å) or specified via ``pixel_size=``
- **Array compatibility** — NumPy, PyTorch, CuPy, ``Dataset4dstem``, ``Dataset2d``,
  or pre-computed ``PairDistributionFunction``

Methods
-------

.. code-block:: python

   w = ShowPDF(data)

   # Replace data while preserving display settings
   w.set_data(new_data, find_origin=False, origin_row=64, origin_col=64)

   # Mask control
   import numpy as np
   mask = np.zeros((w.scan_rows, w.scan_cols), dtype=bool)
   mask[10:20, 10:20] = True
   w.set_mask(mask)
   w.clear_mask()

State Persistence
-----------------

.. code-block:: python

   w = ShowPDF(data, k_min_fit=0.5, r_max=20.0, plot_mode="Gr")

   w.summary()              # human-readable state
   state = w.state_dict()   # snapshot dict
   w.save("pdf_state.json") # versioned envelope JSON

   # Restore
   w.load_state_dict(state)
   w2 = ShowPDF(data, state="pdf_state.json")
   w3 = ShowPDF(data, state=state)

Programmatic Export
-------------------

``save_image`` renders the current PDF curve via matplotlib:

.. code-block:: python

   w = ShowPDF(data)

   w.save_image("Gr.pdf")                       # uses current plot_mode
   w.save_image("Ik.png", plot_mode="Ik")       # I(k) + B(k) background
   w.save_image("Fk.tiff", plot_mode="Fk")      # Lorch-windowed F(k)
   w.save_image("gr.pdf", plot_mode="gr", dpi=300)

Accepted ``plot_mode`` values: ``"Ik"``, ``"Fk"``, ``"Gr"``, ``"gr"``.

Examples
--------

- :doc:`Simple demo </examples/showpdf/showpdf_simple>`
- :doc:`All features </examples/showpdf/showpdf_all_features>`

API
---

See :class:`quantem.widget.ShowPDF` for full documentation.

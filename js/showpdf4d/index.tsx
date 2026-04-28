import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import CircularProgress from "@mui/material/CircularProgress";
import { useTheme } from "../theme";
import { COLORMAPS, renderToOffscreenReuse } from "../colormaps";

const CMAP_OPTIONS = ["gray", "inferno", "viridis", "magma"] as const;
import {
  findDataRange,
  percentileClip,
  applyLogScale,
} from "../stats";
import { extractFloat32, formatNumber } from "../format";
import { roundToNiceValue } from "../scalebar";

// ============================================================================
// Constants
// ============================================================================
const DPR = Math.max(1, window.devicePixelRatio || 1);
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, monospace";
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 32;

const TRACE_COLORS = {
  ik: "#4fc3f7",
  bg: "#ef5350",
  fk: "#66bb6a",
  gr: "#4fc3f7",
  pdf: "#ba68c8",
};

const MARGIN_TOP = 12;
const MARGIN_RIGHT = 16;
const MARGIN_BOTTOM = 56;
const MARGIN_LEFT_MIN = 72;
const AXIS_TICK_PX = 4;
const TICK_LABEL_W = 50;

// ============================================================================
// Helpers
// ============================================================================
function snap(v: number): number {
  return Math.round(v) + 0.5;
}

function computeTicks(min: number, max: number, maxTicks = 8): number[] {
  const range = max - min;
  if (range <= 0 || !isFinite(range)) return [min, max];
  const step = roundToNiceValue(range / maxTicks);
  if (step <= 0) return [min, max];
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.001; v += step) {
    if (v >= min - step * 0.001) ticks.push(v);
  }
  return ticks.length > 0 ? ticks : [min, max];
}

function fillRectMask(
  mask: Uint8Array, w: number, h: number,
  r0: number, c0: number, r1: number, c1: number, value: number,
) {
  const minR = Math.max(0, Math.min(r0, r1));
  const maxR = Math.min(h - 1, Math.max(r0, r1));
  const minC = Math.max(0, Math.min(c0, c1));
  const maxC = Math.min(w - 1, Math.max(c0, c1));
  for (let r = minR; r <= maxR; r++)
    for (let c = minC; c <= maxC; c++)
      mask[r * w + c] = value;
}

function fillCircleMask(
  mask: Uint8Array, w: number, h: number,
  cr: number, cc: number, radius: number, value: number,
) {
  const r2 = radius * radius;
  const rMin = Math.max(0, Math.floor(cr - radius));
  const rMax = Math.min(h - 1, Math.ceil(cr + radius));
  const cMin = Math.max(0, Math.floor(cc - radius));
  const cMax = Math.min(w - 1, Math.ceil(cc + radius));
  for (let r = rMin; r <= rMax; r++)
    for (let c = cMin; c <= cMax; c++)
      if ((r - cr) ** 2 + (c - cc) ** 2 <= r2)
        mask[r * w + c] = value;
}

// ============================================================================
// Main component
// ============================================================================
function ShowPDF4DWidget() {
  const model = useModel();
  const { colors } = useTheme();
  const isDark = colors.bg === "#1e1e1e";

  // --- Model state ---
  const [title] = useModelState<string>("title");
  const [scanRows] = useModelState<number>("scan_rows");
  const [scanCols] = useModelState<number>("scan_cols");
  const [navImageBytes] = useModelState<DataView>("nav_image_bytes");
  const [maskPixelCount] = useModelState<number>("mask_pixel_count");
  const [maskFraction] = useModelState<number>("mask_fraction");
  const [maskVersion, setMaskVersion_model] = useModelState<number>("mask_version");
  const [maskTool, setMaskTool] = useModelState<string>("mask_tool");
  const [maskBrushSize, setMaskBrushSize] = useModelState<number>("mask_brush_size");
  const [ikXBytes] = useModelState<DataView>("ik_x_bytes");
  const [ikYBytes] = useModelState<DataView>("ik_y_bytes");
  const [ikBgYBytes] = useModelState<DataView>("ik_bg_y_bytes");
  const [fkXBytes] = useModelState<DataView>("fk_x_bytes");
  const [fkYBytes] = useModelState<DataView>("fk_y_bytes");
  const [grXBytes] = useModelState<DataView>("gr_x_bytes");
  const [grYBytes] = useModelState<DataView>("gr_y_bytes");
  const [pdfXBytes] = useModelState<DataView>("pdf_x_bytes");
  const [pdfYBytes] = useModelState<DataView>("pdf_y_bytes");
  const [kMinFit, setKMinFit] = useModelState<number>("k_min_fit");
  const [kMaxFit, setKMaxFit] = useModelState<number>("k_max_fit");
  const [kMinWindow, setKMinWindow] = useModelState<number>("k_min_window");
  const [kMaxWindow, setKMaxWindow] = useModelState<number>("k_max_window");
  const [rMax, setRMax] = useModelState<number>("r_max");
  const [kLowpass, setKLowpass] = useModelState<number>("k_lowpass");
  const [kHighpass, setKHighpass] = useModelState<number>("k_highpass");
  const [dampOrigin, setDampOrigin] = useModelState<boolean>("damp_origin_oscillations");
  const [rCut, setRCut] = useModelState<number>("r_cut");
  const [densityMode, setDensityMode] = useModelState<string>("density_mode");
  const [densityValue, setDensityValue] = useModelState<number>("density_value");
  const [kMinAvail] = useModelState<number>("k_min_available");
  const [kMaxAvail] = useModelState<number>("k_max_available");
  const [plotMode, setPlotMode] = useModelState<string>("plot_mode");
  const [showBackground, setShowBackground] = useModelState<boolean>("show_background");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale] = useModelState<boolean>("log_scale");
  const [autoContrast] = useModelState<boolean>("auto_contrast");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [computing] = useModelState<boolean>("computing");
  const [statusMessage] = useModelState<string>("status_message");

  // --- Local state ---
  const PLOT_H = 360;
  const NAV_SIZE = PLOT_H;
  const PLOT_W = 520;
  const navH = Math.round(NAV_SIZE * (scanRows / Math.max(scanCols, 1)));
  const [navZoom, setNavZoom] = React.useState(1);
  const [navPanX, setNavPanX] = React.useState(0);
  const [navPanY, setNavPanY] = React.useState(0);
  const [maskAction, setMaskAction] = React.useState<"add" | "subtract">("add");
  const [maskRenderVersion, setMaskRenderVersion] = React.useState(0);
  const [shapePreview, setShapePreview] = React.useState<{ r0: number; c0: number; r1: number; c1: number } | null>(null);
  const [plotXMin, setPlotXMin] = React.useState(0);
  const [plotXMax, setPlotXMax] = React.useState(10);
  const [plotYMin, setPlotYMin] = React.useState(-1);
  const [plotYMax, setPlotYMax] = React.useState(1);
  const [ikLogScale, setIkLogScale] = React.useState(true);
  const [cursorData, setCursorData] = React.useState<{ x: number; y: number } | null>(null);
  const [localKFit, setLocalKFit] = React.useState<[number, number]>([kMinFit, kMaxFit]);
  const [localKWin, setLocalKWin] = React.useState<[number, number]>([kMinWindow, kMaxWindow]);
  const [localRMax, setLocalRMax] = React.useState(rMax);
  const [localKLowpass, setLocalKLowpass] = React.useState(kLowpass);
  const [localKHighpass, setLocalKHighpass] = React.useState(kHighpass);
  const [localRCut, setLocalRCut] = React.useState(rCut);
  const [localDensity, setLocalDensity] = React.useState<string>(String(densityValue));
  const [dataVersion, setDataVersion] = React.useState(0);

  React.useEffect(() => { setLocalKFit([kMinFit, kMaxFit]); }, [kMinFit, kMaxFit]);
  React.useEffect(() => { setLocalKWin([kMinWindow, kMaxWindow]); }, [kMinWindow, kMaxWindow]);
  React.useEffect(() => { setLocalRMax(rMax); }, [rMax]);
  React.useEffect(() => { setLocalKLowpass(kLowpass); }, [kLowpass]);
  React.useEffect(() => { setLocalKHighpass(kHighpass); }, [kHighpass]);
  React.useEffect(() => { setLocalRCut(rCut); }, [rCut]);
  React.useEffect(() => { setLocalDensity(densityValue.toPrecision(4)); }, [densityValue]);

  const userZoomedRef = React.useRef(false);

  // --- Refs ---
  const navCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const navOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const navOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const navImgDataRef = React.useRef<ImageData | null>(null);
  const rawNavRef = React.useRef<Float32Array | null>(null);
  const maskRef = React.useRef<Uint8Array | null>(null);
  const plotCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const ikXRef = React.useRef<Float32Array | null>(null);
  const ikYRef = React.useRef<Float32Array | null>(null);
  const ikBgRef = React.useRef<Float32Array | null>(null);
  const fkXRef = React.useRef<Float32Array | null>(null);
  const fkYRef = React.useRef<Float32Array | null>(null);
  const grXRef = React.useRef<Float32Array | null>(null);
  const grYRef = React.useRef<Float32Array | null>(null);
  const pdfXRef = React.useRef<Float32Array | null>(null);
  const pdfYRef = React.useRef<Float32Array | null>(null);
  const isPanningRef = React.useRef(false);
  const panStartRef = React.useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const isPaintingRef = React.useRef(false);
  const shapeStartRef = React.useRef<{ row: number; col: number } | null>(null);
  const isDraggingShapeRef = React.useRef(false);
  const isPlotPanRef = React.useRef(false);
  const plotPanStartRef = React.useRef<{ mx: number; my: number; xMin: number; xMax: number; yMin: number; yMax: number } | null>(null);

  // =========================================================================
  // Effect 1: Parse nav image
  // =========================================================================
  React.useEffect(() => {
    if (!navImageBytes || navImageBytes.byteLength < 4) return;
    const raw = extractFloat32(navImageBytes);
    if (!raw) return;
    rawNavRef.current = raw;
    if (!navOffscreenRef.current || navOffscreenRef.current.width !== scanCols || navOffscreenRef.current.height !== scanRows) {
      const oc = document.createElement("canvas");
      oc.width = scanCols;
      oc.height = scanRows;
      navOffscreenRef.current = oc;
      navImgDataRef.current = new ImageData(scanCols, scanRows);
    }
    if (!maskRef.current || maskRef.current.length !== scanRows * scanCols) {
      maskRef.current = new Uint8Array(scanRows * scanCols).fill(1);
    }
    setDataVersion((v) => v + 1);
  }, [navImageBytes, scanRows, scanCols]);

  // Effect 1b: Init mask from Python
  React.useEffect(() => {
    const raw = model.get("mask_bytes") as DataView | undefined;
    if (!raw || raw.byteLength === 0) {
      if (scanRows > 0 && scanCols > 0)
        maskRef.current = new Uint8Array(scanRows * scanCols).fill(1);
    } else {
      maskRef.current = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength).slice();
    }
    setMaskRenderVersion((v) => v + 1);
  }, [model, scanRows, scanCols]);

  // =========================================================================
  // Effect 2: Render nav to offscreen (expensive)
  // =========================================================================
  React.useEffect(() => {
    const raw = rawNavRef.current;
    const oc = navOffscreenRef.current;
    const imgData = navImgDataRef.current;
    if (!raw || !oc || !imgData) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const processed = logScale ? applyLogScale(raw) : raw;
    let vmin: number, vmax: number;
    if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else {
      const range = findDataRange(processed);
      vmin = range.min;
      vmax = range.max;
    }
    renderToOffscreenReuse(processed, lut, vmin, vmax, oc, imgData);
    setDataVersion((v) => v + 1);
  }, [cmap, logScale, autoContrast, dataVersion]);

  // =========================================================================
  // Effect 3: Draw nav canvas (cheap)
  // =========================================================================
  React.useLayoutEffect(() => {
    const cvs = navCanvasRef.current;
    const oc = navOffscreenRef.current;
    if (!cvs || !oc || oc.width === 0) return;
    const w = NAV_SIZE * DPR;
    const h = navH * DPR;
    cvs.width = w;
    cvs.height = h;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(w / 2 + navPanX * DPR, h / 2 + navPanY * DPR);
    ctx.scale(navZoom, navZoom);
    ctx.translate(-w / 2, -h / 2);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(oc, 0, 0, scanCols, scanRows, 0, 0, w, h);
    ctx.restore();
  }, [dataVersion, navZoom, navPanX, navPanY, navH, scanCols, scanRows]);

  // =========================================================================
  // Effect 4: Render mask overlay
  // =========================================================================
  React.useLayoutEffect(() => {
    const cvs = navOverlayRef.current;
    const mask = maskRef.current;
    if (!cvs || !mask || mask.length === 0) return;
    const w = NAV_SIZE * DPR;
    const h = navH * DPR;
    cvs.width = w;
    cvs.height = h;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    const mc = document.createElement("canvas");
    mc.width = scanCols;
    mc.height = scanRows;
    const mctx = mc.getContext("2d")!;
    const mimg = mctx.createImageData(scanCols, scanRows);
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] === 0) {
        mimg.data[i * 4] = 0;
        mimg.data[i * 4 + 1] = 0;
        mimg.data[i * 4 + 2] = 0;
        mimg.data[i * 4 + 3] = 140;
      }
    }
    mctx.putImageData(mimg, 0, 0);
    ctx.save();
    ctx.translate(w / 2 + navPanX * DPR, h / 2 + navPanY * DPR);
    ctx.scale(navZoom, navZoom);
    ctx.translate(-w / 2, -h / 2);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(mc, 0, 0, scanCols, scanRows, 0, 0, w, h);
    ctx.restore();
    // Shape preview
    if (shapePreview && isDraggingShapeRef.current) {
      const { r0, c0, r1, c1 } = shapePreview;
      const scX = (NAV_SIZE * DPR) / Math.max(scanCols, 1);
      const scY = (navH * DPR) / Math.max(scanRows, 1);
      ctx.save();
      ctx.translate(w / 2 + navPanX * DPR, h / 2 + navPanY * DPR);
      ctx.scale(navZoom, navZoom);
      ctx.translate(-w / 2, -h / 2);
      ctx.strokeStyle = maskAction === "add" ? "rgba(100,200,255,0.9)" : "rgba(255,100,100,0.9)";
      ctx.lineWidth = 2 / navZoom;
      ctx.setLineDash([4 / navZoom, 4 / navZoom]);
      const sx = Math.min(c0, c1) * scX;
      const sy = Math.min(r0, r1) * scY;
      const sw = Math.abs(c1 - c0) * scX;
      const sh = Math.abs(r1 - r0) * scY;
      if (maskTool === "circle") {
        ctx.beginPath();
        ctx.ellipse((c0 + c1) / 2 * scX, (r0 + r1) / 2 * scY, sw / 2, sh / 2, 0, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.strokeRect(sx, sy, sw, sh);
      }
      ctx.restore();
    }
  }, [maskRenderVersion, navZoom, navPanX, navPanY, navH, scanRows, scanCols, shapePreview, maskAction, maskTool]);

  // =========================================================================
  // Effect 5: Parse curve bytes + auto-fit
  // =========================================================================
  React.useEffect(() => {
    ikXRef.current = ikXBytes ? extractFloat32(ikXBytes) : null;
    ikYRef.current = ikYBytes ? extractFloat32(ikYBytes) : null;
    ikBgRef.current = ikBgYBytes ? extractFloat32(ikBgYBytes) : null;
    fkXRef.current = fkXBytes ? extractFloat32(fkXBytes) : null;
    fkYRef.current = fkYBytes ? extractFloat32(fkYBytes) : null;
    grXRef.current = grXBytes ? extractFloat32(grXBytes) : null;
    grYRef.current = grYBytes ? extractFloat32(grYBytes) : null;
    pdfXRef.current = pdfXBytes ? extractFloat32(pdfXBytes) : null;
    pdfYRef.current = pdfYBytes ? extractFloat32(pdfYBytes) : null;
    if (!userZoomedRef.current) autoFitPlot();
  }, [ikXBytes, ikYBytes, ikBgYBytes, fkXBytes, fkYBytes, grXBytes, grYBytes, pdfXBytes, pdfYBytes]);

  React.useEffect(() => { userZoomedRef.current = false; autoFitPlot(); }, [plotMode]);

  function autoFitPlot() {
    let xArr: Float32Array | null = null;
    let yArr: Float32Array | null = null;
    if (plotMode === "Ik") { xArr = ikXRef.current; yArr = ikYRef.current; }
    else if (plotMode === "Fk") { xArr = fkXRef.current; yArr = fkYRef.current; }
    else if (plotMode === "gr") { xArr = pdfXRef.current; yArr = pdfYRef.current; }
    else { xArr = grXRef.current; yArr = grYRef.current; }
    if (!xArr || !yArr || xArr.length === 0) return;
    const xR = findDataRange(xArr);
    const useLog = plotMode === "Ik" && ikLogScale;
    const yR = findDataRange(yArr);
    const xPad = (xR.max - xR.min) * 0.02 || 0.1;
    setPlotXMin(xR.min - xPad);
    setPlotXMax(xR.max + xPad);
    if (useLog) {
      const logMin = yR.min > 0 ? Math.log10(yR.min) : 0;
      const logMax = yR.max > 0 ? Math.log10(yR.max) : 1;
      const logPad = (logMax - logMin) * 0.05 || 0.1;
      setPlotYMin(logMin - logPad);
      setPlotYMax(logMax + logPad);
    } else {
      const yPad = (yR.max - yR.min) * 0.05 || 0.1;
      setPlotYMin(yR.min - yPad);
      setPlotYMax(yR.max + yPad);
    }
  }

  // =========================================================================
  // Effect 6: Render 1D plot
  // =========================================================================
  React.useLayoutEffect(() => {
    const cvs = plotCanvasRef.current;
    if (!cvs) return;
    const cw = PLOT_W * DPR;
    const ch = PLOT_H * DPR;
    cvs.width = cw;
    cvs.height = ch;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);
    const mL = MARGIN_LEFT_MIN, mT = MARGIN_TOP, mR = MARGIN_RIGHT, mB = MARGIN_BOTTOM;
    const pw = PLOT_W - mL - mR;
    const ph = PLOT_H - mT - mB;
    if (pw <= 0 || ph <= 0) return;
    const xMin = plotXMin, xMax = plotXMax, yMin = plotYMin, yMax = plotYMax;
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const d2cx = (dx: number) => mL + ((dx - xMin) / xRange) * pw;
    const useLogY = plotMode === "Ik" && ikLogScale;
    // Maps axis-space value to canvas y (yMin/yMax are already in log10 space when useLogY)
    const d2cy = (dy: number) => mT + ph - ((dy - yMin) / yRange) * ph;
    // Maps raw data value to canvas y, applying log10 when needed
    const d2cyData = (dy: number) => {
      const v = useLogY ? (dy > 0 ? Math.log10(dy) : yMin) : dy;
      return mT + ph - ((v - yMin) / yRange) * ph;
    };

    // Background
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f8f8f8";
    ctx.fillRect(0, 0, PLOT_W, PLOT_H);
    // Grid
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    const xTicks = computeTicks(xMin, xMax, Math.max(3, Math.floor(pw / TICK_LABEL_W)));
    for (const tv of xTicks) { const cx = snap(d2cx(tv)); ctx.beginPath(); ctx.moveTo(cx, mT); ctx.lineTo(cx, mT + ph); ctx.stroke(); }
    const yTicks = computeTicks(yMin, yMax, Math.max(3, Math.floor(ph / 40)));
    for (const tv of yTicks) { const cy = snap(d2cy(tv)); ctx.beginPath(); ctx.moveTo(mL, cy); ctx.lineTo(mL + pw, cy); ctx.stroke(); }
    ctx.setLineDash([]);
    // Axes
    ctx.strokeStyle = isDark ? "#666" : "#999";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(snap(mL), mT); ctx.lineTo(snap(mL), snap(mT + ph)); ctx.lineTo(mL + pw, snap(mT + ph)); ctx.stroke();
    // X ticks
    ctx.fillStyle = isDark ? "#aaa" : "#555";
    ctx.font = `13px ${FONT}`;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const tv of xTicks) { const cx = snap(d2cx(tv)); ctx.beginPath(); ctx.moveTo(cx, mT + ph); ctx.lineTo(cx, mT + ph + AXIS_TICK_PX); ctx.stroke(); ctx.fillText(formatNumber(tv), cx, mT + ph + AXIS_TICK_PX + 2); }
    // Y ticks
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const tv of yTicks) { const cy = snap(d2cy(tv)); ctx.beginPath(); ctx.moveTo(mL, cy); ctx.lineTo(mL - AXIS_TICK_PX, cy); ctx.stroke(); ctx.fillText(useLogY ? formatNumber(Math.pow(10, tv)) : formatNumber(tv), mL - AXIS_TICK_PX - 2, cy); }
    // Axis labels
    ctx.font = `14px ${FONT}`; ctx.fillStyle = isDark ? "#ccc" : "#333"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    const xAxisLabel = (plotMode === "Gr" || plotMode === "gr") ? "r (Å)" : "k (Å⁻¹)";
    ctx.fillText(xAxisLabel, mL + pw / 2, mT + ph + AXIS_TICK_PX + 26);
    ctx.save(); ctx.translate(14, mT + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.textBaseline = "top";
    const yAxisLabel = plotMode === "Ik" ? "I(k)" : plotMode === "Fk" ? "F(k)" : plotMode === "gr" ? "g(r)" : "G(r)";
    ctx.fillText(yAxisLabel, 0, 0); ctx.restore();
    // Clip
    ctx.save(); ctx.beginPath(); ctx.rect(mL, mT, pw, ph); ctx.clip();
    // Draw traces
    const drawLine = (xD: Float32Array | null, yD: Float32Array | null, color: string, dashed = false, lw = 1.5) => {
      if (!xD || !yD) return;
      ctx.strokeStyle = color; ctx.lineWidth = lw;
      if (dashed) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
      ctx.beginPath();
      let started = false;
      const len = Math.min(xD.length, yD.length);
      for (let i = 0; i < len; i++) {
        if (!isFinite(yD[i])) continue;
        const cx = d2cx(xD[i]), cy = d2cyData(yD[i]);
        if (!started) { ctx.moveTo(cx, cy); started = true; } else ctx.lineTo(cx, cy);
      }
      ctx.stroke(); ctx.setLineDash([]);
    };
    if (plotMode === "Ik") {
      drawLine(ikXRef.current, ikYRef.current, TRACE_COLORS.ik);
      if (showBackground) drawLine(ikXRef.current, ikBgRef.current, TRACE_COLORS.bg, true);
    } else if (plotMode === "Fk") {
      drawLine(fkXRef.current, fkYRef.current, TRACE_COLORS.fk);
    } else if (plotMode === "gr") {
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)";
      ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      const oneY = d2cy(1); ctx.beginPath(); ctx.moveTo(mL, oneY); ctx.lineTo(mL + pw, oneY); ctx.stroke(); ctx.setLineDash([]);
      drawLine(pdfXRef.current, pdfYRef.current, TRACE_COLORS.pdf);
    } else {
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)";
      ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      const zy = d2cy(0); ctx.beginPath(); ctx.moveTo(mL, zy); ctx.lineTo(mL + pw, zy); ctx.stroke(); ctx.setLineDash([]);
      drawLine(grXRef.current, grYRef.current, TRACE_COLORS.gr);
    }
    ctx.restore();
    // Crosshair
    if (cursorData) {
      const cx = d2cx(cursorData.x), cy = d2cy(cursorData.y);
      if (cx >= mL && cx <= mL + pw && cy >= mT && cy <= mT + ph) {
        ctx.strokeStyle = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.25)";
        ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(snap(cx), mT); ctx.lineTo(snap(cx), mT + ph); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(mL, snap(cy)); ctx.lineTo(mL + pw, snap(cy)); ctx.stroke();
        ctx.setLineDash([]);
        const displayY = useLogY ? Math.pow(10, cursorData.y) : cursorData.y;
        const label = `${formatNumber(cursorData.x)}, ${formatNumber(displayY)}`;
        ctx.font = `12px ${MONO}`;
        const tw = ctx.measureText(label).width + 8;
        let bx = cx + 10; if (bx + tw > mL + pw) bx = cx - tw - 10;
        let by = cy - 22; if (by < mT) by = cy + 10;
        ctx.fillStyle = isDark ? "rgba(30,30,30,0.9)" : "rgba(255,255,255,0.9)";
        ctx.fillRect(bx, by, tw, 18);
        ctx.strokeStyle = isDark ? "#555" : "#ccc"; ctx.lineWidth = 1; ctx.strokeRect(bx, by, tw, 18);
        ctx.fillStyle = isDark ? "#eee" : "#333"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(label, bx + 4, by + 9);
      }
    }
    if (computing) {
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.3)" : "rgba(255,255,255,0.3)";
      ctx.fillRect(mL, mT, pw, ph);
    }
  }, [PLOT_W, PLOT_H, plotXMin, plotXMax, plotYMin, plotYMax, plotMode, showBackground, ikLogScale, cursorData, computing, isDark,
      ikXBytes, ikYBytes, ikBgYBytes, fkXBytes, fkYBytes, grXBytes, grYBytes]);

  // =========================================================================
  // Nav mouse handlers
  // =========================================================================
  function screenToImage(e: React.MouseEvent) {
    const cvs = navCanvasRef.current;
    if (!cvs) return { row: 0, col: 0 };
    const rect = cvs.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const cx = NAV_SIZE / 2, cy = navH / 2;
    const imgX = (sx - cx - navPanX) / navZoom + cx;
    const imgY = (sy - cy - navPanY) / navZoom + cy;
    return { row: Math.round((imgY / navH) * scanRows), col: Math.round((imgX / NAV_SIZE) * scanCols) };
  }

  const maskVersionRef = React.useRef(maskVersion ?? 0);
  function syncMaskToPython() {
    if (!maskRef.current) return;
    const copy = maskRef.current.slice();
    // Encode mask as base64 string (Bytes traits don't sync JS→Python in anywidget)
    let binary = "";
    for (let i = 0; i < copy.length; i++) binary += String.fromCharCode(copy[i]);
    model.set("mask_b64", btoa(binary));
    model.save_changes();
    // Increment mask_version to trigger Python observer
    maskVersionRef.current += 1;
    setMaskVersion_model(maskVersionRef.current);
  }

  const handleNavMouseDown = (e: React.MouseEvent) => {
    if (!maskRef.current) return;
    e.preventDefault();
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      isPanningRef.current = true;
      panStartRef.current = { x: e.clientX, y: e.clientY, px: navPanX, py: navPanY };
      return;
    }
    const { row, col } = screenToImage(e);
    // "+" adds to the mask (excludes the area from analysis = 0);
    // "-" removes from the mask (re-includes the area = 1).
    const value = maskAction === "add" ? 0 : 1;
    if (maskTool === "freeform") {
      isPaintingRef.current = true;
      const half = maskBrushSize - 1;
      fillRectMask(maskRef.current, scanCols, scanRows, row - half, col - half, row + half, col + half, value);
      setMaskRenderVersion((v) => v + 1);
    } else {
      isDraggingShapeRef.current = true;
      shapeStartRef.current = { row, col };
      setShapePreview({ r0: row, c0: col, r1: row, c1: col });
    }
  };

  const handleNavMouseMove = (e: React.MouseEvent) => {
    if (isPanningRef.current && panStartRef.current) {
      setNavPanX(panStartRef.current.px + e.clientX - panStartRef.current.x);
      setNavPanY(panStartRef.current.py + e.clientY - panStartRef.current.y);
      return;
    }
    const { row, col } = screenToImage(e);
    if (isPaintingRef.current && maskRef.current) {
      const half = maskBrushSize - 1;
      fillRectMask(maskRef.current, scanCols, scanRows, row - half, col - half, row + half, col + half, maskAction === "add" ? 0 : 1);
      setMaskRenderVersion((v) => v + 1);
    }
    if (isDraggingShapeRef.current && shapeStartRef.current)
      setShapePreview({ r0: shapeStartRef.current.row, c0: shapeStartRef.current.col, r1: row, c1: col });
  };

  const handleNavMouseUp = () => {
    if (isPanningRef.current) { isPanningRef.current = false; panStartRef.current = null; return; }
    if (isPaintingRef.current) { isPaintingRef.current = false; syncMaskToPython(); return; }
    if (isDraggingShapeRef.current && shapeStartRef.current && maskRef.current && shapePreview) {
      const value = maskAction === "add" ? 0 : 1;
      const { r0, c0, r1, c1 } = shapePreview;
      if (maskTool === "circle") {
        fillCircleMask(maskRef.current, scanCols, scanRows, (r0 + r1) / 2, (c0 + c1) / 2, Math.max(Math.abs(c1 - c0), Math.abs(r1 - r0)) / 2, value);
      } else {
        fillRectMask(maskRef.current, scanCols, scanRows, r0, c0, r1, c1, value);
      }
      setMaskRenderVersion((v) => v + 1);
      syncMaskToPython();
    }
    isDraggingShapeRef.current = false; shapeStartRef.current = null; setShapePreview(null);
  };

  const handleNavWheel = (e: React.WheelEvent) => {
    e.preventDefault(); e.stopPropagation();
    setNavZoom((z) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * (e.deltaY > 0 ? 0.9 : 1.1))));
  };

  // =========================================================================
  // Plot mouse handlers
  // =========================================================================
  function plotScreenToData(e: React.MouseEvent) {
    const cvs = plotCanvasRef.current;
    if (!cvs) return null;
    const rect = cvs.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const pw = PLOT_W - MARGIN_LEFT_MIN - MARGIN_RIGHT;
    const ph = PLOT_H - MARGIN_TOP - MARGIN_BOTTOM;
    return {
      x: plotXMin + ((sx - MARGIN_LEFT_MIN) / pw) * (plotXMax - plotXMin),
      y: plotYMin + ((MARGIN_TOP + ph - sy) / ph) * (plotYMax - plotYMin),
    };
  }

  const handlePlotMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    isPlotPanRef.current = true;
    plotPanStartRef.current = { mx: e.clientX, my: e.clientY, xMin: plotXMin, xMax: plotXMax, yMin: plotYMin, yMax: plotYMax };
  };

  const handlePlotMouseMove = (e: React.MouseEvent) => {
    if (isPlotPanRef.current && plotPanStartRef.current) {
      const dx = e.clientX - plotPanStartRef.current.mx;
      const dy = e.clientY - plotPanStartRef.current.my;
      const pw = PLOT_W - MARGIN_LEFT_MIN - MARGIN_RIGHT;
      const ph = PLOT_H - MARGIN_TOP - MARGIN_BOTTOM;
      const dxD = -(dx / pw) * (plotPanStartRef.current.xMax - plotPanStartRef.current.xMin);
      const dyD = (dy / ph) * (plotPanStartRef.current.yMax - plotPanStartRef.current.yMin);
      setPlotXMin(plotPanStartRef.current.xMin + dxD); setPlotXMax(plotPanStartRef.current.xMax + dxD);
      setPlotYMin(plotPanStartRef.current.yMin + dyD); setPlotYMax(plotPanStartRef.current.yMax + dyD);
      userZoomedRef.current = true;
      return;
    }
    const d = plotScreenToData(e);
    if (d) setCursorData(d);
  };

  const handlePlotMouseUp = () => { isPlotPanRef.current = false; plotPanStartRef.current = null; };

  const handlePlotWheel = (e: React.WheelEvent) => {
    e.preventDefault(); e.stopPropagation();
    const d = plotScreenToData(e);
    if (!d) return;
    const f = e.deltaY > 0 ? 1.1 : 1 / 1.1;
    setPlotXMin((p) => d.x - (d.x - p) * f); setPlotXMax((p) => d.x + (p - d.x) * f);
    setPlotYMin((p) => d.y - (d.y - p) * f); setPlotYMax((p) => d.y + (p - d.y) * f);
    userZoomedRef.current = true;
  };

  // =========================================================================
  // Keyboard
  // =========================================================================
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
    switch (e.key.toLowerCase()) {
      case "r": setNavZoom(1); setNavPanX(0); setNavPanY(0); userZoomedRef.current = false; autoFitPlot(); break;
      case "x": setMaskAction((a) => {
          if (maskRef.current) {
            maskRef.current.fill(a === "add" ? 0 : 1);
            setMaskRenderVersion((v) => v + 1);
            syncMaskToPython();
          }
          return a === "add" ? "subtract" : "add";
        }); break;
      case "c":
        if (maskRef.current) { maskRef.current.fill(1); setMaskRenderVersion((v) => v + 1); syncMaskToPython(); }
        break;
      case "i":
        if (maskRef.current) { for (let i = 0; i < maskRef.current.length; i++) maskRef.current[i] = maskRef.current[i] ? 0 : 1; setMaskRenderVersion((v) => v + 1); syncMaskToPython(); }
        break;
      case "1": setPlotMode("Ik"); break;
      case "2": setPlotMode("Fk"); break;
      case "3": setPlotMode("Gr"); break;
    }
  };

  // Prevent scroll
  React.useEffect(() => {
    const h = (e: WheelEvent) => e.preventDefault();
    const opts: AddEventListenerOptions = { passive: false };
    const n1 = navCanvasRef.current?.parentElement;
    const n2 = plotCanvasRef.current?.parentElement;
    n1?.addEventListener("wheel", h, opts);
    n2?.addEventListener("wheel", h, opts);
    return () => { n1?.removeEventListener("wheel", h); n2?.removeEventListener("wheel", h); };
  }, []);

  // =========================================================================
  // Styles
  // =========================================================================
  const typo = {
    title: { fontSize: 15, fontWeight: 600, color: colors.accent, fontFamily: FONT },
    label: { fontSize: 13, color: colors.text, fontFamily: FONT },
    labelSmall: { fontSize: 12, fontWeight: 600, color: colors.textMuted, fontFamily: FONT },
    value: { fontSize: 12, fontFamily: MONO, color: colors.text },
  };
  const switchSmall = { "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-switchBase": { padding: "4px" } };
  const compactBtn = { fontSize: 12, minWidth: 36, px: 1, py: 0.25, textTransform: "none" as const };
  const imageBox = { position: "relative" as const, border: `1px solid ${colors.border}`, overflow: "hidden", bgcolor: isDark ? "#111" : "#eee" };

  // =========================================================================
  // JSX
  // =========================================================================
  return (
    <Box tabIndex={0} onKeyDown={handleKeyDown} sx={{ fontFamily: FONT, outline: "none", p: 1 }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: `${SPACING.SM}px` }}>
        <Typography sx={typo.title}>{title || "PDF"}</Typography>
        <Stack direction="row" alignItems="center" gap={1}>
          {computing && <CircularProgress size={14} sx={{ color: colors.accent }} />}
          {statusMessage && <Typography sx={{ ...typo.labelSmall, color: computing ? colors.accent : colors.textMuted }}>{statusMessage}</Typography>}
        </Stack>
      </Stack>

      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* LEFT: Nav + Mask */}
        <Box sx={{ width: NAV_SIZE }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 32, flexWrap: "wrap", rowGap: `${SPACING.XS}px` }}>
            <span style={{ fontSize: 12, fontFamily: FONT, color: colors.textMuted }}>Scan ({scanRows}×{scanCols})</span>
            <Stack direction="row" spacing={`2px`} alignItems="center">
              {(["rectangle", "circle", "freeform"] as const).map((tool) => (
                <Button key={tool} size="small" variant={maskTool === tool ? "contained" : "outlined"}
                  sx={{ ...compactBtn, minWidth: 24 }} onClick={() => setMaskTool(tool)}>
                  {tool === "rectangle" ? "▭" : tool === "circle" ? "○" : "✎"}
                </Button>
              ))}
              <Button size="small" variant={maskAction === "add" ? "contained" : "outlined"}
                sx={{ ...compactBtn, minWidth: 24 }} color={maskAction === "add" ? "primary" : "error"}
                onClick={() => setMaskAction((a) => {
                  if (!maskRef.current) return a === "add" ? "subtract" : "add";
                  if (a === "add") {
                    // Switching to subtract: drawing now re-includes — start with everything excluded
                    maskRef.current.fill(0);
                  } else {
                    // Switching to add: drawing now excludes — start with everything included
                    maskRef.current.fill(1);
                  }
                  setMaskRenderVersion((v) => v + 1);
                  syncMaskToPython();
                  return a === "add" ? "subtract" : "add";
                })}>
                {maskAction === "add" ? "+" : "−"}
              </Button>
              <Button size="small" sx={compactBtn} onClick={() => { if (maskRef.current) { maskRef.current.fill(1); setMaskRenderVersion((v) => v + 1); syncMaskToPython(); } }}>Clear</Button>
              <Button size="small" sx={compactBtn} onClick={() => { if (maskRef.current) { for (let i = 0; i < maskRef.current.length; i++) maskRef.current[i] = maskRef.current[i] ? 0 : 1; setMaskRenderVersion((v) => v + 1); syncMaskToPython(); } }}>Invert</Button>
            </Stack>
          </Stack>
          <Box sx={imageBox} style={{ width: NAV_SIZE, height: navH }}>
            <canvas ref={navCanvasRef} style={{ width: NAV_SIZE, height: navH, display: "block" }} />
            <canvas ref={navOverlayRef} style={{ position: "absolute", top: 0, left: 0, width: NAV_SIZE, height: navH, pointerEvents: "none" }} />
            <div style={{ position: "absolute", inset: 0, cursor: maskTool === "freeform" ? "crosshair" : "default" }}
              onMouseDown={handleNavMouseDown} onMouseMove={handleNavMouseMove} onMouseUp={handleNavMouseUp}
              onMouseLeave={() => { isPaintingRef.current = false; isDraggingShapeRef.current = false; setShapePreview(null); }}
              onWheel={handleNavWheel} />
          </Box>
          {showStats && <div style={{ fontSize: 12, fontFamily: MONO, color: colors.textMuted, marginTop: SPACING.XS }}>{maskPixelCount} / {scanRows * scanCols} included ({(maskFraction * 100).toFixed(1)}%)</div>}
          {showControls && (
            <Box sx={{ mt: `${SPACING.SM}px` }}>
              {maskTool === "freeform" && (
                <Stack direction="row" alignItems="center" gap={1} sx={{ mb: `${SPACING.XS}px` }}>
                  <Typography sx={typo.labelSmall}>Brush:</Typography>
                  <Slider value={maskBrushSize} onChange={(_, v) => setMaskBrushSize(v as number)} min={1} max={20} size="small"
                    sx={{ width: 80, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                  <Typography sx={typo.value}>{maskBrushSize}</Typography>
                </Stack>
              )}
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={typo.labelSmall}>Cmap:</Typography>
                <Select value={cmap} onChange={(e) => setCmap(e.target.value)} size="small" sx={{ fontSize: 12, height: 28, minWidth: 80 }}>
                  {CMAP_OPTIONS.map((name) => <MenuItem key={name} value={name} sx={{ fontSize: 12 }}>{name}</MenuItem>)}
                </Select>
              </Stack>
            </Box>
          )}
        </Box>

        {/* RIGHT: Curves */}
        <Box sx={{ width: PLOT_W, flexShrink: 0 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, minHeight: 32, flexWrap: "wrap", rowGap: `${SPACING.XS}px` }}>
            <Stack direction="row" spacing={`${SPACING.XS}px`}>
              {([["Ik", "I(k)"], ["Fk", "F(k)"], ["Gr", "G(r)"], ["gr", "g(r)"]] as const).map(([mode, label]) => (
                <Button key={mode} size="small" variant={plotMode === mode ? "contained" : "outlined"} sx={compactBtn}
                  onClick={() => setPlotMode(mode)}>{label}</Button>
              ))}
            </Stack>
            <Stack direction="row" alignItems="center" gap={1}>
              {plotMode === "Ik" && (<><Typography sx={typo.labelSmall}>Log:</Typography>
                <Switch checked={ikLogScale} onChange={(e) => { setIkLogScale(e.target.checked); userZoomedRef.current = false; setTimeout(autoFitPlot, 0); }} size="small" sx={switchSmall} />
                <Typography sx={typo.labelSmall}>Background:</Typography>
                <Switch checked={showBackground} onChange={(e) => setShowBackground(e.target.checked)} size="small" sx={switchSmall} /></>)}
              <Button size="small" sx={compactBtn} onClick={() => { userZoomedRef.current = false; autoFitPlot(); }}>Reset</Button>
            </Stack>
          </Stack>
          <Box sx={imageBox} style={{ width: PLOT_W, height: PLOT_H }}>
            <canvas ref={plotCanvasRef} style={{ width: PLOT_W, height: PLOT_H, display: "block" }}
              onMouseDown={handlePlotMouseDown} onMouseMove={handlePlotMouseMove} onMouseUp={handlePlotMouseUp}
              onMouseLeave={() => { isPlotPanRef.current = false; setCursorData(null); }}
              onDoubleClick={() => { userZoomedRef.current = false; autoFitPlot(); }} onWheel={handlePlotWheel} />
          </Box>
          {showStats && cursorData && <Typography sx={{ ...typo.value, mt: `${SPACING.XS}px` }}>
            {(plotMode === "Gr" || plotMode === "gr") ? "r" : "k"} = {formatNumber(cursorData.x, 4)}, {plotMode === "Ik" ? "I" : plotMode === "Fk" ? "F" : plotMode === "gr" ? "g" : "G"} = {formatNumber(plotMode === "Ik" && ikLogScale ? Math.pow(10, cursorData.y) : cursorData.y, 4)}
          </Typography>}
          {showControls && (
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>k fit:</Typography>
                <Slider value={localKFit} onChange={(_, v) => setLocalKFit(v as [number, number])}
                  onChangeCommitted={(_, v) => { const val = v as [number, number]; setKMinFit(val[0]); setKMaxFit(val[1]); }}
                  min={kMinAvail} max={kMaxAvail} step={0.01} size="small"
                  sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                <Typography sx={{ ...typo.value, color: colors.textMuted, minWidth: 90 }}>[{localKFit[0].toFixed(2)}, {localKFit[1].toFixed(2)}]</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>k window:</Typography>
                <Slider value={localKWin} onChange={(_, v) => setLocalKWin(v as [number, number])}
                  onChangeCommitted={(_, v) => { const val = v as [number, number]; setKMinWindow(val[0]); setKMaxWindow(val[1]); }}
                  min={kMinAvail} max={kMaxAvail} step={0.01} size="small"
                  sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                <Typography sx={{ ...typo.value, color: colors.textMuted, minWidth: 90 }}>[{localKWin[0].toFixed(2)}, {localKWin[1].toFixed(2)}]</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>r max:</Typography>
                <Slider value={localRMax} onChange={(_, v) => setLocalRMax(v as number)}
                  onChangeCommitted={(_, v) => setRMax(v as number)} min={1} max={50} step={0.5} size="small"
                  sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                <Typography sx={{ ...typo.value, color: colors.textMuted, minWidth: 50 }}>{localRMax.toFixed(1)} Å</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>k lowpass:</Typography>
                <Slider value={localKLowpass} onChange={(_, v) => setLocalKLowpass(v as number)}
                  onChangeCommitted={(_, v) => setKLowpass(v as number)} min={0} max={0.1} step={0.001} size="small"
                  sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                <Typography sx={{ ...typo.value, color: colors.textMuted, minWidth: 50 }}>{localKLowpass > 0 ? localKLowpass.toFixed(3) : "off"}</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>k highpass:</Typography>
                <Slider value={localKHighpass} onChange={(_, v) => setLocalKHighpass(v as number)}
                  onChangeCommitted={(_, v) => setKHighpass(v as number)} min={0} max={0.1} step={0.001} size="small"
                  sx={{ flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                <Typography sx={{ ...typo.value, color: colors.textMuted, minWidth: 50 }}>{localKHighpass > 0 ? localKHighpass.toFixed(3) : "off"}</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={typo.labelSmall}>Damp:</Typography>
                <Switch checked={dampOrigin} onChange={(e) => setDampOrigin(e.target.checked)} size="small" sx={switchSmall} />
                {dampOrigin && (<>
                  <Typography sx={{ ...typo.labelSmall, ml: 1 }}>r_cut:</Typography>
                  <Slider value={localRCut} onChange={(_, v) => setLocalRCut(v as number)}
                    onChangeCommitted={(_, v) => setRCut(v as number)} min={0.1} max={5} step={0.1} size="small"
                    sx={{ width: 80, "& .MuiSlider-thumb": { width: 10, height: 10 } }} />
                  <Typography sx={{ ...typo.value, color: colors.textMuted }}>{localRCut.toFixed(1)} Å</Typography>
                </>)}
              </Stack>
              {plotMode === "gr" && (
                <Stack direction="row" alignItems="center" gap={0}>
                  <Typography sx={{ ...typo.labelSmall, minWidth: 55 }}>Density:</Typography>
                  <Typography sx={typo.labelSmall}>Estimated</Typography>
                  <Switch
                    checked={densityMode === "manual"}
                    onChange={(e) => setDensityMode(e.target.checked ? "manual" : "estimated")}
                    size="small"
                    sx={switchSmall}
                  />
                  <Typography sx={typo.labelSmall}>Manual</Typography>
                  <input
                    type="number"
                    step="0.001"
                    min={0}
                    value={localDensity}
                    disabled={densityMode === "estimated"}
                    onChange={(e) => setLocalDensity(e.target.value)}
                    onBlur={() => {
                      const v = parseFloat(localDensity);
                      if (!isNaN(v) && v > 0 && v !== densityValue) setDensityValue(v);
                      else setLocalDensity(densityValue.toPrecision(4));
                    }}
                    onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                    style={{
                      width: 90, fontSize: 12, padding: "3px 6px", marginLeft: SPACING.LG, fontFamily: MONO,
                      border: `1px solid ${colors.border}`,
                      background: densityMode === "estimated" ? (isDark ? "#222" : "#f0f0f0") : (isDark ? "#1a1a1a" : "#fff"),
                      color: colors.textMuted,
                      outline: "none",
                    }}
                  />
                  <Typography sx={{ ...typo.labelSmall, ml: `${SPACING.XS}px` }}>Å⁻³</Typography>
                </Stack>
              )}
            </Box>
          )}
        </Box>
      </Stack>
    </Box>
  );
}

export const render = createRender(ShowPDF4DWidget);

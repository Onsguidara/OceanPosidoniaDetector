import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Posidonia Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --ocean-dark:   #020c14;
    --ocean-mid:    #062333;
    --ocean-teal:   #0d4f6e;
    --ocean-cyan:   #0aa3b5;
    --seagrass:     #2ec27e;
    --seagrass-dim: #1a7a4e;
    --sand:         #e8c87a;
    --text-primary: #e8f4f8;
    --text-muted:   #7eafc2;
    --border:       rgba(13,79,110,0.6);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--ocean-dark);
    color: var(--text-primary);
}

#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] {
    background: var(--ocean-mid) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Mono', monospace;
    color: var(--ocean-cyan);
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

.hero-header {
    background: linear-gradient(135deg, var(--ocean-mid) 0%, #041822 60%, #051e2e 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(10,163,181,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(46,194,126,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
    margin: 0 0 0.4rem;
}
.hero-title span { color: var(--ocean-cyan); }
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    background: rgba(46,194,126,0.15);
    border: 1px solid var(--seagrass-dim);
    color: var(--seagrass);
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ocean-cyan);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: var(--ocean-mid);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--ocean-cyan); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-color, var(--ocean-cyan));
}
.metric-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.pipeline {
    display: flex;
    gap: 0;
    margin: 1.5rem 0;
    overflow-x: auto;
}
.pipe-step {
    flex: 1;
    min-width: 100px;
    background: var(--ocean-mid);
    border: 1px solid var(--border);
    padding: 0.8rem 0.6rem;
    text-align: center;
    position: relative;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
.pipe-step:first-child { border-radius: 8px 0 0 8px; }
.pipe-step:last-child  { border-radius: 0 8px 8px 0; }
.pipe-step.active {
    background: rgba(10,163,181,0.15);
    border-color: var(--ocean-cyan);
    color: var(--ocean-cyan);
}
.pipe-step.done {
    background: rgba(46,194,126,0.1);
    border-color: var(--seagrass-dim);
    color: var(--seagrass);
}
.pipe-num {
    display: block;
    font-size: 1.1rem;
    margin-bottom: 0.25rem;
}

.info-box {
    background: rgba(10,163,181,0.08);
    border: 1px solid rgba(10,163,181,0.3);
    border-left: 3px solid var(--ocean-cyan);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: var(--text-muted);
    line-height: 1.6;
}
.info-box strong { color: var(--text-primary); }

.formula-box {
    background: var(--ocean-mid);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.4rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--sand);
    margin: 0.5rem 0;
}

.upload-zone {
    border: 2px dashed var(--ocean-teal);
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    background: rgba(6,35,51,0.5);
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--ocean-cyan); }

.result-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
}
.result-row:last-child { border-bottom: none; }
.color-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}
.result-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
}
.result-bar { height: 100%; border-radius: 3px; }

.stSlider label { color: var(--text-muted) !important; font-size: 0.8rem !important; }
[data-testid="stSlider"] > div > div > div { background: var(--ocean-teal) !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--ocean-teal), var(--ocean-cyan)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.8rem 1.5rem !important;
    text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    color: var(--ocean-cyan) !important;
    border-bottom: 2px solid var(--ocean-cyan) !important;
}

.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(46,194,126,0.12);
    border: 1px solid var(--seagrass-dim);
    color: var(--seagrass);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────────────────────
EPS = 1e-8

# ── Helper functions ────────────────────────────────────────────────────────────
def stretch(arr):
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + EPS), 0, 1)

def make_rgb(r, g, b):
    return np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)

def to_uint8(arr):
    norm = (arr - arr.min()) / (arr.max() - arr.min() + EPS)
    return (norm * 255).astype(np.uint8)

def compute_ndvi(nir, red):
    return (nir - red) / (nir + red + EPS)

def compute_ndwi(green, nir):
    return (green - nir) / (green + nir + EPS)

def dark_figure(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#020c14')
    ax.set_facecolor('#020c14')
    for spine in ax.spines.values():
        spine.set_edgecolor('#0d4f6e')
    ax.tick_params(colors='#7eafc2')
    return fig, ax

def dark_figures(rows, cols, figsize):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor('#020c14')
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]
    for ax in axes_flat:
        ax.set_facecolor('#020c14')
        for spine in ax.spines.values():
            spine.set_edgecolor('#0d4f6e')
        ax.tick_params(colors='#7eafc2')
    return fig, axes

# ── Demo data generator ─────────────────────────────────────────────────────────
@st.cache_data
def generate_demo_data(seed=42):
    rng = np.random.default_rng(seed)
    H, W = 256, 256
    blue  = np.zeros((H, W), dtype=np.float32)
    green = np.zeros((H, W), dtype=np.float32)
    red   = np.zeros((H, W), dtype=np.float32)
    nir   = np.zeros((H, W), dtype=np.float32)

    water_mask = np.zeros((H, W), bool)
    water_mask[80:, :180] = True
    water_mask[120:, :] = True

    pos_mask = np.zeros((H, W), bool)
    for (cy, cx, r) in [(140, 60, 25), (155, 110, 18), (135, 150, 20),
                         (170, 80, 15), (160, 40, 12), (145, 170, 10)]:
        Y, X = np.ogrid[:H, :W]
        pos_mask |= (((Y-cy)**2 + (X-cx)**2) < r**2)
    pos_mask &= water_mask
    land_mask = ~water_mask

    blue[land_mask]  = rng.uniform(0.10, 0.20, land_mask.sum())
    green[land_mask] = rng.uniform(0.12, 0.22, land_mask.sum())
    red[land_mask]   = rng.uniform(0.14, 0.28, land_mask.sum())
    nir[land_mask]   = rng.uniform(0.10, 0.18, land_mask.sum())

    blue[water_mask]  = rng.uniform(0.10, 0.20, water_mask.sum())
    green[water_mask] = rng.uniform(0.08, 0.15, water_mask.sum())
    red[water_mask]   = rng.uniform(0.02, 0.06, water_mask.sum())
    nir[water_mask]   = rng.uniform(0.01, 0.04, water_mask.sum())

    blue[pos_mask]  = rng.uniform(0.05, 0.10, pos_mask.sum())
    green[pos_mask] = rng.uniform(0.10, 0.18, pos_mask.sum())
    red[pos_mask]   = rng.uniform(0.04, 0.09, pos_mask.sum())
    nir[pos_mask]   = rng.uniform(0.14, 0.24, pos_mask.sum())

    for arr in [blue, green, red, nir]:
        arr[:] = cv2.GaussianBlur(arr, (7, 7), 2)

    return blue, green, red, nir

# ── Load SAFE folder ────────────────────────────────────────────────────────────
@st.cache_data
def load_safe(safe_path, scale=10):
    import rasterio
    from rasterio.enums import Resampling

    def find_band(folder, code):
        for f in os.listdir(folder):
            if f'_{code}_' in f and f.endswith('.jp2'):
                return os.path.join(folder, f)
        raise FileNotFoundError(f'Band {code} not found in {folder}')

    def load_band(path):
        with rasterio.open(path) as src:
            h, w = src.height // scale, src.width // scale
            band = src.read(1, out_shape=(h, w), resampling=Resampling.bilinear)
            return np.clip(band / 10000.0, 0, 1).astype(np.float32)

    granule = os.listdir(os.path.join(safe_path, 'GRANULE'))[0]
    img_folder = os.path.join(safe_path, 'GRANULE', granule, 'IMG_DATA', 'R10m')
    return (
        load_band(find_band(img_folder, 'B02')),
        load_band(find_band(img_folder, 'B03')),
        load_band(find_band(img_folder, 'B04')),
        load_band(find_band(img_folder, 'B08')),
    )

# ── Load precomputed NPZ ────────────────────────────────────────────────────────
@st.cache_data
def load_npz(npz_path):
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Posidonia Detector")
    st.markdown("---")

    st.markdown("## Data Source")
    data_mode = st.radio(
        "Input mode",
        ["📦 Precomputed (.npz)", "🧪 Demo (synthetic)", "📂 SAFE folder (local)"],
        label_visibility="collapsed"
    )

    safe_path = ""
    npz_path  = ""

    if "📦" in data_mode:
        st.markdown("""
        <div class="info-box" style="font-size:0.78rem;">
            Load results exported directly from your Colab notebook.
            Place <code>posidonia_results.npz</code> next to <code>app.py</code>,
            or paste an absolute path below.
        </div>
        """, unsafe_allow_html=True)
        app_dir = os.path.dirname(os.path.abspath(__file__))
        default_npz = os.path.join(app_dir, "posidonia_results.npz")
        npz_path = st.text_input(
            "Path to .npz file",
            value=default_npz if os.path.exists(default_npz) else "",
            placeholder="posidonia_results.npz"
        )

    elif "📂" in data_mode:
        safe_path = st.text_input(
            "SAFE folder path",
            placeholder="S2A_MSIL2A_....SAFE",
            help="Absolute path or place .SAFE next to app.py"
        )

    scale = st.slider("Downscale factor (SAFE only)", 5, 20, 10, 5,
                      help="Divide image size by this (10 → 100m/px)")

    st.markdown("---")
    st.markdown("## Parameters")
    st.markdown("*Applied on top of loaded data for interactive exploration.*")

    st.markdown("**Filtering**")
    filter_type = st.selectbox(
        "Smoothing filter",
        ["Bilateral", "Gaussian", "Median"],
        label_visibility="collapsed"
    )
    sigma_color, sigma_space, gauss_sigma, median_k = 150, 150, 5, 9
    if filter_type == "Bilateral":
        sigma_color = st.slider("Sigma color", 50, 200, 150, 10)
        sigma_space = st.slider("Sigma space", 50, 200, 150, 10)
    elif filter_type == "Gaussian":
        gauss_sigma = st.slider("Sigma", 1, 10, 5, 1)
    else:
        median_k = st.slider("Kernel size", 3, 15, 9, 2)

    st.markdown("**Thresholds**")
    min_area    = st.slider("Min patch area (px²)", 5, 200, 30, 5,
                             help="Remove patches smaller than this")

    st.markdown("**Morphology kernel**")
    kernel_size = st.slider("Kernel size", 3, 11, 5, 2)

    st.markdown("---")
    run_btn = st.button("▶  RUN DETECTION", use_container_width=True)

# ── Hero header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-badge">🛰 Sentinel-2 · Mediterranean Sea</div>
  <div class="hero-title">Posidonia <span>Oceanica</span><br>Detection System</div>
  <p class="hero-subtitle">
    Automated mapping of protected seagrass meadows using spectral indices,
    bilateral filtering, Otsu thresholding, and morphological analysis — no deep learning.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Pipeline overview ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Processing Pipeline</div>', unsafe_allow_html=True)
st.markdown("""
<div class="pipeline">
  <div class="pipe-step done"><span class="pipe-num">①</span>Load<br>Bands</div>
  <div class="pipe-step done"><span class="pipe-num">②</span>NDVI/<br>NDWI</div>
  <div class="pipe-step done"><span class="pipe-num">③</span>Filter</div>
  <div class="pipe-step done"><span class="pipe-num">④</span>Threshold<br>+ ROI</div>
  <div class="pipe-step done"><span class="pipe-num">⑤</span>Morpho-<br>logy</div>
  <div class="pipe-step done"><span class="pipe-num">⑥</span>Edges &<br>Contours</div>
  <div class="pipe-step active"><span class="pipe-num">⑦</span>Classifi-<br>cation</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — three modes
# ══════════════════════════════════════════════════════════════════════════════

USE_PRECOMPUTED = False  # flag: True means we loaded all arrays from .npz

# ── Mode 1: Precomputed .npz (notebook results) ─────────────────────────────
if "📦" in data_mode:
    if npz_path and os.path.exists(npz_path):
        try:
            with st.spinner("Loading precomputed results from .npz…"):
                npz = load_npz(npz_path)

            # Validate that the essential keys are present
            required = ['blue','green','red','nir','ndvi','ndwi',
                        'ndvi_u8','ndwi_u8','ndvi_filt','ndwi_filt',
                        'water_mask','deep_mask','shallow_mask',
                        'posidonia_raw','posidonia_clean','posidonia_final',
                        'label_map','otsu_val','deep_thresh']
            missing = [k for k in required if k not in npz]
            if missing:
                st.error(f"Missing keys in .npz: {missing}. Re-export from your notebook.")
                st.stop()

            blue             = npz['blue']
            green            = npz['green']
            red              = npz['red']
            nir              = npz['nir']
            ndvi             = npz['ndvi']
            ndwi             = npz['ndwi']
            ndvi_u8          = npz['ndvi_u8'].astype(np.uint8)
            ndwi_u8          = npz['ndwi_u8'].astype(np.uint8)
            ndvi_filt        = npz['ndvi_filt'].astype(np.uint8)
            ndwi_filt        = npz['ndwi_filt'].astype(np.uint8)
            water_mask       = npz['water_mask'].astype(np.uint8)
            deep_mask        = npz['deep_mask'].astype(np.uint8)
            shallow_mask     = npz['shallow_mask'].astype(np.uint8)
            posidonia_raw    = npz['posidonia_raw'].astype(np.uint8)
            posidonia_clean  = npz['posidonia_clean'].astype(np.uint8)
            posidonia_final  = npz['posidonia_final'].astype(np.uint8)
            label_map        = npz['label_map'].astype(np.uint8)
            otsu_val         = float(npz['otsu_val'].flat[0])
            deep_thresh      = int(npz['deep_thresh'].flat[0])
            USE_PRECOMPUTED  = True

            st.sidebar.success(f"✅ Loaded .npz  —  shape: {red.shape}")

        except Exception as e:
            st.error(f"Failed to load .npz: {e}")
            st.stop()
    else:
        st.markdown("""
        <div class="info-box">
            <strong>How to export from your notebook:</strong><br><br>
            Add this cell at the end of your Colab notebook and run it:
            <pre style="background:#020c14;padding:0.8rem;border-radius:6px;margin-top:0.5rem;font-size:0.78rem;color:#e8c87a;">
import numpy as np
np.savez_compressed(
    "posidonia_results.npz",
    blue=blue, green=green, red=red, nir=nir,
    ndvi=ndvi, ndwi=ndwi,
    ndvi_u8=ndvi_u8, ndwi_u8=ndwi_u8,
    ndvi_filt=ndvi_bilateral,
    ndwi_filt=ndwi_filt,
    water_mask=water_mask,
    deep_mask=deep_mask,
    shallow_mask=shallow_mask,
    posidonia_raw=posidonia_raw,
    posidonia_clean=posidonia_clean,
    posidonia_final=posidonia_final,
    label_map=label_map,
    otsu_val=np.array([otsu_val]),
    deep_thresh=np.array([deep_thresh]),
)
print("Saved!")</pre>
            Then download the file and place it next to <code>app.py</code>.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

# ── Mode 2: SAFE folder ──────────────────────────────────────────────────────
elif "📂" in data_mode:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    auto_found = [d for d in os.listdir(app_dir)
                  if d.endswith('.SAFE') and os.path.isdir(os.path.join(app_dir, d))]
    if auto_found and not safe_path:
        safe_path = os.path.join(app_dir, auto_found[0])
        st.sidebar.success(f"Auto-detected: {auto_found[0]}")

    if safe_path:
        full_path = safe_path if os.path.isabs(safe_path) else os.path.join(app_dir, safe_path)
        if os.path.isdir(full_path):
            try:
                with st.spinner("Loading Sentinel-2 bands…"):
                    blue, green, red, nir = load_safe(full_path, scale)
                st.sidebar.success(f"✅ Loaded SAFE  —  shape: {red.shape}")
            except Exception as e:
                st.error(f"Failed to load SAFE: {e}")
                st.stop()
        else:
            st.markdown("""
            <div class="info-box">
                Put your <strong>.SAFE</strong> folder next to <code>app.py</code> — it will be
                detected automatically. Or paste an absolute path in the sidebar.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
    else:
        st.markdown("""
        <div class="info-box">
            Place your <strong>S2A_MSIL2A_....SAFE</strong> folder in the same directory as
            <code>app.py</code> — it will be picked up automatically on next run,
            or type the path in the sidebar.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

# ── Mode 3: Demo synthetic data ───────────────────────────────────────────────
else:
    blue, green, red, nir = generate_demo_data()

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — run on loaded bands (or use precomputed arrays)
# ══════════════════════════════════════════════════════════════════════════════

if not USE_PRECOMPUTED:
    # Recompute everything from raw bands
    ndvi    = compute_ndvi(nir, red)
    ndwi    = compute_ndwi(green, nir)
    ndvi_u8 = to_uint8(ndvi)
    ndwi_u8 = to_uint8(ndwi)

    if filter_type == "Bilateral":
        ndvi_filt = cv2.bilateralFilter(ndvi_u8, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        ndwi_filt = cv2.bilateralFilter(ndwi_u8, d=9, sigmaColor=sigma_color//2, sigmaSpace=sigma_space//2)
    elif filter_type == "Gaussian":
        ndvi_filt = cv2.GaussianBlur(ndvi_u8, (7, 7), gauss_sigma)
        ndwi_filt = cv2.GaussianBlur(ndwi_u8, (7, 7), gauss_sigma)
    else:
        ndvi_filt = cv2.medianBlur(ndvi_u8, median_k)
        ndwi_filt = cv2.medianBlur(ndwi_u8, median_k)

    otsu_val, water_mask = cv2.threshold(ndwi_filt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    deep_thresh = min(int(otsu_val) + 40, 230)
    _, deep_mask = cv2.threshold(ndwi_filt, deep_thresh, 255, cv2.THRESH_BINARY)
    shallow_mask = cv2.subtract(water_mask, deep_mask)

    ndvi_roi = cv2.bitwise_and(ndvi_filt, shallow_mask)
    _, veg_mask = cv2.threshold(ndvi_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    posidonia_raw = cv2.bitwise_and(veg_mask, shallow_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    pos_opened = cv2.morphologyEx(posidonia_raw, cv2.MORPH_OPEN, kernel)
    posidonia_clean = cv2.morphologyEx(pos_opened, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(posidonia_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    posidonia_final = np.zeros_like(posidonia_clean)
    cv2.drawContours(posidonia_final, valid_contours, -1, 255, -1)

    kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    water_clean = cv2.morphologyEx(
        cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel_lg),
        cv2.MORPH_CLOSE, kernel_lg
    )
    _, sand_raw = cv2.threshold(ndwi_filt, 100, 255, cv2.THRESH_BINARY_INV)
    sand_mask = cv2.bitwise_and(sand_raw, cv2.bitwise_not(water_mask))

    label_map = np.full(red.shape, 3, dtype=np.uint8)
    label_map[sand_mask > 0] = 2
    label_map[water_clean > 0] = 1
    label_map[posidonia_final > 0] = 0

else:
    # Precomputed: only recompute contours (can't be stored in .npz) and fast display items
    kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    water_clean = cv2.morphologyEx(
        cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel_lg),
        cv2.MORPH_CLOSE, kernel_lg
    )
    contours, _ = cv2.findContours(posidonia_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

# ── Items always computed (fast, display only) ───────────────────────────────
edges_canny = cv2.Canny(posidonia_clean, 50, 150)
sobel_x = cv2.Sobel(ndvi_filt.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(ndvi_filt.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = to_uint8(np.sqrt(sobel_x**2 + sobel_y**2))

CLASS_NAMES  = ['Posidonia (seagrass)', 'Water', 'Sand / Bare', 'Land / Other']
CLASS_COLORS = ['#2ec27e', '#1565c0', '#e8c87a', '#607d8b']

rgb = make_rgb(red, green, blue)
fcc = make_rgb(nir, red, green)
rgb_u8 = (rgb * 255).astype(np.uint8)
rgb_contours = rgb_u8.copy()
cv2.drawContours(rgb_contours, valid_contours, -1, (46, 194, 126), 2)

# ── Source badge ─────────────────────────────────────────────────────────────
if USE_PRECOMPUTED:
    src_label = "📦 Source: Notebook results (.npz) — exact match guaranteed"
elif "📂" in data_mode:
    src_label = "📂 Source: SAFE folder — reprocessed from raw bands"
else:
    src_label = "🧪 Source: Synthetic demo data"
st.markdown(f'<div class="source-badge">{src_label}</div>', unsafe_allow_html=True)

# ── Metrics row ───────────────────────────────────────────────────────────────
total_px = label_map.size
pos_px   = int((label_map == 0).sum())
water_px = int((label_map == 1).sum())
pos_pct  = 100 * pos_px / total_px
pixel_area_m2 = 100 * 100
pos_area_km2  = pos_px * pixel_area_m2 / 1e6

st.markdown('<div class="section-label">Detection Results</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card" style="--accent-color:#2ec27e">
    <div class="metric-icon">🌿</div>
    <div class="metric-value">{pos_px:,}</div>
    <div class="metric-label">Posidonia pixels</div>
  </div>
  <div class="metric-card" style="--accent-color:#0aa3b5">
    <div class="metric-icon">📐</div>
    <div class="metric-value">{pos_area_km2:.1f} km²</div>
    <div class="metric-label">Estimated area</div>
  </div>
  <div class="metric-card" style="--accent-color:#e8c87a">
    <div class="metric-icon">📊</div>
    <div class="metric-value">{pos_pct:.1f}%</div>
    <div class="metric-label">Of tile coverage</div>
  </div>
  <div class="metric-card" style="--accent-color:#7e57c2">
    <div class="metric-icon">🔷</div>
    <div class="metric-value">{len(valid_contours)}</div>
    <div class="metric-label">Valid patches</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛰 Imagery", "📈 Indices", "🔧 Filtering", "🧹 Morphology", "🗺 Classification"
])

# ── Tab 1: Imagery ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-label">Satellite Imagery</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = dark_figure((6, 5))
        ax.imshow(rgb)
        ax.set_title("True Color (RGB)", color='#e8f4f8', fontsize=11, pad=10)
        ax.axis('off')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with c2:
        fig, ax = dark_figure((6, 5))
        ax.imshow(fcc)
        ax.set_title("False Color (NIR/R/G) — vegetation = bright red",
                     color='#e8f4f8', fontsize=11, pad=10)
        ax.axis('off')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("""
    <div class="info-box">
        <strong>True Color</strong> shows what the human eye would see from space.
        <strong>False Color</strong> places NIR in the red channel — healthy vegetation
        (including Posidonia) appears bright red, while water remains dark.
    </div>
    """, unsafe_allow_html=True)

    # Transformations
    st.markdown('<div class="section-label">Image Transformations</div>', unsafe_allow_html=True)
    rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    H, W = rgb_u8.shape[:2]
    rgb_resized = cv2.resize(rgb_bgr, (W // 2, H // 2))
    rgb_rotated = cv2.rotate(rgb_bgr, cv2.ROTATE_90_CLOCKWISE)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = dark_figure((5, 4))
        ax.imshow(cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB))
        ax.set_title("Resized to 50%", color='#e8f4f8', fontsize=10, pad=8)
        ax.axis('off')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with c2:
        fig, ax = dark_figure((5, 4))
        ax.imshow(cv2.cvtColor(rgb_rotated, cv2.COLOR_BGR2RGB))
        ax.set_title("Rotated 90°", color='#e8f4f8', fontsize=10, pad=8)
        ax.axis('off')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── Tab 2: Indices ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">Spectral Indices</div>', unsafe_allow_html=True)

    col_f, col_n = st.columns(2)
    with col_f:
        st.markdown('<div class="formula-box">NDVI = (NIR − Red) / (NIR + Red)</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="font-size:0.82rem;">
        <strong>+1</strong> → dense vegetation (Posidonia!) &nbsp;
        <strong>0</strong> → bare soil / sand &nbsp;
        <strong>−1</strong> → water
        </div>""", unsafe_allow_html=True)
    with col_n:
        st.markdown('<div class="formula-box">NDWI = (Green − NIR) / (Green + NIR)</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="font-size:0.82rem;">
        <strong>&gt; 0</strong> → water &nbsp;
        <strong>&lt; 0</strong> → land or vegetation
        </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "NDVI min", f"{ndvi.min():.3f}"),
        (c2, "NDVI max", f"{ndvi.max():.3f}"),
        (c3, "NDWI min", f"{ndwi.min():.3f}"),
        (c4, "NDWI max", f"{ndwi.max():.3f}"),
    ]:
        col.metric(label, val)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = dark_figure((6, 5))
        im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title("NDVI — Vegetation Index", color='#e8f4f8', fontsize=11, pad=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with c2:
        fig, ax = dark_figure((6, 5))
        im = ax.imshow(ndwi, cmap='Blues_r', vmin=-1, vmax=1)
        ax.set_title("NDWI — Water Index", color='#e8f4f8', fontsize=11, pad=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── Tab 3: Filtering ──────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-label">Image Filtering Comparison</div>',
                unsafe_allow_html=True)

    ndvi_gaussian  = cv2.GaussianBlur(ndvi_u8, (7, 7), sigmaX=5)
    ndvi_median    = cv2.medianBlur(ndvi_u8, 9)
    ndvi_bilateral = cv2.bilateralFilter(ndvi_u8, d=9, sigmaColor=150, sigmaSpace=150)

    diff_g = cv2.absdiff(ndvi_u8, ndvi_gaussian)
    diff_m = cv2.absdiff(ndvi_u8, ndvi_median)
    diff_b = cv2.absdiff(ndvi_u8, ndvi_bilateral)

    fig, axes = dark_figures(2, 3, (18, 10))
    panels = [
        (ndvi_u8,        "Original NDVI",                   "RdYlGn"),
        (ndvi_gaussian,  "Gaussian Filter (k=7, σ=5)",      "RdYlGn"),
        (ndvi_bilateral, "Bilateral Filter ✓ (chosen)",     "RdYlGn"),
        (diff_g,         f"Diff vs Gaussian  μ={diff_g.mean():.2f}",  "hot"),
        (diff_m,         f"Diff vs Median    μ={diff_m.mean():.2f}",  "hot"),
        (diff_b,         f"Diff vs Bilateral μ={diff_b.mean():.2f}",  "hot"),
    ]
    for ax, (img, title, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color='#e8f4f8', fontsize=10, pad=8)
        ax.axis('off')
    fig.suptitle("Filter Comparison", color='#e8f4f8', fontsize=13, y=1.01)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class="info-box">
        <strong>Bilateral filtering</strong> was selected because it reduces noise while
        preserving important spatial edges (vegetation–water boundaries), unlike Gaussian
        filtering which blurs edges uniformly.
    </div>
    """, unsafe_allow_html=True)

# ── Tab 4: Morphology ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-label">Thresholding, ROI Masking & Morphological Cleaning</div>',
                unsafe_allow_html=True)

    fig, axes = dark_figures(2, 3, (18, 11))
    panels = [
        (rgb,             "True Color (reference)",              None),
        (water_mask,      f"Water Mask (Otsu={int(otsu_val)})",  "Blues"),
        (deep_mask,       f"Deep Water (thresh={deep_thresh})",  "Blues"),
        (shallow_mask,    "Shallow ROI — search zone",           "GnBu"),
        (posidonia_raw,   "Posidonia raw detection",             "Greens"),
        (posidonia_final, f"Final mask ({len(valid_contours)} patches)", "Greens"),
    ]
    for ax, (img, title, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color='#e8f4f8', fontsize=10, pad=8)
        ax.axis('off')
    fig.suptitle("ROI Masking & Morphological Cleaning", color='#e8f4f8', fontsize=13)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Before morphology", f"{int(posidonia_raw.sum()//255):,} px")
    c2.metric("After morphology",  f"{int(posidonia_clean.sum()//255):,} px")
    c3.metric("Noise removed",     f"{int(posidonia_raw.sum()//255) - int(posidonia_clean.sum()//255):,} px")

    # All morphological operations visualized
    st.markdown('<div class="section-label">Morphological Operations Detail</div>',
                unsafe_allow_html=True)
    kernel_display = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pos_eroded   = cv2.erode(posidonia_raw, kernel_display, iterations=1)
    pos_dilated  = cv2.dilate(posidonia_raw, kernel_display, iterations=1)
    pos_opened_d = cv2.morphologyEx(posidonia_raw, cv2.MORPH_OPEN, kernel_display)
    pos_closed_d = cv2.morphologyEx(posidonia_raw, cv2.MORPH_CLOSE, kernel_display)
    pos_gradient = cv2.morphologyEx(posidonia_raw, cv2.MORPH_GRADIENT, kernel_display)

    fig, axes = dark_figures(2, 3, (18, 11))
    morph_panels = [
        (posidonia_raw,  "Raw Mask (before cleaning)"),
        (pos_eroded,     "Erosion — shrinks regions"),
        (pos_dilated,    "Dilation — expands regions"),
        (pos_opened_d,   "Opening — removes noise"),
        (pos_closed_d,   "Closing — fills holes"),
        (pos_gradient,   "Gradient — shows edges"),
    ]
    for ax, (img, title) in zip(axes.flat, morph_panels):
        ax.imshow(img, cmap='Greens')
        ax.set_title(title, color='#e8f4f8', fontsize=10, pad=8)
        ax.axis('off')
    fig.suptitle("Morphological Operations on Posidonia Mask", color='#e8f4f8', fontsize=13)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Edge detection
    st.markdown('<div class="section-label">Edge Detection & Contour Analysis</div>',
                unsafe_allow_html=True)
    fig, axes = dark_figures(1, 3, (18, 5))
    axes[0].imshow(edges_canny, cmap='gray')
    axes[0].set_title("Canny Edges", color='#e8f4f8', fontsize=10)
    axes[1].imshow(sobel_mag, cmap='hot')
    axes[1].set_title("Sobel Magnitude", color='#e8f4f8', fontsize=10)
    axes[2].imshow(rgb_contours)
    axes[2].set_title("Detected Patches on Satellite Image", color='#e8f4f8', fontsize=10)
    for ax in axes: ax.axis('off')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Contour table
    if valid_contours:
        st.markdown('<div class="section-label">Patch Statistics</div>', unsafe_allow_html=True)
        rows = []
        for i, cnt in enumerate(valid_contours[:20]):
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, closed=True)
            if peri == 0:
                continue
            circ  = 4 * np.pi * area / (peri ** 2)
            shape = 'Round' if circ > 0.7 else ('Elongated' if circ < 0.3 else 'Mixed')
            rows.append({"ID": i, "Area (px²)": f"{area:.0f}",
                         "Perimeter": f"{peri:.1f}",
                         "Circularity": f"{circ:.3f}", "Shape": shape})
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ── Tab 5: Classification ─────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-label">Final 4-Class Classification Map</div>',
                unsafe_allow_html=True)

    cmap_cls = plt.matplotlib.colors.ListedColormap(CLASS_COLORS)
    legend_patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
                      for i in range(4)]

    fig, axes = dark_figures(1, 3, (22, 7))
    axes[0].imshow(rgb)
    axes[0].set_title("True Color", color='#e8f4f8', fontsize=11, pad=10)

    axes[1].imshow(label_map, cmap=cmap_cls, vmin=0, vmax=3)
    axes[1].set_title("Classification Map", color='#e8f4f8', fontsize=11, pad=10)
    axes[1].legend(handles=legend_patches, loc='lower right', fontsize=8,
                   facecolor='#020c14', edgecolor='#0d4f6e', labelcolor='#e8f4f8')

    axes[2].imshow(rgb)
    axes[2].imshow(label_map, cmap=cmap_cls, vmin=0, vmax=3, alpha=0.5)
    axes[2].set_title("Overlay", color='#e8f4f8', fontsize=11, pad=10)
    axes[2].legend(handles=legend_patches, loc='lower right', fontsize=8,
                   facecolor='#020c14', edgecolor='#0d4f6e', labelcolor='#e8f4f8')
    for ax in axes: ax.axis('off')
    fig.suptitle("Posidonia Oceanica — Final Classification", color='#e8f4f8', fontsize=13)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Coverage distribution bars
    st.markdown('<div class="section-label">Coverage Distribution</div>', unsafe_allow_html=True)
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        px  = int((label_map == i).sum())
        pct = 100 * px / total_px
        st.markdown(f"""
        <div class="result-row">
          <div class="color-dot" style="background:{color}"></div>
          <div style="min-width:160px;font-size:0.85rem;">{name}</div>
          <div class="result-bar-bg">
            <div class="result-bar" style="width:{pct:.1f}%;background:{color}"></div>
          </div>
          <div style="min-width:90px;text-align:right;font-family:'Space Mono',monospace;
                      font-size:0.8rem;color:#7eafc2;">
            {px:,} px &nbsp; <strong style="color:#e8f4f8">{pct:.1f}%</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:1.5rem;">
        <strong>Interpretation:</strong> Posidonia patches are detected where NDVI is elevated
        inside the shallow coastal water zone (ROI). Large, elongated patches typically indicate
        continuous meadows; small, isolated patches may be sparse coverage or noise.
        The shallow ROI constraint reduces false positives from terrestrial vegetation.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#3a6070;font-size:0.75rem;font-family:'Space Mono',monospace;
            padding:1rem 0;letter-spacing:0.08em;">
  POSIDONIA OCEANICA DETECTION · CLASSICAL CV · SENTINEL-2 · NO DEEP LEARNING
</div>
""", unsafe_allow_html=True)

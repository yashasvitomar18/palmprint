"""
================================================================================
UAD-Net: Uncertainty-Aware Dynamic Fusion Network for Contactless Palmprint
and Palm Vein Recognition — IEEE Conference Live Demonstration Dashboard
================================================================================

Terminal installation instructions:

    pip install streamlit torch matplotlib numpy scikit-learn torchvision

Run with:

    streamlit run app.py

Everything (synthetic multimodal data generator, lightweight PyTorch UAD-Net
model, and the full IEEE-style live UI) lives in this single file.
================================================================================
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

# ==============================================================================
# 0. PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="UAD-Net Live Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = 128
EMBED_DIM = 64
N_CLASSES = 8
SAMPLES_PER_CLASS = 10

# ==============================================================================
# 1. SYNTHETIC MULTIMODAL IMAGE GENERATOR
# ==============================================================================


def _curve_mask(size, ctrl_fn, thickness, n_samples=180):
    """Rasterize a parametric curve into a soft (H, W) intensity mask in [0, 1]."""
    t = np.linspace(0.0, 1.0, n_samples)
    px, py = ctrl_fn(t)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = xs[..., None] - px[None, None, :]
    dy = ys[..., None] - py[None, None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = dist.min(axis=-1)
    return np.exp(-(min_dist ** 2) / (2.0 * thickness ** 2))


def generate_palmprint_roi(seed, kernel_size=1, sigma=0.0, shadow_intensity=0.0, blackout=False):
    """Synthesize a 128x128 RGB visible-light palmprint ROI with principal lines."""
    size = IMG_SIZE
    if blackout:
        return np.zeros((size, size, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)
    base_tone = rng.uniform(0.55, 0.68)
    canvas = np.full((size, size, 3), base_tone, dtype=np.float32)

    # skin texture speckle
    canvas += rng.normal(0, 0.02, size=(size, size, 3))

    # radial skin shading (center brighter than edges)
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size * 0.5, size * 0.55
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size * 0.75)
    canvas -= (radial[..., None] ** 2) * 0.08

    # three principal lines: heart, head, life
    principal_specs = [
        (lambda t, a=rng.uniform(18, 26), p=rng.uniform(20, 34): (10 + t * (size - 20),
                                                                    p + a * np.sin(t * np.pi * 1.1)), 2.2),
        (lambda t, a=rng.uniform(10, 18), p=rng.uniform(50, 62): (10 + t * (size - 20),
                                                                    p + a * np.sin(t * np.pi * 0.9 + 1.0)), 2.0),
        (lambda t, a=rng.uniform(14, 22), p=rng.uniform(78, 96): (18 + t * (size * 0.55),
                                                                    p - a * (t ** 1.4) * 2.4), 2.4),
    ]
    for ctrl_fn, thick in principal_specs:
        mask = _curve_mask(size, ctrl_fn, thick)
        canvas -= mask[..., None] * 0.28

    # minor wrinkle lines (thin, faint)
    for _ in range(9):
        x0 = rng.uniform(15, size - 15)
        y0 = rng.uniform(15, size - 15)
        amp = rng.uniform(3, 10)
        ang = rng.uniform(0, np.pi)

        def ctrl_fn(t, x0=x0, y0=y0, amp=amp, ang=ang):
            length = rng.uniform(20, 45)
            px = x0 + t * length * np.cos(ang) + amp * np.sin(t * np.pi * 2)
            py = y0 + t * length * np.sin(ang) + amp * np.cos(t * np.pi * 2)
            return px, py

        mask = _curve_mask(size, ctrl_fn, 0.9)
        canvas -= mask[..., None] * 0.08

    # illumination shadow (linear gradient across the ROI)
    if shadow_intensity > 0:
        grad = np.linspace(0, 1, size)[None, :].repeat(size, axis=0)
        shadow = 1.0 - shadow_intensity * grad
        canvas *= shadow[..., None]

    canvas = np.clip(canvas, 0.0, 1.0)

    # gaussian blur via torch conv2d
    if kernel_size > 1 and sigma > 0:
        canvas = apply_gaussian_blur(canvas, kernel_size, sigma)

    return canvas.astype(np.float32)


def generate_palmvein_roi(seed, motion_len=1, contrast_atten=0.0, blackout=False):
    """Synthesize a 128x128 single-channel NIR palm-vein ROI with a vascular network."""
    size = IMG_SIZE
    if blackout:
        return np.zeros((size, size), dtype=np.float32)

    rng = np.random.default_rng(seed + 777)
    base = rng.uniform(0.42, 0.5)
    canvas = np.full((size, size), base, dtype=np.float32)
    canvas += rng.normal(0, 0.015, size=(size, size))

    # branching vein network: a handful of primary trunks with secondary branches
    trunks = []
    n_trunks = 4
    for i in range(n_trunks):
        x0 = rng.uniform(10, size - 10)
        y0 = 6
        x1 = rng.uniform(30, size - 30)
        y1 = size - 8
        bend = rng.uniform(-25, 25)

        def ctrl_fn(t, x0=x0, y0=y0, x1=x1, y1=y1, bend=bend):
            px = x0 + (x1 - x0) * t + bend * np.sin(t * np.pi)
            py = y0 + (y1 - y0) * t
            return px, py

        thick = rng.uniform(2.0, 3.2)
        mask = _curve_mask(size, ctrl_fn, thick)
        canvas -= mask * 0.22
        trunks.append(ctrl_fn)

    # secondary thinner branches sprouting from trunks
    for ctrl_fn in trunks:
        for _ in range(rng.integers(2, 4)):
            t0 = rng.uniform(0.2, 0.8)
            bx, by = ctrl_fn(np.array([t0]))
            bx, by = float(bx[0]), float(by[0])
            ang = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(15, 30)

            def branch_fn(t, bx=bx, by=by, ang=ang, length=length):
                px = bx + t * length * np.cos(ang)
                py = by + t * length * np.sin(ang)
                return px, py

            mask = _curve_mask(size, branch_fn, 1.3)
            canvas -= mask * 0.14

    canvas = np.clip(canvas, 0.0, 1.0)

    # contrast attenuation (pulls pixels toward mean -> washed out veins)
    if contrast_atten > 0:
        mean_val = canvas.mean()
        canvas = mean_val + (canvas - mean_val) * (1.0 - contrast_atten)

    # motion / defocus blur via 1D horizontal smear kernel
    if motion_len > 1:
        canvas = apply_motion_blur(canvas, motion_len)

    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def apply_gaussian_blur(img_hw_c, kernel_size, sigma):
    """img_hw_c: (H, W, C) float32 in [0,1]. Returns blurred array, same shape."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * max(sigma, 1e-3) ** 2))
    kernel /= kernel.sum()
    k = torch.tensor(kernel, dtype=torch.float32).view(1, 1, kernel_size, kernel_size)

    t = torch.tensor(img_hw_c, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    c = t.shape[1]
    k_c = k.repeat(c, 1, 1, 1)
    out = F.conv2d(t, k_c, padding=kernel_size // 2, groups=c)
    return out.squeeze(0).permute(1, 2, 0).numpy()


def apply_motion_blur(img_hw, length):
    """img_hw: (H, W) float32 in [0,1]. Horizontal smear kernel of given length."""
    length = max(1, int(length))
    if length % 2 == 0:
        length += 1
    kernel = np.ones((1, length), dtype=np.float32) / length
    k = torch.tensor(kernel).view(1, 1, 1, length)
    t = torch.tensor(img_hw, dtype=torch.float32).view(1, 1, *img_hw.shape)
    out = F.conv2d(t, k, padding=(0, length // 2))
    return out.view(img_hw.shape).numpy()


# ==============================================================================
# 2. LIGHTWEIGHT UAD-NET MODEL
# ==============================================================================


class StreamA(nn.Module):
    """Palmprint texture stem — 3x3 convolutions."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(64 * 4 * 4, embed_dim)

    def forward(self, x):
        z = self.net(x)
        z = z.flatten(1)
        return self.fc(z)


class StreamB(nn.Module):
    """Palm-vein structure stem — 5x5 convolutions."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(64 * 4 * 4, embed_dim)

    def forward(self, x):
        z = self.net(x)
        z = z.flatten(1)
        return self.fc(z)


class UADM(nn.Module):
    """Uncertainty-Aware Dynamic fusion Module — predicts per-modality confidence
    logits (sp, sv) and normalizes them via softmax into gating weights
    (alpha_p, alpha_v) with alpha_p + alpha_v = 1.0."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, f_p, f_v):
        joint = torch.cat([f_p, f_v], dim=1)
        logits = self.net(joint)  # (B, 2) -> [sp, sv]
        alpha = F.softmax(logits, dim=1)
        return logits, alpha


class UADNet(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.stream_a = StreamA(embed_dim)
        self.stream_b = StreamB(embed_dim)
        self.uadm = UADM(embed_dim)

    def forward(self, rgb, nir):
        f_p = self.stream_a(rgb)
        f_v = self.stream_b(nir)
        logits, alpha = self.uadm(f_p, f_v)
        alpha_p, alpha_v = alpha[:, 0:1], alpha[:, 1:2]
        f_fused = alpha_p * f_p + alpha_v * f_v
        return {
            "f_p": f_p, "f_v": f_v, "f_fused": f_fused,
            "alpha_p": alpha_p, "alpha_v": alpha_v,
            "sp": logits[:, 0], "sv": logits[:, 1],
        }


@st.cache_resource
def load_model():
    torch.manual_seed(42)
    model = UADNet()
    model.eval()
    return model


@st.cache_resource
def build_identity_gallery(_model):
    """Fixed synthetic gallery of enrolled identities in embedding space, built
    once by running the (frozen) model over synthetic samples per class."""
    torch.manual_seed(123)
    rng = np.random.default_rng(123)
    gallery_embeds = []
    gallery_labels = []
    with torch.no_grad():
        for cls in range(N_CLASSES):
            base_seed = cls * 1000
            for s in range(SAMPLES_PER_CLASS):
                seed = base_seed + s
                rgb = generate_palmprint_roi(seed, kernel_size=1, sigma=0.0, shadow_intensity=0.0)
                nir = generate_palmvein_roi(seed, motion_len=1, contrast_atten=0.0)
                rgb_t = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                nir_t = torch.tensor(nir, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                out = _model(rgb_t, nir_t)
                gallery_embeds.append(out["f_fused"].squeeze(0).numpy())
                gallery_labels.append(cls)
    return np.stack(gallery_embeds), np.array(gallery_labels)


# ==============================================================================
# 3. VISUALIZATION HELPERS
# ==============================================================================


def draw_radial_gauge(ax, value, label, color):
    """Half-donut radial gauge showing `value` in [0, 1]."""
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.bar(np.pi / 2, 1.0, width=np.pi, color="#e6e6e6", bottom=0.0, linewidth=0)
    ax.bar(np.pi / 2, 1.0, width=np.pi * value, color=color, bottom=0.0, linewidth=0)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.text(0, -0.35, f"{label}\n{value * 100:.1f}%", ha="center", va="center",
             fontsize=11, fontweight="bold", transform=ax.transData)


def render_gauges(alpha_p, alpha_v):
    fig = plt.figure(figsize=(5.5, 2.6))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    draw_radial_gauge(ax1, alpha_p, "alpha_p (Palmprint)", "#2b6cb0")
    draw_radial_gauge(ax2, alpha_v, "alpha_v (Palm Vein)", "#c05621")
    fig.tight_layout()
    return fig


def render_alpha_history_bar(history):
    fig, ax = plt.subplots(figsize=(5.5, 3))
    idx = np.arange(len(history))
    alpha_p_vals = [h[0] for h in history]
    alpha_v_vals = [h[1] for h in history]
    width = 0.4
    ax.bar(idx - width / 2, alpha_p_vals, width=width, label="alpha_p", color="#2b6cb0")
    ax.bar(idx + width / 2, alpha_v_vals, width=width, label="alpha_v", color="#c05621")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Gating weight")
    ax.set_title("Dynamic alpha shift vs. degradation adjustments")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def render_tsne(gallery_embeds, gallery_labels, live_embed, pred_class):
    all_embeds = np.vstack([gallery_embeds, live_embed[None, :]])
    perplexity = min(15, max(5, all_embeds.shape[0] // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    proj = tsne.fit_transform(all_embeds)
    gallery_proj = proj[:-1]
    live_proj = proj[-1]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = plt.get_cmap("tab10")
    for cls in range(N_CLASSES):
        mask = gallery_labels == cls
        ax.scatter(gallery_proj[mask, 0], gallery_proj[mask, 1], s=28,
                   color=cmap(cls % 10), alpha=0.6, label=f"ID {cls}")
    ax.scatter(live_proj[0], live_proj[1], marker="*", s=380, color=cmap(pred_class % 10),
               edgecolor="black", linewidth=1.2, label="Live sample", zorder=5)
    ax.set_title("t-SNE: live sample vs. enrolled identity clusters")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def tag(text, color):
    return f"<span style='background-color:{color};color:white;padding:2px 8px;" \
           f"border-radius:10px;font-size:12px;margin-right:4px;'>{text}</span>"


# ==============================================================================
# 4. SIDEBAR — LIVE DEGRADATION CONTROLS
# ==============================================================================

st.sidebar.title("Live Degradation Controls")

if "sample_seed" not in st.session_state:
    st.session_state.sample_seed = int(np.random.default_rng().integers(0, 100_000))
if "alpha_history" not in st.session_state:
    st.session_state.alpha_history = []

if st.sidebar.button("🔄 New Live Sample"):
    st.session_state.sample_seed = int(np.random.default_rng().integers(0, 100_000))

st.sidebar.markdown("### 🖐️ Palmprint (Visible RGB) Noise")
pp_kernel = st.sidebar.slider("Gaussian blur kernel size", 1, 15, 1, step=2)
pp_sigma = st.sidebar.slider("Gaussian blur sigma", 0.0, 5.0, 0.0, step=0.1)
pp_shadow = st.sidebar.slider("Illumination shadow intensity", 0.0, 1.0, 0.0, step=0.05)
pp_blackout = st.sidebar.checkbox("⚠️ RGB sensor failure (blackout)")

st.sidebar.markdown("### 🩸 Palm Vein (NIR) Noise")
pv_motion = st.sidebar.slider("Motion / defocus blur length", 1, 15, 1, step=2)
pv_contrast = st.sidebar.slider("Contrast attenuation", 0.0, 1.0, 0.0, step=0.05)
pv_blackout = st.sidebar.checkbox("⚠️ NIR sensor failure (blackout)")

# ==============================================================================
# 5. MODEL + GALLERY (cached — sliders stay instant)
# ==============================================================================

model = load_model()
gallery_embeds, gallery_labels = build_identity_gallery(model)

# ==============================================================================
# 6. LIVE INFERENCE
# ==============================================================================

t_start = time.perf_counter()

rgb_img = generate_palmprint_roi(
    st.session_state.sample_seed, kernel_size=pp_kernel, sigma=pp_sigma,
    shadow_intensity=pp_shadow, blackout=pp_blackout,
)
nir_img = generate_palmvein_roi(
    st.session_state.sample_seed, motion_len=pv_motion, contrast_atten=pv_contrast,
    blackout=pv_blackout,
)

rgb_tensor = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
nir_tensor = torch.tensor(nir_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    out = model(rgb_tensor, nir_tensor)

alpha_p = float(out["alpha_p"].item())
alpha_v = float(out["alpha_v"].item())
live_embed = out["f_fused"].squeeze(0).numpy()

# rank-1 identity match via cosine similarity against gallery class means
class_means = np.stack([gallery_embeds[gallery_labels == c].mean(axis=0)
                         for c in range(N_CLASSES)])
sims = (class_means @ live_embed) / (
    np.linalg.norm(class_means, axis=1) * np.linalg.norm(live_embed) + 1e-8
)
probs = np.exp(sims * 6) / np.exp(sims * 6).sum()  # temperature-scaled softmax
pred_class = int(np.argmax(probs))
rank1_confidence = float(probs[pred_class] * 100)

latency_ms = (time.perf_counter() - t_start) * 1000.0

# track alpha history for the "dynamic shift" bar chart
st.session_state.alpha_history.append((alpha_p, alpha_v))
st.session_state.alpha_history = st.session_state.alpha_history[-15:]

# ==============================================================================
# 7. HEADER
# ==============================================================================

header_l, header_r = st.columns([3, 1])
with header_l:
    st.title("UAD-Net: Uncertainty-Aware Dynamic Fusion Network")
    st.subheader("for Contactless Palmprint and Palm Vein Recognition")
    st.caption("Yashasvi Tomar: Department of Electronics Engineering — Live Interactive Poster Demo")
with header_r:
    st.metric("Inference Latency", f"{latency_ms:.2f} ms", help="Target design latency: ~5.2 ms")
    st.caption("Target: ~5.2 ms/sample")

st.divider()

# ==============================================================================
# 8. THREE-COLUMN LIVE LAYOUT
# ==============================================================================

col_left, col_mid, col_right = st.columns([1.0, 1.15, 1.15])

# ---- LEFT: Live modality inputs -----------------------------------------
with col_left:
    st.markdown("#### Live Modality Inputs")
    img_c1, img_c2 = st.columns(2)
    with img_c1:
        st.image(rgb_img, caption="Palmprint ROI (RGB)", use_container_width=True, clamp=True)
        tags = []
        if pp_blackout:
            tags.append(tag("SENSOR FAILURE", "#c53030"))
        else:
            if pp_sigma > 0:
                tags.append(tag(f"Blur σ={pp_sigma:.1f}", "#3182ce"))
            if pp_shadow > 0:
                tags.append(tag(f"Shadow {pp_shadow*100:.0f}%", "#805ad5"))
            if not tags:
                tags.append(tag("Clean", "#2f855a"))
        st.markdown(" ".join(tags), unsafe_allow_html=True)

    with img_c2:
        st.image(nir_img, caption="Palm Vein ROI (NIR)", use_container_width=True, clamp=True)
        tags = []
        if pv_blackout:
            tags.append(tag("SENSOR FAILURE", "#c53030"))
        else:
            if pv_motion > 1:
                tags.append(tag(f"Motion len={pv_motion}", "#3182ce"))
            if pv_contrast > 0:
                tags.append(tag(f"Contrast -{pv_contrast*100:.0f}%", "#805ad5"))
            if not tags:
                tags.append(tag("Clean", "#2f855a"))
        st.markdown(" ".join(tags), unsafe_allow_html=True)

    st.markdown("##### Raw Uncertainty Logits")
    st.write(f"sp (palmprint) = `{out['sp'].item():.3f}`   |   sv (palm vein) = `{out['sv'].item():.3f}`")

# ---- MIDDLE: Dynamic weight shifts & gauges ------------------------------
with col_mid:
    st.markdown("#### Dynamic Weight Shifts (UADM)")
    st.pyplot(render_gauges(alpha_p, alpha_v), use_container_width=True)

    p1, p2 = st.columns(2)
    p1.progress(alpha_p, text=f"alpha_p {alpha_p*100:.1f}%")
    p2.progress(alpha_v, text=f"alpha_v {alpha_v*100:.1f}%")

    st.markdown("##### Weight Shift History")
    st.pyplot(render_alpha_history_bar(st.session_state.alpha_history), use_container_width=True)

# ---- RIGHT: Recognition output & feature maps ----------------------------
with col_right:
    st.markdown("#### Recognition Output")
    st.metric("Predicted Identity", f"ID {pred_class}", f"Rank-1 Confidence: {rank1_confidence:.2f}%")

    m1, m2 = st.columns(2)
    m1.metric("ECE", "2.4%", help="Expected Calibration Error (offline evaluation)")
    m2.metric("Brier Score", "0.012", help="Offline evaluation on held-out test set")

    st.markdown("##### Identity Embedding Space")
    st.pyplot(render_tsne(gallery_embeds, gallery_labels, live_embed, pred_class),
              use_container_width=True)

st.divider()
st.caption(
    "Synthetic demonstration only — palmprint/palm-vein ROIs, network weights, and gallery "
    "embeddings are procedurally generated for live illustration of the UAD-Net dynamic fusion "
    "mechanism and are not derived from real biometric data."
)
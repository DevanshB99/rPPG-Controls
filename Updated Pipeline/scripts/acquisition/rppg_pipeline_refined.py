#!/usr/bin/env python3
import os
os.environ['QT_QPA_PLATFORM']  = 'xcb'
os.environ['QT_QPA_FONTDIR']   = '/usr/share/fonts/truetype/dejavu'
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
import scipy.signal
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from sklearn.mixture import GaussianMixture
from concurrent.futures import ThreadPoolExecutor

from datetime import datetime

VID        = Path("/home/macs/Documents/rPPG-Controls/Measurement Data/"
                  "DocBOT_2026-04-07_15-04-35/recording_2026-04-07T22-04-35Z.mov")
GT_CSV     = Path("/home/macs/Documents/rPPG-Controls/Measurement Data/"
                  "DocBOT_2026-04-07_15-04-35/vitals.csv")
MODEL_PATH = Path("/home/macs/Documents/rPPG-Controls/rppg/face_landmarker.task")

_PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent
_RUN_TS       = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR       = _PIPELINE_DIR / 'results' / 'input_results' / _RUN_TS
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV       = OUT_DIR / 'rppg_output.csv'
OUT_TXT       = OUT_DIR / 'summary.txt'

# ── Models ─────────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
bisenet   = SegformerForSemanticSegmentation.from_pretrained(
                "jonathandinu/face-parsing").to(DEVICE).eval()

_lm_opts = mp_vision.FaceLandmarkerOptions(
    base_options=mp_base.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(_lm_opts)

# ── Module-level landmark index tables (constant — no per-call allocation) ─
# Normal camera orientation (pts[33].x < pts[263].x)
_LI_NORMAL = dict(eye_out=33,  tmpl=234, brow_o=70,  brow_i=107,
                   ala=129, jaw=172,
                   f_lat=54,  f_hi=103, f_mid=67,  f_lo=109, tmpl_top=21)
_RI_NORMAL = dict(eye_out=263, tmpl=454, brow_o=300, brow_i=336,
                   ala=358, jaw=397,
                   f_lat=284, f_hi=332, f_mid=297, f_lo=338, tmpl_top=251)
# Mirrored camera: swap sides
_LI_MIRROR = _RI_NORMAL
_RI_MIRROR = _LI_NORMAL

# ── LBP circular-transition lookup table (computed once at import) ─────────
_LBP_TRANS = np.zeros(256, dtype=np.uint8)
for _v in range(256):
    _b = format(_v, '08b')
    _circ = _b + _b[0]
    _LBP_TRANS[_v] = sum(_circ[i] != _circ[i + 1] for i in range(8))
del _v, _b, _circ

# ── Thread pool for parallel per-frame computation ─────────────────────────
_POOL = ThreadPoolExecutor(max_workers=2)


# ─────────────────────────────────────────────────────────────────────────────
# BiSeNet face parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_skin(crop_rgb: np.ndarray) -> tuple:
    """(bool_mask, float32_prob_map) from BiSeNet.  Prob map ∈ [0,1] for class 1."""
    inp = processor(images=Image.fromarray(crop_rgb), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = bisenet(**inp).logits
    up    = F.interpolate(logits, size=crop_rgb.shape[:2],
                          mode="bilinear", align_corners=False)
    probs = F.softmax(up, dim=1)
    return (up.argmax(1).squeeze().cpu().numpy() == 1), probs[0, 1].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame colour-space pre-computation (called once per frame)
# ─────────────────────────────────────────────────────────────────────────────
def compute_frame_maps(rgb: np.ndarray) -> dict:
    """Compute every colour-space representation once and return a dict.
    All subsequent functions consume this dict — no redundant conversions."""
    ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)          # uint8 (H,W,3)
    hsv   = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)             # uint8 (H,W,3)
    gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)            # uint8 (H,W)
    ycf   = ycbcr.astype(np.float32)
    hsvf  = hsv.astype(np.float32)
    return dict(
        rgb=rgb, ycbcr=ycbcr, hsv=hsv, gray=gray,
        Y=ycf[:,:,0],  Cr=ycf[:,:,1],  Cb=ycf[:,:,2],
        H_deg=hsvf[:,:,0]*2.0,          # [0,360)
        S_n=hsvf[:,:,1]/255.0,
        V_n=hsvf[:,:,2]/255.0,
        R=rgb[:,:,0].astype(np.float32),
        G=rgb[:,:,1].astype(np.float32),
        B=rgb[:,:,2].astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Texture feature maps — computed once per frame, cropped to face bbox
# ─────────────────────────────────────────────────────────────────────────────
def compute_texture_maps(norm_u8: np.ndarray, norm_gray: np.ndarray,
                          bbox: tuple) -> dict:
    """Compute LBP, gradient coherence, SVC, and specular mask.
    Runs on the face bounding-box crop, then maps back to frame coordinates.
    Called once per frame — results shared by forehead and cheek hair rejection.

    bbox: (y1, y2, x1, x2) face crop with padding.
    Returns dict of full-frame float32 maps."""
    y1, y2, x1, x2 = bbox
    u8  = norm_u8  [y1:y2, x1:x2]
    gf  = norm_gray[y1:y2, x1:x2]
    H, W = norm_u8.shape

    # LBP circular transition count
    c   = u8.astype(np.int32)
    lbp = np.zeros_like(u8, dtype=np.uint8)
    for bit, (dy, dx) in enumerate([(-1,0),(-1,1),(0,1),(1,1),
                                     (1,0),(1,-1),(0,-1),(-1,-1)]):
        nb = np.roll(np.roll(u8, dy, axis=0), dx, axis=1).astype(np.int32)
        lbp |= ((nb >= c).astype(np.uint8) << bit)
    lbp_crop = _LBP_TRANS[lbp].astype(np.float32)

    # Gradient orientation coherence (unit-vector field smoothing)
    sx  = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    sy  = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(sx, sy) + 1e-8
    coh_crop = np.hypot(cv2.GaussianBlur(sx/mag, (0,0), 6.0),
                         cv2.GaussianBlur(sy/mag, (0,0), 6.0)).astype(np.float32)

    # Spectral Variation Coefficient
    mu  = cv2.blur(gf, (9, 9))
    mu2 = cv2.blur(gf**2, (9, 9))
    svc_crop = (np.sqrt(np.maximum(mu2-mu**2, 0.0)) / (mu+1.0)).astype(np.float32)

    # Specular highlight mask
    loc_mu      = cv2.blur(gf, (31, 31))
    spec_crop   = gf > (loc_mu * 1.25 + 25)

    # Map cropped results back into full-frame arrays
    def full(crop, dtype=np.float32):
        out = np.zeros((H, W), dtype)
        out[y1:y2, x1:x2] = crop
        return out

    return dict(lbp=full(lbp_crop), coh=full(coh_crop),
                svc=full(svc_crop), specular=full(spec_crop, bool))


def hair_rejection(tex: dict, roi: np.ndarray, pct: int = 70) -> np.ndarray:
    """Flag hair pixels using pre-computed texture maps (see compute_texture_maps).
    Requires TWO independent signals to flag a pixel: non-uniform LBP AND
    (high orientation coherence OR high SVC).  Specular pixels are immune."""
    lbp, coh, svc = tex['lbp'], tex['coh'], tex['svc']
    specular = tex['specular']
    valid = roi & ~specular
    ref   = valid if valid.sum() > 50 else None

    def pth(arr, p):
        return float(np.percentile(arr[ref] if ref is not None else arr, p))

    lbp_t = pth(lbp, pct)
    coh_t = pth(coh, max(pct-5, 50))
    svc_t = pth(svc, max(pct-5, 50))
    return (lbp > lbp_t) & ((coh > coh_t) | (svc > svc_t)) & ~specular


# ─────────────────────────────────────────────────────────────────────────────
# ROI polygons + exclusion masks
# ─────────────────────────────────────────────────────────────────────────────
def build_rois(pts: np.ndarray, H: int, W: int):
    """Adaptive forehead + cheek polygons from 468-point MediaPipe landmarks.
    Uses module-level index tables — no per-call dict allocation."""
    normal = pts[33, 0] < pts[263, 0]
    LI, RI = (_LI_NORMAL, _RI_NORMAL) if normal else (_LI_MIRROR, _RI_MIRROR)
    L = {k: pts[v] for k, v in LI.items()}
    R = {k: pts[v] for k, v in RI.items()}

    fore = np.array([
        L['tmpl_top'], L['f_lat'], L['f_hi'], L['f_mid'], L['f_lo'],
        pts[10],
        R['f_lo'], R['f_mid'], R['f_hi'], R['f_lat'], R['tmpl_top'],
        R['tmpl'], R['brow_o'], R['brow_i'], pts[9],
        L['brow_i'], L['brow_o'], L['tmpl'],
    ], dtype=np.float32)

    def cheek(s):
        ey = float(s['eye_out'][1]) + 0.12*(float(s['jaw'][1])-float(s['eye_out'][1]))
        ly = float(s['eye_out'][1]) + 0.72*(float(s['jaw'][1])-float(s['eye_out'][1]))
        return np.array([
            [s['eye_out'][0], ey], s['tmpl'],
            [s['tmpl'][0], ly], [s['jaw'][0], ly],
            [s['ala'][0],  ly], s['ala'],
            [s['ala'][0],  ey],
        ], dtype=np.float32)

    return fore, cheek(L), cheek(R)


def poly_mask(poly: np.ndarray, H: int, W: int) -> np.ndarray:
    poly = np.clip(poly, [0,0], [W-1, H-1])
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    return m.astype(bool)


def build_all_masks(pts, fp, lp, rp, H, W):
    """Compute the five polygon masks (3 ROI + 2 exclusion) in one call."""
    must_poly  = np.array([pts[129], pts[358], pts[291], pts[61]], np.float32)
    beard_poly = np.array([pts[61], pts[291], pts[397], pts[152], pts[172]], np.float32)
    fm   = poly_mask(fp, H, W)
    lcm  = poly_mask(lp, H, W)
    rcm  = poly_mask(rp, H, W)
    must = poly_mask(must_poly,  H, W)
    brd  = poly_mask(beard_poly, H, W)
    return fm, lcm, rcm, must | brd


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive multi-method colour skin detection
# ─────────────────────────────────────────────────────────────────────────────
def adaptive_color_skin(fm: dict, seed_mask: np.ndarray) -> tuple:
    """
    Three independent colour-space methods seeded from BiSeNet-confirmed pixels,
    fitted per image-half to handle asymmetric lighting.

    Method 1 — Adaptive YCbCr : mean ± 2.5σ on Cb, Cr, Y.
    Method 2 — RGB Kovac      : static structural rule (skin-tone agnostic).
    Method 3 — Adaptive HSV   : mean ± 2.5σ on H and S with circular-hue stats.
    Shadow recovery            : broadened chrominance for dark-lit skin (override).

    fm: dict from compute_frame_maps().
    Returns (skin_mask, vote_map ∈ {0,1,2,3}).
    """
    Y, Cr, Cb = fm['Y'], fm['Cr'], fm['Cb']
    R, G, B   = fm['R'], fm['G'], fm['B']
    H_deg, S_n, V_n = fm['H_deg'], fm['S_n'], fm['V_n']
    W_img = fm['rgb'].shape[1]

    m1 = np.zeros(fm['rgb'].shape[:2], bool)
    m3 = np.zeros(fm['rgb'].shape[:2], bool)

    for x0, x1 in [(0, W_img//2), (W_img//2, W_img)]:
        half = np.zeros(fm['rgb'].shape[:2], bool)
        half[:, x0:x1] = True
        seed   = seed_mask & half
        n_seed = int(seed.sum())

        if n_seed >= 30:
            sCb, sCr, sY = Cb[seed], Cr[seed], Y[seed]
            cb_lo = max(55,  float(sCb.mean() - 2.5*sCb.std()))
            cb_hi = min(148, float(sCb.mean() + 2.5*sCb.std()))
            cr_lo = max(110, float(sCr.mean() - 2.5*sCr.std()))
            cr_hi = min(195, float(sCr.mean() + 2.5*sCr.std()))
            y_lo  = max(8,   float(np.percentile(sY, 0.5))  - 10)
            y_hi  = min(252, float(np.percentile(sY, 99.5)) + 10)
        else:
            cb_lo, cb_hi = 77, 127
            cr_lo, cr_hi = 133, 173
            y_lo,  y_hi  = 20,  245
        m1 |= half & (Cb>=cb_lo)&(Cb<=cb_hi)&(Cr>=cr_lo)&(Cr<=cr_hi)&(Y>=y_lo)&(Y<=y_hi)

        if n_seed >= 30:
            sH  = H_deg[seed];  sS = S_n[seed]
            ang = np.deg2rad(sH)
            h_mean = float(np.rad2deg(np.arctan2(np.sin(ang).mean(),
                                                  np.cos(ang).mean())) % 360)
            h_std  = float(np.std(sH))
            h_lo   = (h_mean - 2.5*h_std) % 360
            h_hi   = (h_mean + 2.5*h_std) % 360
            s_lo   = max(0.05, float(sS.mean() - 2.5*sS.std()))
            s_hi   = min(0.90, float(sS.mean() + 2.5*sS.std()))
        else:
            h_lo, h_hi, s_lo, s_hi = 0.0, 50.0, 0.10, 0.75

        hue_ok = (H_deg>=h_lo)&(H_deg<=h_hi) if h_lo<=h_hi else (H_deg>=h_lo)|(H_deg<=h_hi)
        m3 |= half & hue_ok & (S_n>=s_lo) & (S_n<=s_hi) & (V_n>=0.10)

    Rmax = np.maximum(np.maximum(R,G),B);  Rmin = np.minimum(np.minimum(R,G),B)
    m2   = (R>95)&(G>40)&(B>20)&((Rmax-Rmin)>15)&(np.abs(R-G)>15)&(R>G)&(R>B)
    sh   = (Cb>=60)&(Cb<=145)&(Cr>=115)&(Cr<=190)&(Y>=10)&(Y<45)&(S_n>=0.08)

    vote = m1.astype(np.uint8) + m2.astype(np.uint8) + m3.astype(np.uint8)
    return (vote >= 2) | sh, vote


# ─────────────────────────────────────────────────────────────────────────────
# GMM clustering — fit every N frames, predict every frame with cached model
# FCM removed: in 2D Cb-Cr space with a 0.5 hard threshold it was
# redundant with GMM.  GMM with full covariance is strictly more expressive
# than the adaptive rectangular YCbCr thresholds and earns its place.
# ─────────────────────────────────────────────────────────────────────────────
def fit_gmm(ycbcr: np.ndarray, seed_mask: np.ndarray, n_comp: int = 3) -> dict | None:
    """Fit a GMM on BiSeNet-confirmed skin pixels in Cb-Cr space.
    The skin component is the one whose centroid is closest to the seed mean.
    Subsamples to ≤4000 points; called every BISENET_EVERY frames."""
    Cb = ycbcr[:,:,2].astype(np.float32)
    Cr = ycbcr[:,:,1].astype(np.float32)
    X  = np.column_stack([Cb[seed_mask], Cr[seed_mask]])
    if X.shape[0] < n_comp*10:
        return None
    idx = np.random.default_rng(0).choice(X.shape[0], min(4000, X.shape[0]), replace=False)
    Xs  = X[idx];  mu = Xs.mean(axis=0)
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full',
                           max_iter=60, random_state=42).fit(Xs)
    si  = int(np.argmin(np.linalg.norm(gmm.means_ - mu, axis=1)))
    return dict(gmm=gmm, si=si)


def predict_gmm(model: dict | None, ycbcr: np.ndarray,
                roi: np.ndarray) -> np.ndarray:
    """Predict GMM skin membership for ROI pixels. Returns bool mask."""
    out = np.zeros(ycbcr.shape[:2], bool)
    if model is None:
        return out
    Cb = ycbcr[:,:,2].astype(np.float32)
    Cr = ycbcr[:,:,1].astype(np.float32)
    ys, xs = np.where(roi)
    if len(ys) == 0:
        return out
    X     = np.column_stack([Cb[ys,xs], Cr[ys,xs]])
    proba = model['gmm'].predict_proba(X)[:, model['si']]
    out[ys, xs] = proba > 0.50
    return out



if __name__ == '__main__':
    gt_raw      = pd.read_csv(GT_CSV)
    gt_time_raw = gt_raw['offset_seconds'].values
    gt_hr_raw   = gt_raw['heart_rate'].values
    valid_gt    = ~np.isnan(gt_hr_raw)
    gt_time_raw = gt_time_raw[valid_gt]
    gt_hr_raw   = gt_hr_raw[valid_gt]

    cap     = cv2.VideoCapture(str(VID))
    fs      = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    R_t, G_t, B_t, lum_t, npx_t, det_t    = [], [], [], [], [], []
    bisenet_conf_t, vote_mean_t             = [], []
    highconf_pct_t, roi_coverage_t         = [], []
    green_cv_t, hair_rej_frac_t            = [], []
    clust_agree_t, quality_t               = [], []
    Rf_t, Gf_t, Bf_t                       = [], [], []  # forehead region
    Rl_t, Gl_t, Bl_t                       = [], [], []  # left cheek region
    Rr_t, Gr_t, Br_t                       = [], [], []  # right cheek region
    fi = 0

    print(f"Processing {n_total} frames at {fs:.2f} fps  [{DEVICE}]")
    print("Press Q in the preview window to stop early.\n")

    DISPLAY_W     = 700
    BISENET_EVERY = 10
    MAX_CROP_PX   = 384
    DISPLAY_EVERY = 3     # refresh preview every N frames (display doesn't need 30 fps)

    # Persistent caches — all invalidated together on BiSeNet frames
    sk_cache      = None   # bool (H,W)
    sk_prob_cache = None   # float32 (H,W)
    gmm_cache     = None   # fitted GMM dict
    gmm_sk_cache  = None   # bool (H,W) — GMM skin mask, cached until next BiSeNet frame
    mask_cache    = None   # (fm,lcm,rcm,excl) — invalidated when landmarks move
    pts_cache     = None   # landmark array from mask_cache frame, for movement check

    cv2.namedWindow("rPPG — Original | Skin ROI", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ok, bgr = cap.read()
        if not ok:
            break
        fi += 1
        if fi % 150 == 0:
            print(f"  {fi}/{n_total}  skin_px={npx_t[-1] if npx_t else 0}")

        H, W  = bgr.shape[:2]
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fm_d  = compute_frame_maps(rgb)          # single source of truth for all colour maps

        disp_h = int(DISPLAY_W * H / W)
        status = "No face"

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_res = face_landmarker.detect_for_video(mp_img, timestamp_ms)

        if mp_res.face_landmarks:
            pts = np.array([[lm.x*W, lm.y*H] for lm in mp_res.face_landmarks[0]],
                           dtype=np.float32)

            # Face bounding box with padding (for BiSeNet crop + texture crop)
            bx1 = max(0, int(pts[:,0].min()) - 20)
            by1 = max(0, int(pts[:,1].min()) - 20)
            bx2 = min(W, int(pts[:,0].max()) + 20)
            by2 = min(H, int(pts[:,1].max()) + 20)
            bbox = (by1, by2, bx1, bx2)

            # ── ROI polygon masks — recompute only if landmarks moved > 4 px ─
            if (mask_cache is None or pts_cache is None or
                    float(np.max(np.abs(pts - pts_cache))) > 4.0):
                fp, lp, rp       = build_rois(pts, H, W)
                fm, lcm, rcm, excl = build_all_masks(pts, fp, lp, rp, H, W)
                roi_union          = fm | lcm | rcm
                mask_cache         = (fp, lp, rp, fm, lcm, rcm, excl, roi_union)
                pts_cache          = pts.copy()
            else:
                fp, lp, rp, fm, lcm, rcm, excl, roi_union = mask_cache

            # ── BiSeNet + GMM fit every N frames, cache masks ─────────────────
            run_bisenet = (fi % BISENET_EVERY == 1 or sk_cache is None)
            if run_bisenet:
                crop    = rgb[by1:by2, bx1:bx2]
                h_c, w_c = crop.shape[:2]
                scale   = min(1.0, MAX_CROP_PX / max(h_c, w_c, 1))
                crop_s  = cv2.resize(crop, (max(1,int(w_c*scale)), max(1,int(h_c*scale))))
                sk_s, sk_prob_s = parse_skin(crop_s)
                sk_cr   = cv2.resize(sk_s.astype(np.uint8), (w_c, h_c),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                prob_cr = cv2.resize(sk_prob_s.astype(np.float32), (w_c, h_c),
                                      interpolation=cv2.INTER_LINEAR)
                sk_full      = np.zeros((H,W), bool)
                sk_prob_full = np.zeros((H,W), np.float32)
                sk_full     [by1:by2, bx1:bx2] = sk_cr
                sk_prob_full[by1:by2, bx1:bx2] = prob_cr
                sk_cache      = sk_full
                sk_prob_cache = sk_prob_full
                # Fit GMM and pre-compute its mask for the next N frames
                gmm_cache    = fit_gmm(fm_d['ycbcr'], sk_full & roi_union)
                gmm_sk_cache = predict_gmm(gmm_cache, fm_d['ycbcr'], roi_union)
            else:
                sk_full      = sk_cache
                sk_prob_full = sk_prob_cache
                # gmm_sk_cache remains from last BiSeNet frame — reused as-is

            gmm_sk = gmm_sk_cache if gmm_sk_cache is not None else np.zeros((H,W), bool)

            # ── Seed for adaptive colour methods ──────────────────────────────
            face_seed = sk_full & roi_union

            # ── Illumination-normalised grayscale ─────────────────────────────
            eye_dist  = float(np.linalg.norm(pts[33] - pts[263]))
            gray_f    = fm_d['gray'].astype(np.float32)
            local_mu  = cv2.GaussianBlur(gray_f, (0,0), max(15.0, eye_dist*0.4))
            norm_gray = gray_f / (local_mu + 1.0) * 128.0
            norm_u8   = np.clip(norm_gray, 0, 255).astype(np.uint8)

            # ── Parallel: texture maps | colour segmentation ──────────────────
            # Both are independent once face_seed and norm_u8 are ready.
            # numpy/OpenCV release the GIL → genuine parallel execution.
            fut_tex   = _POOL.submit(compute_texture_maps, norm_u8, norm_gray, bbox)
            fut_color = _POOL.submit(adaptive_color_skin, fm_d, face_seed)

            tex              = fut_tex.result()
            color_sk, vote_map = fut_color.result()

            # ── Hair rejection (shared texture maps — no recomputation) ───────
            hair_fore  = hair_rejection(tex, fm,      pct=78)
            hair_cheek = hair_rejection(tex, lcm|rcm, pct=72)

            # ── Final skin combination ─────────────────────────────────────────
            # Accept pixel as skin if:
            #   (a) BiSeNet labels it skin, OR
            #   (b) ≥2 colour methods agree, OR
            #   (c) GMM and at least 1 colour method agree
            # Then subtract hair texture and geometric exclusion zones.
            comb_sk   = sk_full | color_sk | (gmm_sk & (vote_map >= 1))
            spec_mask = tex['specular']
            fore_skin = comb_sk & fm   & ~hair_fore  & ~excl & ~spec_mask
            l_skin    = comb_sk & lcm  & ~hair_cheek & ~excl & ~spec_mask
            r_skin    = comb_sk & rcm  & ~hair_cheek & ~excl & ~spec_mask
            final     = fore_skin | l_skin | r_skin
            n_px      = int(final.sum())
            status    = f"skin px: {n_px}"

            if n_px >= 50:
                f   = rgb.astype(np.float32)
                lum = float(f[final].mean())
                fn  = f / max(lum, 1.0) * 128.0
                # BiSeNet probability-weighted face-wide means
                w_all = sk_prob_full[final].clip(0.01)
                R_t.append(float(np.average(fn[:,:,0][final], weights=w_all)))
                G_t.append(float(np.average(fn[:,:,1][final], weights=w_all)))
                B_t.append(float(np.average(fn[:,:,2][final], weights=w_all)))
                # Per-region tracking for spatial PCA
                for reg, Rl, Gl, Bl in [(fore_skin, Rf_t, Gf_t, Bf_t),
                                         (l_skin,    Rl_t, Gl_t, Bl_t),
                                         (r_skin,    Rr_t, Gr_t, Br_t)]:
                    if int(reg.sum()) >= 20:
                        wr = sk_prob_full[reg].clip(0.01)
                        Rl.append(float(np.average(fn[:,:,0][reg], weights=wr)))
                        Gl.append(float(np.average(fn[:,:,1][reg], weights=wr)))
                        Bl.append(float(np.average(fn[:,:,2][reg], weights=wr)))
                    else:
                        Rl.append(R_t[-1]);  Gl.append(G_t[-1]);  Bl.append(B_t[-1])
                lum_t.append(lum);  npx_t.append(n_px);  det_t.append(1)

                # ── Per-frame confidence metrics ──────────────────────────────
                roi_area    = int(roi_union.sum())
                bc          = float(sk_prob_full[final].mean())
                vm          = float(vote_map[final].mean()) / 3.0
                hc          = float((vote_map[final]==3).sum()) / max(n_px,1)
                rc          = n_px / max(roi_area, 1)
                g_vals      = fm_d['G'][final]
                gcv         = float(np.std(g_vals) / (np.mean(g_vals)+1e-9))
                hair_px     = int((hair_fore&fm).sum() + (hair_cheek&(lcm|rcm)).sum())
                hrf         = hair_px / max(roi_area,1)
                clust_agree = float(gmm_sk[final].sum()) / max(n_px,1)
                q = (0.25*bc + 0.20*vm + 0.20*clust_agree
                   + 0.15*min(1.0,rc/0.40)
                   + 0.12*max(0.0,1.0-gcv/0.15)
                   + 0.08*max(0.0,1.0-hrf/0.50))

                bisenet_conf_t.append(bc);  vote_mean_t.append(vm)
                highconf_pct_t.append(hc);  roi_coverage_t.append(rc)
                green_cv_t.append(gcv);     hair_rej_frac_t.append(hrf)
                clust_agree_t.append(clust_agree);  quality_t.append(float(q))

            # ── Display (throttled — human eye doesn't need 30 fps here) ─────
            if fi % DISPLAY_EVERY == 0:
                sx, sy     = DISPLAY_W/W, disp_h/H
                left       = cv2.resize(bgr, (DISPLAY_W, disp_h))
                right      = left.copy()
                ROI_COLORS = [(fp,(0,255,255)), (lp,(255,255,0)), (rp,(255,0,255))]
                for poly, col in ROI_COLORS:
                    pd_ = (poly*[sx,sy]).astype(np.int32).reshape(-1,1,2)
                    cv2.polylines(left,  [pd_], True, col, 2)
                    cv2.polylines(right, [pd_], True, col, 2)
                if n_px >= 50:
                    ov = bgr.astype(np.float32)
                    ov[final,0] *= 0.55
                    ov[final,1]  = np.clip(ov[final,1]*0.55+144*0.45, 0, 255)
                    ov[final,2] *= 0.55
                    right = cv2.resize(ov.astype(np.uint8), (DISPLAY_W, disp_h))
                    for poly, col in ROI_COLORS:
                        pd_ = (poly*[sx,sy]).astype(np.int32).reshape(-1,1,2)
                        cv2.polylines(right, [pd_], True, col, 2)
                q_str = f"  Q={quality_t[-1]:.2f}" if quality_t else ""
                cv2.putText(left,  f"frame {fi}/{n_total}", (8,26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                cv2.putText(right, status+q_str, (8,26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                cv2.putText(right, "Fore=cyan  L=yellow  R=magenta  Q=quality[0-1]",
                            (8,disp_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
                cv2.imshow("rPPG — Original | Skin ROI", np.hstack([left, right]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    _POOL.shutdown(wait=False)
    face_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nExtracted {len(R_t)} valid frames.")

    # ── CHROM projection + linear detrend ─────────────────────────────────────
    T      = len(R_t)
    t_axis = np.arange(T) / fs
    R, G, B = np.array(R_t), np.array(G_t), np.array(B_t)
    Rn, Gn, Bn = R/R.mean(), G/G.mean(), B/B.mean()
    Xs  = 3*Rn - 2*Gn
    Ys  = 1.5*Rn + Gn - 1.5*Bn
    alp = np.std(Xs) / (np.std(Ys)+1e-9)
    S   = Xs - alp*Ys
    tv  = np.arange(T, dtype=float)
    S_det = S - np.polyval(np.polyfit(tv, S, 1), tv)

    # Spatial region PCA — 9 signals (3 regions × 3 channels) → best cardiac component
    X_reg = np.stack([np.array(Rf_t), np.array(Gf_t), np.array(Bf_t),
                      np.array(Rl_t), np.array(Gl_t), np.array(Bl_t),
                      np.array(Rr_t), np.array(Gr_t), np.array(Br_t)])
    X_reg = (X_reg - X_reg.mean(1, keepdims=True)) / (X_reg.std(1, keepdims=True) + 1e-9)
    _, _, Vt = np.linalg.svd(X_reg, full_matrices=False)
    card_pow = np.zeros(min(3, len(Vt)))
    for i in range(len(card_pow)):
        fp_w, pp_w = scipy.signal.welch(Vt[i], fs=fs, nperseg=min(T//2, int(fs*10)))
        band_w     = (fp_w >= 0.67) & (fp_w <= 3.5)
        card_pow[i] = pp_w[band_w].max() / (pp_w.mean() + 1e-9)
    best_pc = int(np.argmax(card_pow))
    S_pca   = Vt[best_pc]
    S_pca   = S_pca - np.polyval(np.polyfit(tv, S_pca, 1), tv)

    # ── Four IIR bandpass filters ──────────────────────────────────────────────
    Wn = [0.7, 3.5]
    b_bw,a_bw = scipy.signal.butter( 4,        Wn, btype='band', fs=fs)
    b_c1,a_c1 = scipy.signal.cheby1( 4, 0.5,   Wn, btype='band', fs=fs)
    b_c2,a_c2 = scipy.signal.cheby2( 4, 40,    Wn, btype='band', fs=fs)
    b_el,a_el = scipy.signal.ellip(  4, 0.5,40, Wn, btype='band', fs=fs)
    S_bw = scipy.signal.filtfilt(b_bw,a_bw,S_det)
    S_c1 = scipy.signal.filtfilt(b_c1,a_c1,S_det)
    S_c2 = scipy.signal.filtfilt(b_c2,a_c2,S_det)
    S_el = scipy.signal.filtfilt(b_el,a_el,S_det)

    # ── Ground-truth BPM interpolated to frame times ──────────────────────────
    gt_bpm_frame = np.interp(t_axis, gt_time_raw, gt_hr_raw, left=np.nan, right=np.nan)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        'frame_index':      np.arange(1, T+1),
        'time_s':           t_axis,
        'R_skin_raw':       R,
        'R_normalized':     Rn,
        'G_skin_raw':       G,
        'G_normalized':     Gn,
        'B_skin_raw':       B,
        'B_normalized':     Bn,
        'frame_luminance':  np.array(lum_t),
        'face_detected':    np.array(det_t, dtype=float),
        'skin_pixel_count': np.array(npx_t, dtype=float),
        'BVP_detrended':    S_det,
        'BVP_butterworth':  S_bw,
        'BVP_cheby1':       S_c1,
        'BVP_cheby2':       S_c2,
        'BVP_elliptic':     S_el,
        'gt_bpm':           gt_bpm_frame,
        # Quality columns
        # bisenet_conf   : mean BiSeNet P(skin) over accepted pixels
        # vote_mean      : 3-method colour agreement normalised [0,1]
        # highconf_pct   : fraction of skin px where all 3 colour methods agreed
        # roi_coverage   : skin_px / anatomical_ROI_area
        # green_cv       : spatial std/mean of G — low = uniform skin
        # hair_rej_frac  : fraction of ROI removed by LBP+coherence+SVC filter
        # clust_agree    : fraction of skin px also in GMM skin cluster
        # quality_score  : composite 0-1
        'bisenet_conf':     np.array(bisenet_conf_t),
        'vote_mean':        np.array(vote_mean_t),
        'highconf_pct':     np.array(highconf_pct_t),
        'roi_coverage':     np.array(roi_coverage_t),
        'green_cv':         np.array(green_cv_t),
        'hair_rej_frac':    np.array(hair_rej_frac_t),
        'clust_agree':      np.array(clust_agree_t),
        'quality_score':    np.array(quality_t),
        'BVP_pca':          S_pca,
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}  ({T} rows × {len(out.columns)} cols)")

    q = np.array(quality_t)
    summary = [
        f"rPPG pipeline run : {_RUN_TS}",
        f"Video             : {VID}",
        f"Frames            : {T} valid  |  fps: {fs:.2f}  |  PCA component: {best_pc}",
        "",
        f"Pipeline quality summary across {T} frames:",
        f"  quality_score  mean={q.mean():.3f}  std={q.std():.3f}  p10={np.percentile(q,10):.3f}  p90={np.percentile(q,90):.3f}",
        f"  bisenet_conf   mean={np.mean(bisenet_conf_t):.3f}",
        f"  vote_mean      mean={np.mean(vote_mean_t):.3f}  (1.0 = all 3 colour methods agree)",
        f"  clust_agree    mean={np.mean(clust_agree_t):.3f}  (GMM consensus)",
        f"  roi_coverage   mean={np.mean(roi_coverage_t):.3f}  (skin fraction of face ROI)",
        f"  green_cv       mean={np.mean(green_cv_t):.3f}  (< 0.10 = spatially uniform)",
        f"  hair_rej_frac  mean={np.mean(hair_rej_frac_t):.3f}  (fraction removed as hair)",
    ]
    for line in summary:
        print(line)
    OUT_TXT.write_text('\n'.join(summary) + '\n')
    print(f"Summary → {OUT_TXT}")

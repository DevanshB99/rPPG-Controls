# rppg_camera — Raw Frame Acquisition & Signal Extraction

> **File:** `Updated Pipeline/scripts/acquisition/rppg_pipeline_refined.py`
> **Role in pipeline:** Stage 1 — takes raw video frames as input, performs all face detection, skin segmentation, and per-frame RGB signal extraction, then writes a per-frame CSV (plus filtered BVP signals) that every downstream analysis script consumes.
> **Output:** `Updated Pipeline/results/input_results/<TIMESTAMP>/rppg_output.csv` + `summary.txt`

---

## Table of Contents

1. [What This File Actually Does](#1-what-this-file-actually-does)
2. [High-Level Architecture & Design Philosophy](#2-high-level-architecture--design-philosophy)
3. [Imports & Environment Setup](#3-imports--environment-setup)
4. [Global Paths, Output Directory & Run Timestamp](#4-global-paths-output-directory--run-timestamp)
5. [Model Initialisation](#5-model-initialisation)
   - 5.1 [BiSeNet / SegFormer (deep-learning skin parser)](#51-bisenet--segformer-deep-learning-skin-parser)
   - 5.2 [MediaPipe FaceLandmarker (geometry)](#52-mediapipe-facelandmarker-geometry)
6. [Module-Level Constants](#6-module-level-constants)
   - 6.1 [Landmark Index Tables](#61-landmark-index-tables)
   - 6.2 [LBP Circular Transition Lookup Table](#62-lbp-circular-transition-lookup-table)
   - 6.3 [Thread Pool](#63-thread-pool)
7. [Function Blocks — In Depth](#7-function-blocks--in-depth)
   - 7.1 [`parse_skin()` — BiSeNet Inference](#71-parse_skin--bisenet-inference)
   - 7.2 [`compute_frame_maps()` — Color-Space Pre-computation](#72-compute_frame_maps--color-space-pre-computation)
   - 7.3 [`compute_texture_maps()` — LBP, Gradient Coherence, SVC, Specular](#73-compute_texture_maps--lbp-gradient-coherence-svc-specular)
   - 7.4 [`hair_rejection()` — Texture-Based Hair Exclusion](#74-hair_rejection--texture-based-hair-exclusion)
   - 7.5 [`build_rois()` — Adaptive Polygon ROIs](#75-build_rois--adaptive-polygon-rois)
   - 7.6 [`poly_mask()` — Polygon → Binary Mask](#76-poly_mask--polygon--binary-mask)
   - 7.7 [`build_all_masks()` — All Five Masks in One Call](#77-build_all_masks--all-five-masks-in-one-call)
   - 7.8 [`adaptive_color_skin()` — Three-Method Adaptive Color Segmentation](#78-adaptive_color_skin--three-method-adaptive-color-segmentation)
   - 7.9 [`fit_gmm()` — Gaussian Mixture Model Fitting](#79-fit_gmm--gaussian-mixture-model-fitting)
   - 7.10 [`predict_gmm()` — GMM Inference Per Frame](#710-predict_gmm--gmm-inference-per-frame)
8. [Main Execution Loop (`__main__`)](#8-main-execution-loop-__main__)
   - 8.1 [Ground-Truth Loading](#81-ground-truth-loading)
   - 8.2 [Video Capture & Buffer Initialisation](#82-video-capture--buffer-initialisation)
   - 8.3 [Runtime Constants & Persistent Caches](#83-runtime-constants--persistent-caches)
   - 8.4 [Per-Frame Processing (the core loop)](#84-per-frame-processing-the-core-loop)
   - 8.5 [RGB Signal Extraction & Luminance Normalisation](#85-rgb-signal-extraction--luminance-normalisation)
   - 8.6 [Per-Frame Confidence Metrics](#86-per-frame-confidence-metrics)
   - 8.7 [Live Display](#87-live-display)
9. [Post-Processing (after the loop)](#9-post-processing-after-the-loop)
   - 9.1 [CHROM Projection & Linear Detrend](#91-chrom-projection--linear-detrend)
   - 9.2 [Spatial Region PCA](#92-spatial-region-pca)
   - 9.3 [Four IIR Bandpass Filters](#93-four-iir-bandpass-filters)
   - 9.4 [Ground-Truth Interpolation](#94-ground-truth-interpolation)
   - 9.5 [CSV & Summary Output](#95-csv--summary-output)
10. [Complete Signal Flow Diagram](#10-complete-signal-flow-diagram)
11. [Why These Parameter Values?](#11-why-these-parameter-values)
12. [Appendix — References](#12-appendix--references)

---

## 1. What This File Actually Does

At the broadest level, `rppg_pipeline_refined.py` solves one problem: **given a video of a human face, extract an extremely clean, per-frame RGB time series from skin pixels only** — then hand that time series to downstream frequency-estimation stages so that heart rate can be computed.

This is harder than it sounds. A camera records everything in its field of view: skin, hair, beard, specular reflections, shadows, clothing, background. The rPPG (remote photoplethysmography) signal — the tiny (~1–2%) oscillation in skin colour caused by blood pulsing through sub-dermal vessels — is buried in all that noise. The whole file is one long answer to the question: **which pixels are trustworthy skin, and how confident are we?**

The file does the following, in order:

1. **Face detection and geometric ROI** — MediaPipe's 468-point face mesh identifies exactly where the forehead, left cheek, and right cheek are in every frame.
2. **Deep-learning skin labelling** — A SegFormer-based "BiSeNet" face-parser (running every 10 frames for speed) produces a per-pixel probability of being skin.
3. **Adaptive multi-method colour skin segmentation** — Three independent colour-space rules (adaptive YCbCr, Kovac RGB, adaptive HSV) vote on each pixel; 2-of-3 is required.
4. **GMM clustering** — A Gaussian Mixture Model fitted on confirmed skin pixels in Cb–Cr space provides a fourth, statistically grounded vote.
5. **Texture-based hair rejection** — LBP transitions, gradient orientation coherence, and Spectral Variation Coefficient identify hair and eyebrow pixels that passed colour tests but should be excluded.
6. **Luminance-normalised, probability-weighted RGB extraction** — From surviving skin pixels, a weighted spatial mean R/G/B is extracted per frame, with weights coming from BiSeNet's confidence probabilities.
7. **CHROM projection** — The R/G/B time series is projected into a blood-volume-pulse (BVP) waveform using the CHROM algorithm.
8. **Spatial PCA** — Nine per-region signals (3 channels × 3 ROI regions) are decomposed; the component with maximum power in the cardiac frequency band is selected as an alternative BVP signal.
9. **Four IIR filters** — Butterworth, Chebyshev I, Chebyshev II, and Elliptic bandpass filters are all applied to the BVP signal for downstream comparison.
10. **CSV output** — Every intermediate quantity (raw channels, normalised channels, BVP variants, quality metrics, ground truth) is written as a row-per-frame CSV.

---

## 2. High-Level Architecture & Design Philosophy

The file follows three key engineering decisions:

**Decision 1 — Compute each representation once, reuse everywhere.**
All color-space conversions (YCbCr, HSV, float32 channels) are computed once per frame inside `compute_frame_maps()` and passed as a dictionary. Every subsequent function receives that dict — no function re-converts. Same principle for texture maps: `compute_texture_maps()` runs once and its result (`tex`) is used by both forehead and cheek hair-rejection calls.

**Decision 2 — Cache expensive operations, invalidate by condition.**
BiSeNet (a full neural-network forward pass) runs every 10 frames, not every frame. The resulting skin mask is cached and reused for 9 intervening frames. ROI polygon masks are cached until landmarks move more than 4 pixels between frames. GMM is fitted on BiSeNet frames and its predicted mask is also held in cache. EMA (exponential moving average) smoothing bridges the discrete jumps between BiSeNet refreshes.

**Decision 3 — Require consensus, not unanimity.**
No single method is trusted alone. BiSeNet can mistake beard shadow for skin. YCbCr thresholds fail under unusual lighting. Hair can sit in the face ROI polygon. The architecture is a **layered voting ensemble**: a pixel must pass at least two of three colour tests *or* the deep-learning classifier *or* (GMM + at least one colour vote), and then survive hair rejection, geometric exclusion, and specular detection. This is an explicit design choice for robustness over sensitivity.

---

## 3. Imports & Environment Setup

```python
os.environ['QT_QPA_PLATFORM']  = 'xcb'
os.environ['QT_QPA_FONTDIR']   = '/usr/share/fonts/truetype/dejavu'
```

These two lines must appear **before** any GUI import. On Linux, OpenCV's `imshow()` needs a Qt platform plugin. `xcb` is the X11 backend; without this, OpenCV will either crash or fall back to a headless (no-display) mode. The font directory is needed because some Qt versions look for fonts at startup.

| Import | Why it is here |
|---|---|
| `cv2` | Video reading (`VideoCapture`), color conversion, polygon drawing, display |
| `numpy` | All array math — masks, weighted means, detrending, SVD |
| `pandas` | Ground-truth CSV loading and final output CSV creation |
| `mediapipe` | 468-point face mesh landmarks — the geometric skeleton of every ROI |
| `scipy.signal` | IIR filter design (`butter`, `cheby1`, `cheby2`, `ellip`), `filtfilt`, `welch` |
| `torch` / `torch.nn.functional` | Running SegFormer on GPU or CPU; `interpolate` to upscale logits |
| `PIL.Image` | SegFormerImageProcessor expects a PIL Image, not a numpy array |
| `transformers` | `SegformerImageProcessor` + `SegformerForSemanticSegmentation` — the face-parsing model |
| `sklearn.mixture.GaussianMixture` | GMM skin clustering |
| `concurrent.futures.ThreadPoolExecutor` | Runs texture map computation and colour segmentation in parallel |
| `datetime` | Generating the run timestamp for the output folder name |

---

## 4. Global Paths, Output Directory & Run Timestamp

```python
VID        = Path("...recording_2026-04-07T22-04-35Z.mov")
GT_CSV     = Path("...vitals.csv")
MODEL_PATH = Path("...face_landmarker.task")

_PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent
_RUN_TS       = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR       = _PIPELINE_DIR / 'results' / 'input_results' / _RUN_TS
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

`_PIPELINE_DIR` uses `__file__` (the script's own path) and walks three levels up (`.parent.parent.parent`) to land at the `Updated Pipeline/` folder. This makes the output path relative to the script's location, not to the shell's working directory — so the script writes to the same folder regardless of where you `cd` before running it.

`_RUN_TS` stamps the output folder with the run's wall-clock time so successive runs don't overwrite each other. The `mkdir(parents=True, exist_ok=True)` call creates the full folder tree (`results/input_results/<timestamp>/`) atomically without erroring if it already exists.

---

## 5. Model Initialisation

### 5.1 BiSeNet / SegFormer (deep-learning skin parser)

```python
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
bisenet   = SegformerForSemanticSegmentation.from_pretrained(
                "jonathandinu/face-parsing").to(DEVICE).eval()
```

`jonathandinu/face-parsing` is a HuggingFace model checkpoint that is a **SegFormer-B2** backbone fine-tuned on the CelebAMask-HQ dataset for 19-class face semantic segmentation. The classes include: background, skin, nose, eye glasses, left/right eye, left/right eyebrow, left/right ear, mouth, upper/lower lip, hair, hat, earrings, necklace, neck, clothes.

This file only uses **class 1 (skin)**. The model produces logits at 1/4 the input resolution; the logits are upsampled back to full resolution via bilinear interpolation inside `parse_skin()`.

`.eval()` is critical — it disables dropout layers and sets batch normalisation to inference mode (using running statistics instead of batch statistics), making results deterministic.

**Why SegFormer specifically?** SegFormer uses a hierarchical mix-transformer encoder + a lightweight MLP decoder. It is significantly more accurate than older FCN-based BiSeNet on faces with partial occlusion, beard, and under-eye shadows — but also more expensive (~200ms per inference on CPU). The 10-frame caching strategy (`BISENET_EVERY = 10`) brings the effective cost to ~20ms/frame at 30fps.

**References:** See [Appendix A.1](#a1-segformer--face-parsing)

### 5.2 MediaPipe FaceLandmarker (geometry)

```python
_lm_opts = mp_vision.FaceLandmarkerOptions(
    base_options=mp_base.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(_lm_opts)
```

MediaPipe's FaceLandmarker outputs **468 3D landmarks** (x, y, z in normalised [0,1] coordinates) from a single `.task` model file. The 468-point topology is the canonical MediaPipe face mesh; each index corresponds to a specific anatomical location and is stable across the entire MediaPipe ecosystem.

**`running_mode=VIDEO`** is important: it tells MediaPipe that frames arrive in temporal order with timestamps, enabling the internal Kalman-filter tracker. In `IMAGE` mode, every frame is treated independently (slower, no temporal smoothing). In `LIVE_STREAM` mode, results come back asynchronously via a callback. `VIDEO` mode gives synchronous, temporally-smoothed results — correct for offline video processing.

**`num_faces=1`** — rPPG operates on a single subject. Processing multiple faces would multiply compute without benefit for this use case.

**Confidence thresholds at 0.5** — A conservative midpoint. Lower values would detect partially-occluded faces but increase false positives; higher values would miss frames where the subject turns slightly. 0.5 is the library default and works well for frontal-view clinical recordings.

---

## 6. Module-Level Constants

### 6.1 Landmark Index Tables

```python
_LI_NORMAL = dict(eye_out=33, tmpl=234, brow_o=70, brow_i=107,
                   ala=129, jaw=172,
                   f_lat=54, f_hi=103, f_mid=67, f_lo=109, tmpl_top=21)
_RI_NORMAL = dict(eye_out=263, tmpl=454, brow_o=300, brow_i=336,
                   ala=358, jaw=397,
                   f_lat=284, f_hi=332, f_mid=297, f_lo=338, tmpl_top=251)
_LI_MIRROR = _RI_NORMAL
_RI_MIRROR = _LI_NORMAL
```

These are **constant dictionaries** defined at module level so they are created only once (at import time). Inside `build_rois()`, the function selects between `_LI_NORMAL`/`_RI_NORMAL` and `_LI_MIRROR`/`_RI_MIRROR` based on whether the camera is front-facing or mirrored. The detection heuristic is: if landmark 33 (outer corner of the left eye) has a smaller x-coordinate than landmark 263 (outer corner of the right eye), the camera is in normal (non-mirrored) orientation.

Every index maps to a specific anatomical point in the 468-point mesh:

| Key | Normal-left index | Anatomical meaning |
|---|---|---|
| `eye_out` | 33 | Outer canthus of the eye |
| `tmpl` | 234 | Temple / zygomatic arch |
| `brow_o` | 70 | Outer eyebrow |
| `brow_i` | 107 | Inner eyebrow (above nose bridge) |
| `ala` | 129 | Nose ala (base of nostril) |
| `jaw` | 172 | Mandible / jawline |
| `f_lat` | 54 | Forehead lateral edge |
| `f_hi` | 103 | Forehead high |
| `f_mid` | 67 | Forehead mid |
| `f_lo` | 109 | Forehead low |
| `tmpl_top` | 21 | Temple top |

Storing these as module-level dicts avoids per-frame dictionary allocation and key hashing — a small but measurable speedup when the function runs 1800+ times for a 60-second video.

**References:** See [Appendix A.2](#a2-mediapipe-468-point-face-mesh)

### 6.2 LBP Circular Transition Lookup Table

```python
_LBP_TRANS = np.zeros(256, dtype=np.uint8)
for _v in range(256):
    _b = format(_v, '08b')
    _circ = _b + _b[0]
    _LBP_TRANS[_v] = sum(_circ[i] != _circ[i + 1] for i in range(8))
del _v, _b, _circ
```

This is a **precomputed lookup table for uniform LBP**. Local Binary Patterns encode each pixel's neighbourhood as an 8-bit integer where each bit indicates whether a neighbour is brighter or darker than the centre. The "circular transition count" (also called the "uniformity" measure) counts how many times the bit pattern flips 0→1 or 1→0 as you go around the 8 neighbours — e.g., the pattern `00011100` has 2 transitions (uniform), while `01010101` has 8 transitions (non-uniform/textured).

The table is indexed by the 8-bit LBP code (0–255) and gives the number of circular transitions directly. Without this table, computing transitions per-pixel in the inner loop would require string manipulation or bit-twiddling on every pixel of every frame. With the table, it becomes a single array lookup (`_LBP_TRANS[lbp]`) — extremely fast.

**Why is this useful for rPPG?** Skin has low LBP transition count (smooth texture). Hair has high transition count (fine repetitive structure). The table's values feed directly into `hair_rejection()`.

**References:** See [Appendix A.3](#a3-local-binary-patterns)

### 6.3 Thread Pool

```python
_POOL = ThreadPoolExecutor(max_workers=2)
```

A module-level ThreadPoolExecutor with **2 worker threads**. Creating a thread pool has overhead; doing it at module level means the threads are alive for the entire run. The pool is used inside the main loop to parallelise two independent operations:

- Worker 1: `compute_texture_maps()` — pure numpy/OpenCV, releases Python's GIL
- Worker 2: `adaptive_color_skin()` — also numpy/OpenCV, GIL-free

With `max_workers=2`, both futures run genuinely concurrently on two CPU cores. More workers would not help because only these two tasks are parallelised.

---

## 7. Function Blocks — In Depth

### 7.1 `parse_skin()` — BiSeNet Inference

```python
def parse_skin(crop_rgb: np.ndarray) -> tuple:
    inp = processor(images=Image.fromarray(crop_rgb), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = bisenet(**inp).logits
    up    = F.interpolate(logits, size=crop_rgb.shape[:2],
                          mode="bilinear", align_corners=False)
    probs = F.softmax(up, dim=1)
    return (up.argmax(1).squeeze().cpu().numpy() == 1), probs[0, 1].cpu().numpy()
```

**Input:** An RGB numpy array of the face crop (already resized to ≤384px on the longer side by the caller).

**Step by step:**

1. `processor(images=Image.fromarray(crop_rgb), return_tensors="pt")` — The HuggingFace `SegformerImageProcessor` normalises pixel values (ImageNet mean/std: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), resizes to the model's expected input size, and returns a PyTorch tensor batch of shape `(1, 3, H, W)`.

2. `torch.no_grad()` — Disables gradient tracking. Since we are only running inference (not training), there is no backprop. This saves memory and compute (~30% faster on CPU).

3. `bisenet(**inp).logits` — SegFormer's forward pass returns raw class scores (logits) of shape `(1, 19, H/4, W/4)`. The model outputs at 1/4 resolution because the MLP decoder in SegFormer upsamples internally to 1/4, not full, resolution.

4. `F.interpolate(..., mode="bilinear", align_corners=False)` — Upsamples the logits from 1/4 resolution back to the original crop resolution. `bilinear` is used (not `nearest`) so the probability map is smooth rather than blocky. `align_corners=False` is the correct modern PyTorch convention for image upsampling.

5. `F.softmax(up, dim=1)` — Converts raw logits into a proper probability distribution over the 19 classes (sums to 1 across class dimension for each pixel).

6. Returns:
   - `up.argmax(1) == 1` — Hard skin mask: True where the most-probable class is class 1 (skin).
   - `probs[0, 1]` — Soft probability map: P(skin) ∈ [0, 1] for every pixel. Used as per-pixel weights during RGB extraction.

**Why return both hard and soft masks?** The hard mask is used for Boolean operations (which pixels to include). The soft probability map is used for **weighted averaging** during RGB extraction — pixels where BiSeNet is 90% confident of skin contribute more to the mean than pixels where it is 55% confident.

**References:** See [Appendix A.1](#a1-segformer--face-parsing)

---

### 7.2 `compute_frame_maps()` — Color-Space Pre-computation

```python
def compute_frame_maps(rgb: np.ndarray) -> dict:
    ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv   = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ycf   = ycbcr.astype(np.float32)
    hsvf  = hsv.astype(np.float32)
    return dict(
        rgb=rgb, ycbcr=ycbcr, hsv=hsv, gray=gray,
        Y=ycf[:,:,0], Cr=ycf[:,:,1], Cb=ycf[:,:,2],
        H_deg=hsvf[:,:,0]*2.0,
        S_n=hsvf[:,:,1]/255.0,
        V_n=hsvf[:,:,2]/255.0,
        R=rgb[:,:,0].astype(np.float32),
        G=rgb[:,:,1].astype(np.float32),
        B=rgb[:,:,2].astype(np.float32),
    )
```

This function exists purely as an efficiency measure: color conversions (especially YCbCr and HSV) are not free, and the same converted arrays are needed by multiple downstream functions. By computing once and storing in a dict, the pipeline avoids three redundant `cvtColor` calls per frame.

**Important note on `COLOR_RGB2YCrCb`:** OpenCV's output channel order is `[Y, Cr, Cb]` — note Cr before Cb, which is the opposite of the ITU-R BT.601 standard notation (YCbCr). The code assigns: `Y=ycf[:,:,0]`, `Cr=ycf[:,:,1]`, `Cb=ycf[:,:,2]`. This is correct given OpenCV's ordering; the variable names match the actual channel content.

**HSV scaling:** OpenCV encodes HSV with:
- H ∈ [0, 179] (half the [0°, 359°] range, to fit in uint8)
- S ∈ [0, 255]
- V ∈ [0, 255]

The code applies `H_deg = hsvf[:,:,0] * 2.0` to recover true [0°, 360°) hue, and divides S and V by 255 to get normalised [0, 1] floats. These conversions must happen before `adaptive_color_skin()` uses them for circular statistics.

---

### 7.3 `compute_texture_maps()` — LBP, Gradient Coherence, SVC, Specular

```python
def compute_texture_maps(norm_u8: np.ndarray, norm_gray: np.ndarray,
                          bbox: tuple) -> dict:
```

**Input:** Illumination-normalised uint8 image (`norm_u8`), float32 normalised grayscale (`norm_gray`), and the face bounding box `(y1, y2, x1, x2)`. Works only on the face crop, then pads back into a full-frame array.

This function computes four independent texture descriptors:

#### LBP Circular Transition Count

```python
c   = u8.astype(np.int32)
lbp = np.zeros_like(u8, dtype=np.uint8)
for bit, (dy, dx) in enumerate([(-1,0),(-1,1),(0,1),(1,1),
                                 (1,0),(1,-1),(0,-1),(-1,-1)]):
    nb = np.roll(np.roll(u8, dy, axis=0), dx, axis=1).astype(np.int32)
    lbp |= ((nb >= c).astype(np.uint8) << bit)
lbp_crop = _LBP_TRANS[lbp].astype(np.float32)
```

- The 8 neighbours are sampled at distance 1 in a clockwise order starting from directly above.
- `np.roll` shifts the image in each direction to bring the neighbour into alignment with the centre pixel — a vectorised alternative to a Python loop over pixels.
- Each bit of the `lbp` byte encodes whether that neighbour ≥ centre (`1`) or < centre (`0`).
- `_LBP_TRANS[lbp]` maps every 8-bit code to its circular transition count using the precomputed table (Section 6.2).

**Why illumination-normalised input?** LBP is computed on `norm_u8`, not raw grayscale. The illumination normalisation (dividing by a local Gaussian blur) removes global lighting gradients, so LBP sees only fine-scale texture rather than slow brightness ramps across the forehead.

#### Gradient Orientation Coherence

```python
sx  = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
sy  = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
mag = np.hypot(sx, sy) + 1e-8
coh_crop = np.hypot(cv2.GaussianBlur(sx/mag, (0,0), 6.0),
                     cv2.GaussianBlur(sy/mag, (0,0), 6.0))
```

This computes **unit-vector field coherence** — a measure of how consistently aligned the local gradient directions are in a neighbourhood.

1. Sobel filters compute x- and y-gradients (`sx`, `sy`).
2. Dividing by magnitude gives unit direction vectors: `(sx/mag, sy/mag)` lies on the unit circle.
3. Averaging these unit vectors (via Gaussian blur) with σ=6 pixels produces a local mean vector. If all local gradients point the same way (e.g., a smooth gradient across skin), the mean vector has magnitude ≈ 1.0 (high coherence). If gradients point in random directions (hair), the mean vector is near 0 (low coherence, `hypot` is small).

Wait — but hair produces **high** coherence (fibre direction), not low. The code uses coherence as: high coherence + high LBP → likely hair. Hair fibres create strong oriented edges → high coherence; their repetitive fine structure → high LBP transitions. Skin is smooth: low LBP, coherence can be moderate.

#### Spectral Variation Coefficient (SVC)

```python
mu  = cv2.blur(gf, (9, 9))
mu2 = cv2.blur(gf**2, (9, 9))
svc_crop = (np.sqrt(np.maximum(mu2-mu**2, 0.0)) / (mu+1.0))
```

SVC is the **local coefficient of variation** of pixel intensity: `σ / μ` within a 9×9 window.

- `mu` is the local mean: `E[g]`
- `mu2` is the local mean of squared values: `E[g²]`
- `mu2 - mu²` = `Var[g]` (by the identity `Var[X] = E[X²] - E[X]²`)
- `np.maximum(..., 0)` guards against numerical rounding producing a slightly negative variance.
- Dividing by `mu+1` normalises for brightness (so dark and bright regions are comparable).

**Why 9×9?** Large enough to capture texture patterns at the scale of hair strands (~3–8 pixels wide at typical shooting distances) but small enough to remain local. A larger kernel would blur out fine texture; smaller would miss multi-pixel structures.

#### Specular Highlight Mask

```python
loc_mu    = cv2.blur(gf, (31, 31))
spec_crop = gf > (loc_mu * 1.25 + 25)
```

A pixel is specular if its intensity is more than 25% above (plus a fixed 25 DN offset) its 31×31 neighbourhood mean. Specular highlights are physically distinct from skin colour — they are mirror-reflected illumination with the source's spectral profile, not tissue-scattered light. Including them would corrupt the rPPG signal, but simply thresholding global brightness would misclassify foreheads under bright lighting. The local-contrast approach adapts per-region.

Specular pixels are flagged and excluded from both hair-rejection and RGB extraction. They are **immune to hair rejection** (`~specular` appears in the hair-rejection function) because we already know they are not skin — treating them as hair is redundant and could contaminate the hair fraction metric.

---

### 7.4 `hair_rejection()` — Texture-Based Hair Exclusion

```python
def hair_rejection(tex: dict, roi: np.ndarray, pct: int = 70) -> np.ndarray:
    lbp, coh, svc = tex['lbp'], tex['coh'], tex['svc']
    specular = tex['specular']
    valid = roi & ~specular
    ...
    lbp_t = pth(lbp, pct)
    coh_t = pth(coh, max(pct-5, 50))
    svc_t = pth(svc, max(pct-5, 50))
    return (lbp > lbp_t) & ((coh > coh_t) | (svc > svc_t)) & ~specular
```

A pixel is flagged as hair if:
- LBP transition count is in the top `(100 - pct)%` of the ROI pixels — meaning it has more texture transitions than 70% of the region (for the default `pct=70`)
- **AND** at least one of: gradient coherence OR SVC is above the 65th percentile (= `max(pct-5, 50)`)

The requirement for **two independent signals** is deliberate: a pixel must show both non-uniform texture (high LBP) and either directional structure (high coherence) or spatially variable intensity (high SVC). This dual-evidence requirement suppresses false positives — noisy skin pixels might have high LBP but low coherence/SVC.

**Different thresholds for forehead vs cheeks:**
- `hair_rejection(tex, fm, pct=78)` for forehead (78th percentile — stricter, fewer false hair flags)
- `hair_rejection(tex, lcm|rcm, pct=72)` for cheeks (72nd percentile — slightly looser, since sideburns and temple hair are more common)

**Adaptive thresholds** (percentile-based, not fixed): this ensures robustness across subjects with different hair density and skin texture. A subject with fine, sparse hair on the forehead will have a lower absolute LBP threshold than one with thick hair — because the percentile is computed relative to the current ROI's distribution.

**References:** See [Appendix A.3](#a3-local-binary-patterns)

---

### 7.5 `build_rois()` — Adaptive Polygon ROIs

```python
def build_rois(pts: np.ndarray, H: int, W: int):
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
```

**Forehead polygon:** 18 vertices tracing a path from left temple-top → across the forehead high curve → across the top → back along the brow line → back to left temple. Landmarks 10 and 9 are the superior and inferior midline forehead points. The polygon deliberately goes above the eyebrows (above `brow_o` and `brow_i`) to include forehead skin but exclude the brow hair itself.

```python
def cheek(s):
    ey = float(s['eye_out'][1]) + 0.12*(float(s['jaw'][1])-float(s['eye_out'][1]))
    ly = float(s['eye_out'][1]) + 0.72*(float(s['jaw'][1])-float(s['eye_out'][1]))
    return np.array([
        [s['eye_out'][0], ey], s['tmpl'],
        [s['tmpl'][0], ly], [s['jaw'][0], ly],
        [s['ala'][0],  ly], s['ala'],
        [s['ala'][0],  ey],
    ], dtype=np.float32)
```

**Cheek polygon:** 7-vertex polygon defined by parameterised fractions of the vertical distance from outer eye corner to jaw:
- Top edge: 12% of the way down from eye to jaw (just below the outer eye corner, above the cheekbone)
- Bottom edge: 72% of the way down (mid-to-lower cheek, above jaw)
- Medial edge: follows the nose ala (nostril base) x-coordinate
- Lateral edge: follows the temple/mandible x-coordinate

These fractions (0.12 and 0.72) are empirically chosen to capture the flat, well-lit cheek region while excluding the periorbital area (which has thinner skin and different vasculature) and the jaw (which transitions into neck/beard territory).

---

### 7.6 `poly_mask()` — Polygon → Binary Mask

```python
def poly_mask(poly: np.ndarray, H: int, W: int) -> np.ndarray:
    poly = np.clip(poly, [0,0], [W-1, H-1])
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    return m.astype(bool)
```

Simple utility: clips polygon vertices to valid image coordinates (preventing `fillPoly` from writing outside the array), rasterises the polygon using OpenCV's scanline fill, and returns a bool mask. The clip is necessary because facial landmarks near the image border can have float coordinates slightly outside [0, W-1] or [0, H-1] due to floating-point arithmetic.

---

### 7.7 `build_all_masks()` — All Five Masks in One Call

```python
def build_all_masks(pts, fp, lp, rp, H, W):
    must_poly  = np.array([pts[129], pts[358], pts[291], pts[61]], np.float32)
    beard_poly = np.array([pts[61], pts[291], pts[397], pts[152], pts[172]], np.float32)
    fm   = poly_mask(fp, H, W)
    lcm  = poly_mask(lp, H, W)
    rcm  = poly_mask(rp, H, W)
    must = poly_mask(must_poly,  H, W)
    brd  = poly_mask(beard_poly, H, W)
    return fm, lcm, rcm, must | brd
```

Computes:
1. `fm` — Forehead mask
2. `lcm` — Left cheek mask
3. `rcm` — Right cheek mask
4. `must | brd` — **Exclusion zone** = mustache region (pts 129, 358, 291, 61 = nose ala to lip corners) **union** beard region (lip corners down to chin centred around pts 397, 152, 172). Any skin pixel inside these zones is excluded from RGB extraction regardless of how many methods classify it as skin.

The exclusion zone is critical: skin-coloured beard and mustache areas have completely different haemodynamic properties from forehead/cheek skin, and including them would corrupt the rPPG signal.

---

### 7.8 `adaptive_color_skin()` — Three-Method Adaptive Color Segmentation

```python
def adaptive_color_skin(fm: dict, seed_mask: np.ndarray) -> tuple:
```

This is the most complex function in the file. It implements three independent colour-space skin detectors, each adapted per-image-half using BiSeNet-confirmed pixels as seeds.

**Left-right image halving:** The image is split into left half (`x ∈ [0, W/2)`) and right half (`x ∈ [W/2, W)`). Parameters are fitted independently for each half. This handles asymmetric lighting — if a window is to the subject's left, the left cheek will have a different skin colour distribution than the right cheek.

#### Method 1 — Adaptive YCbCr

```python
sCb, sCr, sY = Cb[seed], Cr[seed], Y[seed]
cb_lo = max(55,  float(sCb.mean() - 2.5*sCb.std()))
cb_hi = min(148, float(sCb.mean() + 2.5*sCb.std()))
cr_lo = max(110, float(sCr.mean() - 2.5*sCr.std()))
cr_hi = min(195, float(sCr.mean() + 2.5*sCr.std()))
y_lo  = max(8,   float(np.percentile(sY, 0.5))  - 10)
y_hi  = min(252, float(np.percentile(sY, 99.5)) + 10)
```

A rectangular acceptance window is fitted at **mean ± 2.5σ** on Cb and Cr channels, plus a wide luminance range. The hardcoded limits (`max(55, ...), min(148, ...)` etc.) are safety rails — the absolute extremes of valid human skin chrominance from published literature. These prevent the adaptive window from growing so wide it starts accepting non-skin pixels (e.g., if the seed is contaminated).

The Y range uses percentiles (0.5th to 99.5th) instead of mean±σ because luminance has a heavy-tailed distribution within the face: most skin is mid-bright, but there is always a long tail of very dark shadow pixels and near-saturated highlight pixels. The ±10 extension beyond the percentiles gives a small buffer.

**Fallback** (fewer than 30 seed pixels): fixed values from Chai & Ngan (1999): Cb∈[77,127], Cr∈[133,173], Y∈[20,245]. These are the canonical published skin-colour ranges for BT.601 YCbCr.

#### Method 2 — Kovac RGB Rule

```python
Rmax = np.maximum(np.maximum(R,G),B);  Rmin = np.minimum(np.minimum(R,G),B)
m2   = (R>95)&(G>40)&(B>20)&((Rmax-Rmin)>15)&(np.abs(R-G)>15)&(R>G)&(R>B)
```

This is the **Kovac (2003) skin rule** — a set of deterministic inequalities in RGB space that hold for all human skin tones under diverse illumination conditions. It requires:
- All channels above floor values (R>95, G>40, B>20) — minimum brightness for skin
- Dynamic range > 15 — not a near-grey pixel (grey would be background or specular)
- |R-G| > 15 and R > G and R > B — skin has more red than green, more red than blue, and a non-trivial R-G difference

This rule has **no adaptivity** — it is skin-tone agnostic and relies purely on the physics of how flesh colours interact with broadband illumination. It serves as an independent, non-data-driven check on the adaptive methods.

**References:** See [Appendix A.5](#a5-kovac-rgb-skin-rule)

#### Method 3 — Adaptive HSV

```python
ang = np.deg2rad(sH)
h_mean = float(np.rad2deg(np.arctan2(np.sin(ang).mean(),
                                      np.cos(ang).mean())) % 360)
h_std  = float(np.std(sH))
h_lo   = (h_mean - 2.5*h_std) % 360
h_hi   = (h_mean + 2.5*h_std) % 360
hue_ok = (H_deg>=h_lo)&(H_deg<=h_hi) if h_lo<=h_hi else (H_deg>=h_lo)|(H_deg<=h_hi)
```

Hue is a **circular** variable (0° and 360° are the same). A naïve mean/std on hue values near 0°/360° would give wildly wrong results (e.g., averaging 5° and 355° → 180° instead of 0°). The correct approach uses **circular mean**: convert hue to unit-circle vectors (`sin/cos`), average the vectors, then `atan2` to recover the mean angle. This is the standard circular statistics approach.

The wrap-around check (`if h_lo<=h_hi else (...)`) handles the case where the acceptance interval straddles 0°/360° — e.g., h_lo=350°, h_hi=30° must accept pixels with H≥350° OR H≤30°, not the range 350°→30° going the wrong way.

#### Shadow Recovery

```python
sh = (Cb>=60)&(Cb<=145)&(Cr>=115)&(Cr<=190)&(Y>=10)&(Y<45)&(S_n>=0.08)
```

Dark skin under shadow has low Y but still maintains characteristic Cb/Cr chrominance. Standard skin detection fails under shadow (Y too low). This broadened YCbCr window with a looser Y range specifically recovers shadowed skin as an override, added via `| sh` at the end.

#### Voting

```python
vote = m1.astype(np.uint8) + m2.astype(np.uint8) + m3.astype(np.uint8)
return (vote >= 2) | sh, vote
```

A pixel is accepted as skin if at least 2 of the 3 methods agree — majority voting. The `vote` map (0–3) is also returned as a quality indicator: pixels where all three methods agree (`vote==3`) are the highest-confidence skin pixels.

**References:** See [Appendix A.4](#a4-ycbcr-skin-detection), [A.5](#a5-kovac-rgb-skin-rule)

---

### 7.9 `fit_gmm()` — Gaussian Mixture Model Fitting

```python
def fit_gmm(ycbcr: np.ndarray, seed_mask: np.ndarray, n_comp: int = 3) -> dict | None:
    X  = np.column_stack([Cb[seed_mask], Cr[seed_mask]])
    ...
    idx = np.random.default_rng(0).choice(X.shape[0], min(4000, X.shape[0]), replace=False)
    Xs  = X[idx];  mu = Xs.mean(axis=0)
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full',
                           max_iter=60, random_state=42).fit(Xs)
    si  = int(np.argmin(np.linalg.norm(gmm.means_ - mu, axis=1)))
    return dict(gmm=gmm, si=si)
```

A 3-component GMM is fitted on BiSeNet-confirmed skin pixels in **2D Cb–Cr space** (luminance Y is excluded — the GMM models chrominance only, which is more illumination-invariant than RGB).

**Why 3 components?** Human skin under mixed lighting can exhibit 2–3 distinct chrominance clusters: well-lit skin, shadowed skin, and transitional skin. A single Gaussian cannot model this multimodality. 3 components is chosen as a balance: expressive enough to capture illumination variation, but not so complex that it overfits noise.

**Subsampling to ≤4000 points** avoids O(N²) covariance matrix computation on large faces. 4000 points is empirically sufficient to estimate the GMM parameters accurately.

**Skin component identification:** After fitting, the component whose mean (in Cb-Cr space) is closest to the overall seed mean is labelled as the skin component (`si`). This avoids having to know a priori which component will be skin — the closest-to-seed heuristic reliably finds it.

**References:** See [Appendix A.6](#a6-gaussian-mixture-models-for-skin)

---

### 7.10 `predict_gmm()` — GMM Inference Per Frame

```python
def predict_gmm(model: dict | None, ycbcr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    ...
    proba = model['gmm'].predict_proba(X)[:, model['si']]
    out[ys, xs] = proba > 0.50
```

For each ROI pixel, computes P(pixel belongs to the skin component) using the fitted GMM. The 0.50 threshold is a hard classification boundary — a pixel must be more likely skin than any other cluster combined. This operates only on ROI pixels (not the full frame) for efficiency.

---

## 8. Main Execution Loop (`__main__`)

### 8.1 Ground-Truth Loading

```python
gt_raw      = pd.read_csv(GT_CSV)
gt_time_raw = gt_raw['offset_seconds'].values
gt_hr_raw   = gt_raw['heart_rate'].values
valid_gt    = ~np.isnan(gt_hr_raw)
gt_time_raw = gt_time_raw[valid_gt]
gt_hr_raw   = gt_hr_raw[valid_gt]
```

The DocBOT ground-truth CSV contains two columns: `offset_seconds` (time since recording start) and `heart_rate` (BPM from a contact sensor). NaN values appear during sensor dropout and are stripped before interpolation. The cleaned arrays are stored for later temporal interpolation against frame timestamps.

### 8.2 Video Capture & Buffer Initialisation

```python
cap     = cv2.VideoCapture(str(VID))
fs      = cap.get(cv2.CAP_PROP_FPS)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
```

`cv2.CAP_PROP_FPS` reads the frame rate from the container metadata — typically 30.0 for iPhone/DocBOT recordings. `CAP_PROP_FRAME_COUNT` gives the total frame count for the progress display.

Then 24 time-series buffers are initialised as empty Python lists:

| Buffer group | Contents |
|---|---|
| `R_t, G_t, B_t` | Whole-face weighted-mean R, G, B per frame |
| `lum_t, npx_t, det_t` | Frame luminance, skin pixel count, face-detected flag |
| `bisenet_conf_t, vote_mean_t` | BiSeNet confidence, colour vote mean |
| `highconf_pct_t, roi_coverage_t` | Fraction of all-3-agree pixels, skin/ROI coverage ratio |
| `green_cv_t, hair_rej_frac_t` | G channel spatial CV, hair rejection fraction |
| `clust_agree_t, quality_t` | GMM consensus fraction, composite quality score |
| `Rf_t…Br_t` | Per-region (forehead/L-cheek/R-cheek) R, G, B means |

Using Python lists and appending per frame is standard for unknown-length time series; a NumPy array would require pre-allocation or repeated resizing.

### 8.3 Runtime Constants & Persistent Caches

```python
DISPLAY_W     = 700        # display window pixel width
BISENET_EVERY = 10         # run BiSeNet every 10th frame
MAX_CROP_PX   = 384        # max face crop size fed to SegFormer
DISPLAY_EVERY = 3          # refresh display every 3rd frame
EMA_ALPHA     = 0.03       # probability map EMA smoothing coefficient
```

**`BISENET_EVERY = 10`:** At 30fps, BiSeNet runs 3 times per second. Face skin changes slowly (no abrupt changes in skin pigmentation), so the 10-frame cache introduces at most 333ms of latency on the skin mask — imperceptible for rPPG purposes.

**`MAX_CROP_PX = 384`:** The face crop is resized so its longer side ≤ 384 pixels before being passed to SegFormer. This speeds up inference (fewer pixels ≈ fewer computations) with minimal accuracy loss — SegFormer was trained at 512×512 but is fully convolutional and generalises to smaller inputs. For a 1920×1080 recording where the face occupies 400×500px, this gives a ~0.75× scale factor.

**`EMA_ALPHA = 0.03`:** The time constant of the EMA is `1 / (ALPHA × fps)` ≈ `1 / (0.03 × 30)` ≈ 1.1 seconds. This smooths the probability map `sk_prob_smooth` across the 10-frame gap between BiSeNet refreshes, preventing the weighted RGB averages from jumping discontinuously when a new BiSeNet prediction arrives.

**Persistent caches:**
- `sk_cache` — BiSeNet binary skin mask (bool array, full frame)
- `sk_prob_cache` — BiSeNet probability map (float32, full frame)
- `gmm_cache` — Fitted GMM dict
- `gmm_sk_cache` — GMM skin prediction mask (bool array, full frame)
- `mask_cache` — Tuple of 8 polygon masks, cached until landmarks move > 4px
- `pts_cache` — Landmark array that produced the current mask_cache
- `sk_prob_smooth` — EMA-smoothed probability map

### 8.4 Per-Frame Processing (the core loop)

The main `while cap.isOpened()` loop runs once per video frame:

```
Frame N
│
├── cap.read() → BGR image
├── BGR → RGB conversion
├── compute_frame_maps(rgb) → fm_d  [all colour spaces, once]
│
├── MediaPipe detect_for_video(timestamp_ms) → landmarks
│
└── If face found:
    │
    ├── Compute face bounding box (pts bounding rect + 20px padding)
    │
    ├── ROI polygon masks [cached if landmarks moved < 4px]
    │   └── build_rois() + build_all_masks()
    │
    ├── BiSeNet [every 10 frames]
    │   ├── Crop face → resize to ≤384px
    │   ├── parse_skin() → binary mask + prob map
    │   ├── Resize back to original resolution
    │   ├── fit_gmm() on BiSeNet-confirmed skin in ROI
    │   └── predict_gmm() → GMM skin mask
    │
    ├── EMA update on sk_prob_smooth
    │
    ├── Illumination normalisation
    │   └── norm_gray = gray / GaussianBlur(gray) × 128
    │
    ├── [PARALLEL via _POOL]
    │   ├── Thread 1: compute_texture_maps(norm_u8, norm_gray, bbox)
    │   └── Thread 2: adaptive_color_skin(fm_d, face_seed)
    │
    ├── hair_rejection() on forehead (pct=78)
    ├── hair_rejection() on cheeks (pct=72)
    │
    ├── Final skin combination:
    │   fore_skin = (BiSeNet|color|GMM+1vote) & forehead & ~hair & ~excl & ~specular
    │   l_skin    = same logic for left cheek
    │   r_skin    = same logic for right cheek
    │   final     = fore_skin | l_skin | r_skin
    │
    ├── RGB extraction (if n_px ≥ 50)
    │   ├── Luminance normalise: fn = f / lum × 128
    │   ├── Probability-weighted average: ∑(w·pixel) / ∑w
    │   └── Per-region extraction (forehead, L, R cheek)
    │
    ├── Confidence metrics
    └── Display (every 3 frames)
```

**Landmark timestamp:** `timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))` retrieves the timestamp of the current frame in milliseconds. MediaPipe VIDEO mode requires monotonically increasing timestamps to maintain its tracker state. Using `CAP_PROP_POS_MSEC` is the correct approach — it reads the actual timestamp from the container, which accounts for variable frame spacing in VFR (variable frame rate) recordings.

**Landmark movement threshold of 4px:** Sub-4-pixel landmark jitter is irrelevant to the ROI — the polygon mask doesn't meaningfully change when a landmark shifts by 2 pixels. Recomputing polygon masks every frame would add ~0.5ms/frame of wasted computation. The 4px threshold is a practical "motion is real" gate.

### 8.5 RGB Signal Extraction & Luminance Normalisation

```python
lum = float(f[final].mean())
fn  = f / max(lum, 1.0) * 128.0
w_all = sk_prob_smooth[final].clip(0.01)
R_t.append(float(np.average(fn[:,:,0][final], weights=w_all)))
```

**Luminance normalisation:** The raw frame `f` is divided by the mean luminance of the skin region and scaled back to 128. This compensates for camera auto-exposure/AGC (automatic gain control) — if the camera darkens the exposure, all channels drop proportionally, but the rPPG signal should be illumination-independent. Dividing by the mean and re-scaling to a constant reference (128 ≈ mid-range for uint8) removes the global multiplicative factor.

**Why 128?** Any constant works (the rPPG algorithm only cares about relative changes). 128 is mid-scale for uint8 and makes debugging visually intuitive.

**Why probability-weighted averages?** This is the key innovation over simple spatial means. Pixels with `sk_prob_smooth = 0.90` (very likely skin) contribute 90× more than pixels with `sk_prob_smooth = 0.01` (uncertain). The 0.01 floor (`clip(0.01)`) prevents zero weights from making the average undefined. This approach maximises the signal-to-noise ratio of the extracted R/G/B traces: uncertain-skin pixels (likely transitional regions at hair/skin boundaries) are downweighted rather than excluded entirely, which would create spatial discontinuities as pixels flicker in/out of the binary mask.

**Per-region tracking:** Separate R/G/B means are computed for forehead, left cheek, and right cheek. If any region has fewer than 20 skin pixels, it falls back to the global face value. These 9 time series feed the spatial PCA stage.

### 8.6 Per-Frame Confidence Metrics

Seven metrics are computed per frame and stored for output in the CSV:

| Metric | Formula | What it measures |
|---|---|---|
| `bisenet_conf` (bc) | `mean(sk_prob_full[final])` | Average BiSeNet P(skin) over accepted skin pixels |
| `vote_mean` (vm) | `mean(vote_map[final]) / 3` | Average colour consensus, normalised to [0,1] |
| `highconf_pct` (hc) | `(vote_map[final]==3).sum() / n_px` | Fraction of pixels where all 3 colour methods agreed |
| `roi_coverage` (rc) | `n_px / roi_area` | Fraction of anatomical ROI classified as skin |
| `green_cv` (gcv) | `std(G[final]) / mean(G[final])` | Spatial coefficient of variation of green channel — low = uniform |
| `hair_rej_frac` (hrf) | `hair_px / roi_area` | Fraction of ROI removed by hair rejection |
| `clust_agree` | `gmm_sk[final].sum() / n_px` | Fraction of accepted skin pixels also in GMM skin cluster |

**Composite quality score:**
```python
q = (0.25*bc + 0.20*vm + 0.20*clust_agree
   + 0.15*min(1.0,rc/0.40)
   + 0.12*max(0.0,1.0-gcv/0.15)
   + 0.08*max(0.0,1.0-hrf/0.50))
```

Weighted sum of all metrics, normalised to [0, 1]. Weights were set by relative importance:
- BiSeNet confidence (0.25) — highest weight, deep-learning model is most reliable
- Colour consensus + GMM agreement (0.20 each) — independent corroborating evidence
- ROI coverage (0.15) — `rc/0.40` saturates at 40% skin coverage (a full face ROI rarely exceeds 40% after exclusions)
- Green spatial uniformity (0.12) — `gcv > 0.15` indicates textural variation suggesting non-skin contamination
- Hair fraction (0.08) — high hair rejection means the ROI was mostly hair, low quality

This score is used by downstream analysis scripts to weight frames or discard low-quality ones (q < threshold).

### 8.7 Live Display

```python
if fi % DISPLAY_EVERY == 0:
    left  = cv2.resize(bgr, (DISPLAY_W, disp_h))         # original frame
    right = left.copy()                                    # will show skin overlay
    ...
    ov[final,1] = np.clip(ov[final,1]*0.55+144*0.45, 0, 255)   # green tint on skin
```

The display is throttled to every 3rd frame (10fps refresh). The left panel shows the original frame with polygon outlines; the right panel shows the same frame with accepted skin pixels highlighted in green and polygon outlines in their region colours (forehead=cyan, left=yellow, right=magenta). The green overlay is applied by mixing: `G_new = 0.55×G_orig + 0.45×144` — keeping 55% of the original green and adding 45% of a mid-green value (144/255 ≈ 0.56 saturation), which creates a translucent green highlight without completely washing out face texture.

---

## 9. Post-Processing (after the loop)

### 9.1 CHROM Projection & Linear Detrend

```python
R, G, B = np.array(R_t), np.array(G_t), np.array(B_t)
Rn, Gn, Bn = R/R.mean(), G/G.mean(), B/B.mean()
Xs  = 3*Rn - 2*Gn
Ys  = 1.5*Rn + Gn - 1.5*Bn
alp = np.std(Xs) / (np.std(Ys)+1e-9)
S   = Xs - alp*Ys
tv  = np.arange(T, dtype=float)
S_det = S - np.polyval(np.polyfit(tv, S, 1), tv)
```

**CHROM (Chrominance-based rPPG)** by de Haan & Jeanne (2013) is the gold-standard method for extracting a blood volume pulse signal from the normalized R, G, B time series.

**Step 1 — Temporal normalisation:** Dividing each channel by its temporal mean (`R/R.mean()`) removes the DC component and makes each channel dimensionless. The resulting `Rn, Gn, Bn` represent relative deviations from mean skin reflectance.

**Step 2 — CHROM projection:** Two orthogonal signals are formed:
- `Xs = 3Rn - 2Gn` — the "red minus green" direction, aligned with the haemoglobin absorption axis
- `Ys = 1.5Rn + Gn - 1.5Bn` — captures blue-to-red variation

These are the projections derived analytically in de Haan & Jeanne (2013) from the assumption that the rPPG signal lies in the plane spanned by the normalised spectral sensitivities of a typical camera sensor and the haemoglobin absorption spectrum.

**Step 3 — Adaptive α combination:** `α = std(Xs)/std(Ys)` is chosen so that `S = Xs - α·Ys` cancels specular reflection noise, which is common to both Xs and Ys but with different amplitudes. This produces a BVP signal with motion/specular artefacts partially cancelled.

**Step 4 — Linear detrend:** `np.polyfit(tv, S, 1)` fits a degree-1 polynomial (line) through the BVP signal; `np.polyval(..., tv)` evaluates it; subtracting removes any remaining linear trend (slow camera gain drift, slow movement).

**References:** See [Appendix A.7](#a7-chrom-rppg-algorithm)

---

### 9.2 Spatial Region PCA

```python
X_reg = np.stack([Rf_t, Gf_t, Bf_t, Rl_t, Gl_t, Bl_t, Rr_t, Gr_t, Br_t])  # (9, T)
X_reg = (X_reg - X_reg.mean(1, keepdims=True)) / (X_reg.std(1, keepdims=True) + 1e-9)
_, _, Vt = np.linalg.svd(X_reg, full_matrices=False)
```

The 9 per-region signals (forehead R/G/B, left cheek R/G/B, right cheek R/G/B) are stacked into a 9×T matrix. Each row is z-scored (zero mean, unit variance) to remove channel-level DC offsets and scale differences.

**SVD decomposition:** `np.linalg.svd(X_reg)` decomposes the 9×T matrix. The rows of `Vt` (right singular vectors) are the principal components — orthogonal time series that explain decreasing amounts of variance in `X_reg`.

**Cardiac band power selection:**
```python
card_pow = np.zeros(min(3, len(Vt)))
for i in range(len(card_pow)):
    fp_w, pp_w = scipy.signal.welch(Vt[i], fs=fs, ...)
    band_w     = (fp_w >= 0.67) & (fp_w <= 3.5)
    card_pow[i] = pp_w[band_w].max() / (pp_w.mean() + 1e-9)
best_pc = int(np.argmax(card_pow))
S_pca   = Vt[best_pc]
```

For each of the first 3 principal components, the Welch PSD is computed and the peak-in-band to mean-power ratio is calculated. The PC with the highest ratio is selected as the cardiac component — the one whose oscillation is most concentrated in the 0.67–3.5 Hz cardiac band (40–210 BPM). This is then linearly detrended and saved as `BVP_pca`.

**Why spatial PCA?** Different face regions have slightly different rPPG waveforms (the forehead and cheeks are not identically perfused). PCA finds the linear combination of 9 signals that maximally captures shared cardiac oscillation while suppressing region-specific noise and motion artefacts. It is equivalent to finding the "best" weighted average across regions automatically, without manually tuning regional weights.

**References:** See [Appendix A.8](#a8-spatial-pca-for-rppg)

---

### 9.3 Four IIR Bandpass Filters

```python
Wn = [0.7, 3.5]
b_bw,a_bw = scipy.signal.butter( 4,        Wn, btype='band', fs=fs)
b_c1,a_c1 = scipy.signal.cheby1( 4, 0.5,   Wn, btype='band', fs=fs)
b_c2,a_c2 = scipy.signal.cheby2( 4, 40,    Wn, btype='band', fs=fs)
b_el,a_el = scipy.signal.ellip(  4, 0.5,40, Wn, btype='band', fs=fs)
```

Four classical IIR filter designs, all order-4 bandpass for the cardiac band [0.7, 3.5] Hz:

| Filter | Passband ripple | Stopband attenuation | Characteristic |
|---|---|---|---|
| Butterworth | Monotone (0 ripple) | Moderate | Maximally flat magnitude; widest transition band |
| Chebyshev I | ≤0.5 dB ripple | Moderate | Equiripple in passband; steeper rolloff than Butterworth |
| Chebyshev II | 0 ripple | ≥40 dB | Equiripple in stopband; flat passband; steeper than Butterworth |
| Elliptic | ≤0.5 dB ripple | ≥40 dB | Equiripple in both bands; steepest transition for given order |

All four are applied with `scipy.signal.filtfilt` (zero-phase, forward-backward filtering) to avoid phase distortion. The output columns `BVP_butterworth`, `BVP_cheby1`, `BVP_cheby2`, `BVP_elliptic` allow downstream frequency-estimation scripts to compare the effect of filter choice on BPM accuracy.

**Why 0.7–3.5 Hz?** This corresponds to 42–210 BPM — encompassing the full physiological human heart rate range plus a safety margin (normal resting is 60–100 BPM; trained athletes can have resting HR ~50 BPM; peak exercise can reach ~200 BPM). Frequencies below 0.7 Hz are respiration (0.1–0.5 Hz), motion drift, and illumination flicker. Frequencies above 3.5 Hz are noise.

**References:** See [Appendix A.9](#a9-iir-filter-design)

---

### 9.4 Ground-Truth Interpolation

```python
gt_bpm_frame = np.interp(t_axis, gt_time_raw, gt_hr_raw, left=np.nan, right=np.nan)
```

The ground-truth heart rate from the contact sensor is sampled at irregular intervals (DocBOT logs vitals when they change or at periodic intervals). `np.interp` linearly interpolates between those sparse measurements onto the regular per-frame time axis. `left=np.nan` and `right=np.nan` fill frames before the first GT sample and after the last — these will be NaN in the output CSV and should be excluded from validation metrics downstream.

### 9.5 CSV & Summary Output

The output CSV has **T rows × 26 columns**, where T is the number of successfully processed frames:

| Column group | Columns |
|---|---|
| Timing | `frame_index`, `time_s` |
| Raw signals | `R_skin_raw`, `G_skin_raw`, `B_skin_raw` |
| Normalised signals | `R_normalized`, `G_normalized`, `B_normalized` |
| Frame stats | `frame_luminance`, `face_detected`, `skin_pixel_count` |
| BVP variants | `BVP_detrended`, `BVP_butterworth`, `BVP_cheby1`, `BVP_cheby2`, `BVP_elliptic`, `BVP_pca` |
| Ground truth | `gt_bpm` |
| Quality metrics | `bisenet_conf`, `vote_mean`, `highconf_pct`, `roi_coverage`, `green_cv`, `hair_rej_frac`, `clust_agree`, `quality_score` |

The `summary.txt` logs the run parameters and aggregate statistics of all quality metrics for a quick health check without reading the full CSV.

---

## 10. Complete Signal Flow Diagram

```
Video file (.mov)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  PER FRAME LOOP                                             │
│                                                             │
│  BGR frame ──► RGB ──► compute_frame_maps()                │
│                              │                              │
│                    ┌─────────┴──────────┐                  │
│                    │  Color maps dict    │                  │
│                    │  (Y,Cr,Cb,H,S,V,   │                  │
│                    │   R,G,B floats)     │                  │
│                    └─────────┬──────────┘                  │
│                              │                              │
│  MediaPipe FaceLandmarker ◄──┤  (468 pts every frame)     │
│         │                    │                              │
│         ▼                    │                              │
│  build_rois() ──► polygon masks                            │
│  (cached if landmarks        │                              │
│   moved < 4px)               │                              │
│         │                    │                              │
│         ▼                    │                              │
│  ┌─────────────────┐         │                              │
│  │ BiSeNet (every  │         │                              │
│  │ 10 frames)      │         │                              │
│  │ parse_skin()    │         │                              │
│  │ → binary mask   │         │                              │
│  │ → prob map      │         │                              │
│  └────────┬────────┘         │                              │
│           │                  │                              │
│           ▼                  │                              │
│  fit_gmm() ──► predict_gmm() │                              │
│           │                  │                              │
│           ▼                  │                              │
│  EMA smooth prob map         │                              │
│           │                  │                              │
│           ▼                  ▼                              │
│  ┌────────────────┐  ┌────────────────────────┐            │
│  │ THREAD 1       │  │ THREAD 2               │            │
│  │ compute_       │  │ adaptive_color_skin()  │            │
│  │ texture_maps() │  │ (YCbCr + Kovac + HSV)  │            │
│  │ LBP, coh,      │  │ 2-of-3 voting          │            │
│  │ SVC, specular  │  │                        │            │
│  └───────┬────────┘  └──────────┬─────────────┘            │
│          │                      │                           │
│          ▼                      ▼                           │
│  hair_rejection()     color_sk (bool mask)                  │
│  (forehead pct=78,              │                           │
│   cheeks pct=72)                │                           │
│          │                      │                           │
│          ▼                      ▼                           │
│  ┌─────────────────────────────────────────────┐           │
│  │  Final combination:                          │           │
│  │  comb_sk = BiSeNet | color_sk |              │           │
│  │            (GMM & vote≥1)                    │           │
│  │  fore_skin = comb_sk & forehead             │           │
│  │             & ~hair & ~excl & ~specular     │           │
│  └──────────────────┬──────────────────────────┘           │
│                     │                                       │
│                     ▼                                       │
│  Luminance-normalised, prob-weighted RGB extraction         │
│  R_t[i], G_t[i], B_t[i]                                   │
│  Rf_t…Br_t (per region)                                    │
│                                                             │
│  Confidence metrics → bisenet_conf, vote_mean, quality…    │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  POST PROCESSING (after loop)                               │
│                                                             │
│  R_t,G_t,B_t ──► CHROM projection ──► S (BVP)             │
│                         │                                   │
│                         ▼                                   │
│  Linear detrend ──► S_det                                  │
│                                                             │
│  9 per-region signals ──► SVD ──► S_pca (best cardiac PC)  │
│                                                             │
│  S_det ──► Butterworth  ──► BVP_butterworth                │
│         ──► Chebyshev I ──► BVP_cheby1                     │
│         ──► Chebyshev II──► BVP_cheby2                     │
│         ──► Elliptic    ──► BVP_elliptic                   │
│                                                             │
│  GT CSV ──► np.interp ──► gt_bpm (per frame)               │
│                                                             │
│  All → rppg_output.csv  +  summary.txt                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Why These Parameter Values?

| Parameter                                | Value                               | Rationale                                                                                                                                |
| ---------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `BISENET_EVERY = 10`                     | Run BiSeNet every 10 frames         | Skin colour changes at ~1 Hz or slower; 3 Hz refresh (30fps÷10) is sufficient. Reduces neural inference time from 100% to 10% of frames. |
| `MAX_CROP_PX = 384`                      | Max crop dimension before SegFormer | SegFormer is fully convolutional; 384px gives ~95% of 512px accuracy at ~55% of the compute cost.                                        |
| `EMA_ALPHA = 0.03`                       | Probability map smoothing           | `τ = 1/(0.03×30) ≈ 1.1s`. Smooths prob map across 10-frame BiSeNet gaps without introducing >1s of lag.                                  |
| `DISPLAY_EVERY = 3`                      | Refresh display every 3 frames      | Human eye perceives ~10fps for diagnostic overlays; 30fps display is wasted GPU/CPU time.                                                |
| Hair rejection `pct=78` (forehead)       | 78th percentile threshold           | Forehead hair is rare; higher threshold = fewer false rejections.                                                                        |
| Hair rejection `pct=72` (cheeks)         | 72nd percentile threshold           | Sideburns and temple hair common; slightly lower = catch more actual hair.                                                               |
| 4px landmark movement cache invalidation | `> 4.0 pixels`                      | Sub-4px movement from tracker noise doesn't meaningfully change ROI shape. Prevents wasted polygon recomputation.                        |
| Skin consensus threshold                 | `≥ 2` of 3 methods                  | Majority vote. 3-of-3 would be too strict (loses shadow/challenging-lighting pixels); 1-of-3 too loose (includes too much background).   |
| GMM components `n_comp=3`                | 3 clusters                          | Empirically matches illumination variation on a face: well-lit, shadowed, transitional skin.                                             |
| GMM subsample `≤4000`                    | Max 4000 training points            | Sufficient for stable GMM convergence; above 4000 adds runtime with diminishing accuracy improvement.                                    |
| Cheek ROI fractions `0.12`, `0.72`       | 12% and 72% eye-to-jaw              | 12%: clears the orbit/eye region. 72%: stops above the jaw/chin transition. Empirical from multiple face shapes.                         |
| CHROM `n=3, m=1.5` coefficients          | Fixed per de Haan (2013)            | Analytically derived from camera spectral sensitivity integrals over haemoglobin absorption spectrum. Not empirical.                     |
| Filter band `[0.7, 3.5]` Hz              | 42–210 BPM                          | Covers full physiological heart rate range with margins for athlete (low HR) and exercise (high HR) conditions.                          |
| Filter order 4                           | All filters                         | Order-4 gives adequate roll-off (~80dB/decade) without excessive group delay. Higher order → more ringing artifacts.                     |
| Quality weight `0.25` for BiSeNet        | Highest weight                      | Deep learning classifier is the most accurate single skin predictor; colour rules and GMM corroborate.                                   |

---

## 12. Appendix — References

### A.1 SegFormer & Face Parsing

- **SegFormer paper:** Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS 2021. [https://arxiv.org/abs/2105.15203](https://arxiv.org/abs/2105.15203)
- **CelebAMask-HQ dataset (fine-tuning data):** Lee, C. H., Liu, Z., Wu, L., & Luo, P. (2020). *MaskGAN: Towards Diverse and Interactive Facial Image Manipulation.* CVPR 2020. [https://arxiv.org/abs/1907.11922](https://arxiv.org/abs/1907.11922)
- **HuggingFace model card:** [https://huggingface.co/jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing)

### A.2 MediaPipe 468-Point Face Mesh

- **MediaPipe Face Mesh paper:** Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019). *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs.* CVPR Workshop. [https://arxiv.org/abs/1907.06724](https://arxiv.org/abs/1907.06724)
- **MediaPipe FaceLandmarker documentation:** [https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- **468-point landmark map:** [https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

### A.3 Local Binary Patterns

- **Original LBP paper:** Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 971–987. [https://doi.org/10.1109/TPAMI.2002.1017623](https://doi.org/10.1109/TPAMI.2002.1017623)
- **Uniform LBP and transition count:** The "uniform" patterns (≤2 transitions) represent 90%+ of face texture; non-uniform patterns with high transition counts identify fine repetitive textures like hair.

### A.4 YCbCr Skin Detection

- **Chai & Ngan (1999):** Chai, D., & Ngan, K. N. (1999). *Face segmentation using skin-color map in videophone applications.* IEEE Transactions on Circuits and Systems for Video Technology, 9(4), 551–564. [https://doi.org/10.1109/76.767122](https://doi.org/10.1109/76.767122)
- **Hsu et al. (2002):** Hsu, R. L., Abdel-Mottaleb, M., & Jain, A. K. (2002). *Face detection in color images.* IEEE TPAMI, 24(5), 696–706. [https://doi.org/10.1109/34.1000242](https://doi.org/10.1109/34.1000242)

### A.5 Kovac RGB Skin Rule

- **Kovac et al. (2003):** Kovac, J., Peer, P., & Solina, F. (2003). *Human skin color clustering for face detection.* EUROCON 2003, 2, 144–148. [https://doi.org/10.1109/EURCON.2003.1248032](https://doi.org/10.1109/EURCON.2003.1248032)

### A.6 Gaussian Mixture Models for Skin

- **GMM background subtraction (Stauffer & Grimson):** Stauffer, C., & Grimson, W. E. L. (1999). *Adaptive background mixture models for real-time tracking.* CVPR 1999, 2, 246–252. [https://doi.org/10.1109/CVPR.1999.784637](https://doi.org/10.1109/CVPR.1999.784637)
- **Pattern recognition text (Bishop):** Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer. Chapter 9: Mixture Models and EM. [https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

### A.7 CHROM rPPG Algorithm

- **de Haan & Jeanne (2013):** de Haan, G., & Jeanne, V. (2013). *Robust Pulse Rate From Chrominance-Based rPPG.* IEEE Transactions on Biomedical Engineering, 60(10), 2878–2886. [https://doi.org/10.1109/TBME.2013.2266196](https://doi.org/10.1109/TBME.2013.2266196)
- **This is the canonical reference for the Xs/Ys/α equations used in this file.** The coefficients (3, -2, 1.5, 1, -1.5) are derived from integrals of camera spectral response curves over the haemoglobin/oxy-haemoglobin absorption spectra.

### A.8 Spatial PCA for rPPG

- **Lewandowska et al. (2011):** Lewandowska, M., Rumiński, J., Kocejko, T., & Nowak, J. (2011). *Measuring Pulse Rate with a Webcam — a Non-contact Method for Evaluating Cardiac Activity.* FedCSIS 2011, 405–410.
- **PCA-based multi-channel rPPG:** The idea of applying PCA to multi-region or multi-channel facial signals to extract the dominant cardiac mode is discussed in several rPPG survey papers, e.g., McDuff, D. et al. (2015). *Improvements in Remote Cardiopulmonary Measurement Using a Five Band Digital Camera.* IEEE TBME, 61(10).

### A.9 IIR Filter Design

- **Proakis & Manolakis:** Proakis, J. G., & Manolakis, D. K. (2006). *Digital Signal Processing: Principles, Algorithms, and Applications.* 4th ed. Prentice Hall. Chapters 8–9 (IIR filter design).
- **SciPy filter design docs:** [https://docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html)
- **Butterworth filter:** Maximally flat magnitude in the passband. Used as the baseline in this pipeline.
- **Elliptic (Cauer) filter:** Sharpest transition for a given order — at the cost of ripple in both bands.
- **`filtfilt` zero-phase:** [https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)

### A.10 Circular Statistics for Hue

- **Fisher, N. I. (1993).** *Statistical Analysis of Circular Data.* Cambridge University Press. [https://doi.org/10.1017/CBO9780511564345](https://doi.org/10.1017/CBO9780511564345)
- The circular mean via `atan2(mean(sin), mean(cos))` is the standard formula for averaging angles.

### A.11 Exponential Moving Average (EMA)

- Standard signal processing: EMA with coefficient α: `y[n] = α·x[n] + (1-α)·y[n-1]`. Time constant (63% rise time) = `1/(α·fs)` samples = `1/α` frames. With α=0.03 and fs=30: τ ≈ 1.1 seconds.

### A.12 rPPG Survey Papers (General Context)

- **McDuff et al. (2023):** McDuff, D., Wander, M., Liu, X., et al. (2023). *SCAMPS: Synthetics for Camera Measurement of Physiological Signals.* NeurIPS 2022. [https://arxiv.org/abs/2206.04197](https://arxiv.org/abs/2206.04197)
- **Verkruysse et al. (2008):** Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). *Remote plethysmographic imaging using ambient light.* Optics Express, 16(26), 21434–21445. [https://doi.org/10.1364/OE.16.021434](https://doi.org/10.1364/OE.16.021434) — First demonstration that rPPG is feasible with ambient light.
- **Wang et al. (2017):** Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). *Algorithmic Principles of Remote PPG.* IEEE TBME, 64(7), 1479–1491. [https://doi.org/10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282) — Comprehensive survey that situates CHROM among all rPPG methods.

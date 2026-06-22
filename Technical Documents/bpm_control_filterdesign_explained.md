# bpm_control_filterdesign.m — Complete Line-by-Line Explanation

**Author:** Devansh Bajwala & Prof. Xu Chen  
**Purpose:** Compare IIR and FIR bandpass filter designs for rPPG heart-rate extraction, and quantify how filter choice affects BPM accuracy against a ground-truth (GT) pulse-oximeter signal.

---

## Table of Contents

1. [Workspace Initialization (Line 1)](#1-workspace-initialization)
2. [File Paths (Lines 3–4)](#2-file-paths)
3. [Ground-Truth Loading & Statistics (Lines 6–14)](#3-ground-truth-loading--statistics)
4. [First-Frame Read for Face Detection (Lines 16–18)](#4-first-frame-read-for-face-detection)
5. [Face Detection (Lines 20–33)](#5-face-detection)
6. [RGB Extraction Loop (Lines 35–53)](#6-rgb-extraction-loop)
7. [CHROM Projection & Detrending (Lines 55–67)](#7-chrom-projection--detrending)
8. [Filter Specifications (Lines 69–78)](#8-filter-specifications)
9. [IIR Filter Design (Lines 80–94)](#9-iir-filter-design)
10. [FIR Order Estimation (Lines 96–106)](#10-fir-order-estimation)
11. [FIR Hamming Window Filters (Lines 108–111)](#11-fir-hamming-window-filters)
12. [FIR Kaiser Window Filter (Lines 113–120)](#12-fir-kaiser-window-filter)
13. [FIR Parks-McClellan Filter (Lines 122–129)](#13-fir-parks-mcclellan-filter)
14. [Frequency Response Evaluation (Lines 131–144)](#14-frequency-response-evaluation)
15. [Figure 1 — IIR Magnitude (Lines 146–160)](#15-figure-1--iir-magnitude)
16. [Figure 2 — FIR Magnitude (Lines 162–179)](#16-figure-2--fir-magnitude)
17. [Figure 3 — Phase Response (Lines 181–201)](#17-figure-3--phase-response)
18. [Figure 4 — Group Delay (Lines 203–234)](#18-figure-4--group-delay)
19. [Applying Filters to Signal (Lines 236–245)](#19-applying-filters-to-signal)
20. [Figure 5 — Time-Domain BVP (Lines 247–268)](#20-figure-5--time-domain-bvp)
21. [Figure 6 — PSD vs Ground Truth (Lines 270–293)](#21-figure-6--psd-vs-ground-truth)
22. [Sliding-Window BPM Evaluation (Lines 295–323)](#22-sliding-window-bpm-evaluation)
23. [Error Computation (Lines 325–327)](#23-error-computation)
24. [Figure 7 — Sliding BPM vs GT, IIR (Lines 329–341)](#24-figure-7--sliding-bpm-vs-gt-iir)
25. [Figure 8 — Sliding BPM vs GT, FIR (Lines 343–356)](#25-figure-8--sliding-bpm-vs-gt-fir)
26. [Figure 9 — Per-Window Error (Lines 358–379)](#26-figure-9--per-window-error)
27. [Summary Table (Lines 381–407)](#27-summary-table)
28. [Helper Function: sosfreqz (Lines 409–416)](#28-helper-function-sosfreqz)

---

## 1. Workspace Initialization

```matlab
clear; close all; clc;
```

Three MATLAB commands chained on one line with semicolons acting as statement separators (not suppressors here — they have no output to suppress anyway):

- **`clear`** — Removes every variable from the current MATLAB workspace. This guarantees the script starts from a blank slate. Without this, a leftover variable from a previous run (e.g., a stale `R_t` array) could silently corrupt the current run.
- **`close all`** — Closes every open MATLAB figure window. Prevents confusion from old plots persisting alongside new ones. This script creates nine figures; without this line, re-running would open nine *more* on top of the old ones.
- **`clc`** — Clears the Command Window text. Makes `fprintf` output from the current run readable without scrolling past output from prior runs.

---

## 2. File Paths

```matlab
VID_PATH = '/home/macs/Documents/rPPG-Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/recording_2026-04-07T22-04-35Z.mov';
CSV_PATH = '/home/macs/Documents/rPPG-Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/vitals.csv';
```

- **`VID_PATH`** — Absolute path to the `.mov` video file recorded by the DocBOT robot. The filename encodes a UTC timestamp (`T22-04-35Z`). This is the raw facial video from which the rPPG signal will be extracted.
- **`CSV_PATH`** — Absolute path to `vitals.csv`, which contains ground-truth heart rate readings recorded simultaneously by a pulse-oximeter or similar physiological sensor. Having both on the same absolute path ensures the script is deterministic regardless of MATLAB's current working directory.

> **Why absolute paths?** MATLAB's `cd` can change between runs. Absolute paths remove that ambiguity. The trade-off is the script must be updated if data moves.

---

## 3. Ground-Truth Loading & Statistics

```matlab
csv_data      = readtable(CSV_PATH);
```
`readtable` reads the CSV into a MATLAB `table` object. Each column in the CSV becomes a named field. Variable names are taken from the CSV header row automatically.

```matlab
gt_time       = csv_data.offset_seconds;
```
Extracts the `offset_seconds` column — time (in seconds) elapsed since recording began for each GT measurement. The dot-notation accesses a named column from the table.

```matlab
gt_hr         = csv_data.heart_rate;
```
Extracts the `heart_rate` column — the ground-truth BPM values recorded by the physiological sensor at each timestep.

```matlab
valid         = ~isnan(gt_hr);
```
`isnan` returns a logical array: `true` wherever `gt_hr` is NaN (Not a Number). The `~` negates it, so `valid` is `true` wherever the GT reading is a real number. GT devices sometimes produce NaN during dropouts (probe off finger, motion artifact, etc.).

```matlab
gt_time       = gt_time(valid);  gt_hr = gt_hr(valid);
```
Logical indexing: keep only the time and heart-rate entries where `valid == true`. Both arrays are filtered together to stay synchronized — row *i* of `gt_time` always matches row *i* of `gt_hr`.

```matlab
bpm_gt_mean   = mean(gt_hr);
bpm_gt_median = median(gt_hr);
```
Scalar summary statistics over the entire recording session.
- **Mean** — sensitive to outliers (a momentary spike in GT BPM pulls it).
- **Median** — robust to outliers; better represents the "typical" heart rate.

Both are stored because the script later uses them as reference lines on plots.

```matlab
fprintf('GT: mean=%.1f  median=%.1f  range=[%d %d]  n=%d\n', ...
    bpm_gt_mean, bpm_gt_median, min(gt_hr), max(gt_hr), numel(gt_hr));
```
Prints a formatted summary to the Command Window.
- `%.1f` — floating-point with 1 decimal place.
- `%d` — integer format for `min` and `max` BPM.
- `%d` again for `numel(gt_hr)`, the total count of valid GT samples.
- `...` is MATLAB's line continuation operator.

This printout is the first sanity check: confirm the GT data loaded correctly before spending time processing the video.

---

## 4. First-Frame Read for Face Detection

```matlab
vid_tmp     = VideoReader(VID_PATH);
```
Creates a `VideoReader` object for the `.mov` file. This does *not* load the entire video into RAM — it opens a handle and lets you pull frames on demand.

```matlab
first_frame = readFrame(vid_tmp);
```
Reads exactly one frame (the very first frame) into `first_frame`, a `uint8` array of shape `[H × W × 3]` (Height × Width × RGB channels).

```matlab
[H_vid, W_vid, ~] = size(first_frame);  clear vid_tmp;
```
- `size(first_frame)` returns `[H, W, C]` where C=3 for RGB.
- `~` discards the third output (C=3, which we don't need).
- `H_vid` and `W_vid` store the pixel dimensions of every frame (they're constant throughout the video).
- `clear vid_tmp` immediately releases the `VideoReader` object. A `VideoReader` holds a file handle and sometimes a decode buffer. Clearing it frees both, because the full-video reader created later on line 35 will need its own handle.

---

## 5. Face Detection

```matlab
try
    detector  = vision.CascadeObjectDetector();
    all_boxes = step(detector, first_frame);
catch;  all_boxes = []; end
```

A `try/catch` block wraps the face detection so the script doesn't crash if the Computer Vision Toolbox is missing.

- **`vision.CascadeObjectDetector()`** — Creates a face detector based on the Viola-Jones algorithm (2001). Internally, it uses a cascade of Haar-like feature classifiers trained on thousands of face images. With no arguments, it defaults to detecting frontal faces.
- **`step(detector, first_frame)`** — Runs the detector on `first_frame`. Returns an `M×4` matrix where each row is `[x, y, width, height]` (in pixels) of a detected bounding box. Returns an empty matrix if no face is found.
- **`catch; all_boxes = [];`** — If the toolbox is absent or detection throws any error, set `all_boxes` to empty so the fallback below can take over.

```matlab
if isempty(all_boxes)
    x1=floor(W_vid*0.25); y1=floor(H_vid*0.05);
    x2=floor(W_vid*0.75); y2=floor(H_vid*0.75);
```
**Fallback bounding box.** If the detector failed or found nothing:
- `x1 = 25%` from the left edge
- `x2 = 75%` from the left edge (center half of width)
- `y1 = 5%` from the top (skip the very top — usually background)
- `y2 = 75%` from the top (face is in the upper 3/4 of frame for most camera setups)

`floor` rounds down to integer pixel coordinates.

```matlab
else
    [~,idx] = max(all_boxes(:,3).*all_boxes(:,4));
```
If one or more faces were detected, select the **largest** one by area. `all_boxes(:,3)` is the column of widths; `all_boxes(:,4)` is the column of heights. Their element-wise product gives the area of each detected box. `max` returns the value (discarded with `~`) and the index `idx` of the largest.

```matlab
    bbox = all_boxes(idx,:);
```
Extract the single row corresponding to the largest face bounding box.

```matlab
    x1=max(bbox(1),1);               y1=max(bbox(2),1);
    x2=min(bbox(1)+bbox(3)-1,W_vid); y2=min(bbox(2)+bbox(4)-1,H_vid);
end
```
Convert from `[x, y, width, height]` format to `[x1, y1, x2, y2]` corner-pair format:
- `x2 = x + width - 1` (the `-1` because the box starts at pixel `x`, so it ends at `x + width - 1`, not `x + width`)
- `max(...,1)` prevents coordinates going below pixel index 1 (MATLAB is 1-indexed).
- `min(...,W_vid)` and `min(...,H_vid)` clamp to image boundaries so array indexing never goes out of range.

---

## 6. RGB Extraction Loop

```matlab
vid = VideoReader(VID_PATH);
fs  = vid.FrameRate;
R_t=[]; G_t=[]; B_t=[];
```
- Opens a **fresh** `VideoReader` for the full video pass.
- `vid.FrameRate` — reads the frames-per-second from the video metadata (e.g., 30.0 or 29.97 fps). This becomes the **sampling frequency** `fs` of the rPPG signal.
- `R_t`, `G_t`, `B_t` are initialized as empty row vectors. They will grow one element per valid frame inside the loop.

```matlab
while hasFrame(vid)
    frame = readFrame(vid);
```
`hasFrame` returns `true` while unread frames remain. `readFrame` pulls the next frame sequentially. Each `frame` is a `uint8 [H×W×3]` array with values in `[0,255]`.

```matlab
    fc    = frame(y1:y2, x1:x2, :);
```
Crops the frame to the face bounding box using array slicing. `y1:y2` selects rows (height), `x1:x2` selects columns (width), `:` keeps all 3 color channels. The result `fc` is a `[H_face × W_face × 3]` sub-image.

```matlab
    fcd   = double(fc);
```
Converts from `uint8` (integer, 0–255) to `double` (64-bit float). This is necessary before arithmetic — integer arithmetic in MATLAB saturates at 255 and doesn't support fractional values. All subsequent math operates on `fcd`.

```matlab
    lum   = mean(fcd(:));  if lum > 0; fcd = fcd/lum*128; end
```
**Luminance normalization.** 
- `fcd(:)` reshapes the entire 3D array into one long column vector.
- `mean(fcd(:))` — scalar mean across all pixels and all channels: the average brightness of the face crop.
- Division by `lum` and multiplication by `128` re-scales the frame so the mean pixel intensity is always 128 (mid-grey). This compensates for slow illumination drift (e.g., a lamp dimming over minutes) that would otherwise show up as a low-frequency artifact in the BVP signal.
- The guard `lum > 0` prevents division by zero for a pathologically dark frame.

```matlab
    Yf  =  0.299*fcd(:,:,1)    + 0.587*fcd(:,:,2)    + 0.114*fcd(:,:,3);
    Cbf = -0.168736*fcd(:,:,1) - 0.331264*fcd(:,:,2) + 0.5*fcd(:,:,3)      + 128;
    Crf =  0.5*fcd(:,:,1)      - 0.418688*fcd(:,:,2) - 0.081312*fcd(:,:,3) + 128;
```
**RGB → YCbCr conversion** using the BT.601 standard coefficients.

The YCbCr color space separates luminance (Y) from chrominance (Cb = blue difference, Cr = red difference):

| Component | Meaning |
|-----------|---------|
| Y  | Luma — perceived brightness |
| Cb | Blue-difference chroma |
| Cr | Red-difference chroma |

The `+128` offsets shift Cb and Cr to the range `[16,240]` from a signed `[-112,112]` range. Skin has a characteristic cluster in Cb/Cr space that is robust across ethnicities and lighting changes.

Each of `Yf`, `Cbf`, `Crf` is a 2D array of the same spatial size as `fcd(:,:,1)`.

```matlab
    Mf  = (Cbf>=77)&(Cbf<=127)&(Crf>=133)&(Crf<=173)&(Yf>40);
```
**Skin mask.** A pixel is classified as skin if it simultaneously satisfies all five conditions:
- `Cb ∈ [77, 127]` — within the blue-difference skin range
- `Cr ∈ [133, 173]` — within the red-difference skin range
- `Y > 40` — bright enough to not be a shadow

These thresholds come from empirical studies (Chai & Ngan, 1999; Kovac et al., 2003) and work well under varied lighting and skin tones. `Mf` is a logical 2D array: `true` = skin pixel.

```matlab
    if sum(Mf(:)) < 50;  continue;  end
```
If fewer than 50 pixels are classified as skin in this frame, skip it entirely (`continue` jumps to the next loop iteration without appending anything to `R_t/G_t/B_t`). This guards against:
- Frames where the face is occluded
- Frames where YCbCr thresholds fail due to extreme lighting
- Near-black or near-white overexposed frames

The threshold of 50 pixels is a practical minimum — too few pixels give a noisy spatial mean.

```matlab
    pix = reshape(fcd,[],3);  msk = Mf(:);
```
- `reshape(fcd,[],3)` — reshapes `fcd` from `[H×W×3]` to `[H*W × 3]`. Each row is one pixel with columns [R, G, B].
- `Mf(:)` — reshapes the 2D logical mask into a column vector of length `H*W`.

```matlab
    R_t(end+1) = mean(pix(msk,1));
    G_t(end+1) = mean(pix(msk,2));
    B_t(end+1) = mean(pix(msk,3));
```
- `pix(msk,1)` — all Red values for skin pixels only (logical row selection on the pixel matrix).
- `mean(...)` — spatial mean across all skin pixels in this frame: one scalar per channel.
- `end+1` appends to the growing row vector.

The `%#ok<SAGROW>` comment suppresses MATLAB's "growing array in loop" performance warning. Ideally you'd pre-allocate, but the exact number of valid frames isn't known in advance (some may be skipped due to the `continue` above).

After the loop, `R_t`, `G_t`, `B_t` are row vectors of length T (number of valid frames), each holding the spatially-averaged skin-pixel intensity per frame per channel. Together they form the raw rPPG signal, modulated by the blood volume pulse beneath the skin.

---

## 7. CHROM Projection & Detrending

```matlab
T      = length(R_t);
t_axis = (0:T-1)/fs;
```
- `T` — total number of valid frames extracted.
- `t_axis` — time vector in seconds. Frame 0 is at t=0, frame T-1 is at t=(T-1)/fs. Used as the x-axis for time-domain plots.

```matlab
R_n = R_t/mean(R_t);  G_n = G_t/mean(G_t);  B_n = B_t/mean(B_t);
```
**Normalization.** Each channel is divided by its own temporal mean.

After normalization, `R_n`, `G_n`, `B_n` all hover around 1.0. This makes the channels dimensionless and compensates for the fact that the three channels have different absolute intensities due to camera sensor sensitivity and skin reflectance spectra. Without normalization, the CHROM formula below would be numerically imbalanced.

```matlab
Xs  = 3*R_n - 2*G_n;
Ys  = 1.5*R_n + G_n - 1.5*B_n;
```
**CHROM projection** (de Haan & Jeanne, 2013).

The idea: skin reflectance has two components:
1. A specular (white-light) component that changes with illumination
2. A diffuse blood-volume component (the rPPG signal)

By projecting normalized RGB into two orthogonal "chrominance" directions Xs and Ys, the specular component cancels. The coefficients (3, -2) and (1.5, 1, -1.5) were derived from the known spectral properties of hemoglobin and melanin under D65 illumination.

```matlab
alpha = std(Xs)/std(Ys);
```
**Alpha scaling factor.** The ratio of standard deviations of the two chrominance signals. This compensates for the fact that the specular contamination projects differently onto Xs and Ys depending on the current illumination color temperature.

```matlab
S     = Xs - alpha*Ys;
```
**Final CHROM signal.** Subtracting `alpha*Ys` from `Xs` removes the residual specular contamination. `S` is the blood volume pulse (BVP) signal — its dominant oscillation should be at the heart rate frequency.

```matlab
t_vec  = (1:T)';
coeffs = [t_vec, ones(T,1)] \ S(:);
```
**Linear detrending via least squares.**
- `t_vec` is a column vector `[1; 2; ...; T]` (frame indices).
- `[t_vec, ones(T,1)]` constructs a `[T×2]` design matrix: first column is time (slope), second column is all-ones (intercept).
- `S(:)` ensures S is a column vector.
- `\` is MATLAB's backslash operator — solves the least-squares problem `A*x = b` for `x`. Here it finds `coeffs = [slope; intercept]` that best fits a line through S.

```matlab
S_det  = S(:) - (coeffs(1)*t_vec + coeffs(2));
```
Subtracts the best-fit line from S. This removes any slow linear drift in the BVP signal (caused by gradual illumination changes, subject movement drift, or DC bias). The result `S_det` is zero-mean and drift-free, ready for bandpass filtering.

```matlab
fprintf('Signal: T=%d frames  fs=%.4f Hz\n', T, fs);
```
Prints the number of extracted frames and the sampling rate. A quick sanity check that the video was parsed correctly.

---

## 8. Filter Specifications

```matlab
% ── Filter specifications — adjust these to explore different designs ─────
f_p1 = 0.7;   % Hz  lower passband edge (42 BPM)
f_p2 = 3.5;   % Hz  upper passband edge (210 BPM)
f_s1 = 0.4;   % Hz  lower stopband edge
f_s2 = 4.5;   % Hz  upper stopband edge
Rp   = 1.0;   % dB  max passband ripple
Rs   = 40;    % dB  min stopband attenuation
```

These six numbers fully define the **filter specification mask** — the requirements every designed filter must meet:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `f_p1` | 0.7 Hz | Lower edge of the passband = 42 BPM minimum (resting heart rate lower bound) |
| `f_p2` | 3.5 Hz | Upper edge of the passband = 210 BPM (maximum physiological heart rate) |
| `f_s1` | 0.4 Hz | Lower edge of the stopband — below here, signal must be attenuated by at least Rs dB |
| `f_s2` | 4.5 Hz | Upper edge of the stopband — above here, signal must be attenuated by at least Rs dB |
| `Rp`   | 1.0 dB | Maximum allowed ripple within the passband (amplitude variation ≤ ±0.5 dB from flat) |
| `Rs`   | 40 dB  | Minimum attenuation in the stopband (signal power reduced by factor of 10,000) |

The **transition bands** are the gaps between passband and stopband:
- Lower transition: `[f_s1, f_p1]` = `[0.4, 0.7]` Hz (0.3 Hz wide)
- Upper transition: `[f_p2, f_s2]` = `[3.5, 4.5]` Hz (1.0 Hz wide)

Narrower transition bands require higher filter order.

```matlab
Wp = [f_p1 f_p2]/(fs/2);
Ws = [f_s1 f_s2]/(fs/2);
```
**Normalization to Nyquist frequency.**

MATLAB's filter design functions (`buttord`, `butter`, etc.) work in **normalized frequency** where 1.0 = Nyquist = fs/2. Dividing by `fs/2` maps:
- `f_p1 = 0.7 Hz` → `Wp(1) = 0.7/(fs/2)` (e.g., 0.0467 if fs=30)
- `f_p2 = 3.5 Hz` → `Wp(2) = 3.5/(fs/2)` (e.g., 0.2333 if fs=30)

`Wp` and `Ws` are now 2-element row vectors specifying a bandpass specification.

---

## 9. IIR Filter Design

IIR (Infinite Impulse Response) filters are recursive — their output depends on both past inputs and past outputs. They achieve steep frequency roll-off with low filter order, but have **nonlinear phase** (different frequency components are delayed by different amounts).

### Butterworth

```matlab
[N_bw, Wn_bw]   = buttord(Wp, Ws, Rp, Rs);
```
`buttord` (Butterworth order estimation) solves the classical Butterworth filter-order equation to find the **minimum integer order** `N_bw` such that the filter exactly meets the passband ripple `Rp` and stopband attenuation `Rs` simultaneously. `Wn_bw` is the corresponding 3-dB cutoff frequency (or frequencies, for bandpass — a 2-element vector).

Mathematically, the Butterworth filter has a maximally flat magnitude response in the passband: |H(jω)|² = 1 / (1 + (ω/ωc)^(2N)). Higher N → steeper roll-off.

```matlab
N_bw = min(N_bw, 6);
```
Caps Butterworth order at 6. The Butterworth filter often needs a very high order to meet tight specs with a narrow transition band (it has no ripple to "spend"), and high-order digital IIR filters become numerically unstable in transfer function (`[b,a]`) form. The cap forces a compromise and is why SOS form (below) is also used.

```matlab
[sos_bw, g_bw]  = butter(N_bw, Wn_bw, 'bandpass');
```
`butter` designs the Butterworth filter and returns it in **Second-Order Sections (SOS)** form:
- `sos_bw` — an `[M×6]` matrix where each row is one biquad (2nd-order section): `[b0 b1 b2 a0 a1 a2]` (numerator and denominator of H(z) for that section)
- `g_bw` — an overall gain scalar

A bandpass filter of order N is implemented as N/2 cascaded biquads (for even N). SOS avoids the catastrophic numerical cancellation that occurs when polynomial coefficients of a high-degree polynomial are multiplied out into a single `[b, a]` form.

### Chebyshev Type I

```matlab
[N_c1, Wn_c1]   = cheb1ord(Wp, Ws, Rp, Rs);
[sos_c1, g_c1]  = cheby1(N_c1, Rp, Wn_c1, 'bandpass');
```
Chebyshev Type I: **equiripple in the passband, monotonic in the stopband.**

By allowing controlled ripple (`Rp` dB) in the passband, Chebyshev I achieves a steeper transition than Butterworth at the same order. `cheb1ord` finds the minimum order, `cheby1` returns SOS form.

### Chebyshev Type II

```matlab
[N_c2, Wn_c2]   = cheb2ord(Wp, Ws, Rp, Rs);
[sos_c2, g_c2]  = cheby2(N_c2, Rs, Wn_c2, 'bandpass');
```
Chebyshev Type II: **monotonic in the passband, equiripple in the stopband.**

This is often preferred for biomedical signals because the passband is smooth (no ripple in the cardiac band), while the stopband has controlled ripple at exactly the Rs level. Note that `cheby2` takes `Rs` (stopband ripple) as its second argument, not `Rp`.

### Elliptic

```matlab
[N_el, Wn_el]   = ellipord(Wp, Ws, Rp, Rs);
[sos_el, g_el]  = ellip(N_el, Rp, Rs, Wn_el, 'bandpass');
```
Elliptic (Cauer) filter: **equiripple in both passband and stopband.**

The elliptic filter achieves the **minimum possible order** for given (Rp, Rs, transition-band-width) specs — it uses all available "design freedom" for ripple in both bands simultaneously. It is the most computationally efficient IIR filter for meeting a given spec, but has the worst group delay distortion and the most complex pole-zero pattern.

```matlab
fprintf('\nIIR orders — Butterworth:%d  ChebyI:%d  ChebyII:%d  Elliptic:%d\n', N_bw,N_c1,N_c2,N_el);
```
Prints the filter orders so you can compare: typically Elliptic < Chebyshev < Butterworth for the same spec. This number directly determines computational cost.

---

## 10. FIR Order Estimation

FIR (Finite Impulse Response) filters are non-recursive — output depends only on past inputs. They have **linear phase** (constant group delay for all frequencies), which means all frequency components are delayed by the same amount — no phase distortion. The trade-off: much higher order than IIR for the same spec.

```matlab
delta_f     = min(f_p1-f_s1, f_s2-f_p2);
```
Finds the **narrower** of the two transition bands:
- Lower transition width: `f_p1 - f_s1 = 0.7 - 0.4 = 0.3 Hz`
- Upper transition width: `f_s2 - f_p2 = 4.5 - 3.5 = 1.0 Hz`

`min(0.3, 1.0) = 0.3 Hz` — the lower transition band is the bottleneck.

```matlab
delta_omega = 2*pi*delta_f/fs;
```
Converts the transition bandwidth from Hz to **radians per sample** (normalized angular frequency). The Kaiser formula operates in radians/sample.

`δω = 2π × Δf / fs`

```matlab
N_est = ceil((Rs-7.95)/(2.285*delta_omega));
```
**Kaiser's FIR order formula:**

`N ≈ (Rs - 7.95) / (2.285 × δω)`

where:
- Rs = stopband attenuation in dB (40 dB here)
- δω = transition bandwidth in radians/sample
- `ceil` rounds up to the nearest integer (we need at least this many taps)

This is an empirical formula that approximates the minimum number of filter taps needed. Substituting: `(40 - 7.95) / (2.285 × 2π × 0.3/30) ≈ 32.05 / 0.0191 ≈ 1678` taps. This is very high because the transition band (0.3 Hz) is narrow relative to the sampling rate.

```matlab
if mod(N_est,2)==0; N_est=N_est+1; end
```
`mod(N_est, 2)` computes the remainder when dividing by 2. If `N_est` is even, add 1 to make it **odd**. An odd-length FIR filter of order `N_est-1` has even order, which is a **Type I linear-phase FIR** filter — it has exact symmetry and can represent a bandpass correctly. (Even-order, i.e., odd-length, Type II filters have a zero at Nyquist which can distort the upper stopband.)

```matlab
N_lo  = 51;    % low order — wide transition, useful for comparison
N_mid = 151;
N_hi  = N_est; % meets Rs spec
```
Three Hamming-window FIR filters will be designed at these orders to show how order affects performance. `N_lo=51` and `N_mid=151` are intentionally below the Kaiser estimate — they deliberately fail the spec to illustrate the trade-off.

```matlab
fprintf('FIR Kaiser estimate: N=%d  (latency = %.1f s)\n', N_est, (N_est/2)/fs);
```
Prints the estimated order and the **group delay** (latency) introduced by the FIR filter: `(N/2) / fs` seconds. A linear-phase FIR of order N delays the signal by exactly N/2 samples = N/(2fs) seconds. For N≈1679 at fs=30, that's ~28 seconds of latency — a real-time limitation.

---

## 11. FIR Hamming Window Filters

```matlab
b_ham_lo  = fir1(N_lo -1, Wp, 'bandpass', hamming(N_lo));
b_ham_mid = fir1(N_mid-1, Wp, 'bandpass', hamming(N_mid));
b_ham_hi  = fir1(N_hi -1, Wp, 'bandpass', hamming(N_hi));
```

`fir1` designs a **windowed-sinc** FIR filter:
- **First argument:** filter order = `N - 1` (number of taps minus 1). A filter of order `N-1` has `N` coefficients.
- **Second argument:** `Wp` — normalized passband cutoff frequencies `[f_p1/(fs/2), f_p2/(fs/2)]`
- **`'bandpass'`** — specifies the filter type (also accepts `'low'`, `'high'`, `'stop'`)
- **Fourth argument:** window function of length `N` (same as number of taps)

**The Hamming window:**  
`w(n) = 0.54 - 0.46·cos(2πn/(N-1))`

It tapers the ideal sinc impulse response smoothly to zero at both ends. This controls the Gibbs phenomenon (ringing at transitions) by reducing the side-lobe level. The Hamming window achieves ~41 dB of stopband attenuation at the cost of a wider main lobe (wider transition band) compared to rectangular windowing.

`b_ham_lo`, `b_ham_mid`, `b_ham_hi` are row vectors of FIR coefficients (the impulse response h[n]).

---

## 12. FIR Kaiser Window Filter

```matlab
dev_stop = 10^(-Rs/20);
```
Converts the stopband attenuation from dB to **linear amplitude deviation**:  
`δ_stop = 10^(-40/20) = 10^(-2) = 0.01`

This means the stopband gain must be ≤ 0.01 (i.e., 99% of stopband signal is rejected).

```matlab
dev_pass = (10^(Rp/10)-1)/(10^(Rp/10)+1);
```
Converts the passband ripple from dB to **linear deviation**:  
`δ_pass = (10^(1/10) - 1)/(10^(1/10) + 1) ≈ 0.0559`

This formula comes from the relationship between peak ripple amplitude and dB: `Rp = -20·log10(1 - δ_pass)` approximately. It gives the passband deviation from unity gain.

```matlab
[N_ksr, Wn_ksr, beta_ksr, ftype_ksr] = kaiserord(...
    [f_s1 f_p1 f_p2 f_s2], [0 1 0], [dev_stop dev_pass dev_stop], fs);
```
`kaiserord` estimates the **Kaiser window parameters** from the frequency specification:
- **First arg:** band edges in Hz (4 values defining 3 bands: stopband / passband / stopband)
- **Second arg:** desired gain in each band (0 = stopband, 1 = passband)
- **Third arg:** tolerable deviation in each band (`[dev_stop, dev_pass, dev_stop]`)
- **Fourth arg:** `fs` — the sampling rate (so edges are in Hz not normalized)

Returns:
- `N_ksr` — estimated filter order
- `Wn_ksr` — cutoff frequencies (normalized)
- `beta_ksr` — Kaiser window shape parameter β
- `ftype_ksr` — filter type string (e.g., `'bandpass'`)

**Kaiser window β** controls the trade-off between main-lobe width and side-lobe level. Larger β → better stopband attenuation but wider transition band. `kaiserord` solves for the β that achieves the desired Rs.

```matlab
if mod(N_ksr,2)==0; N_ksr=N_ksr+1; end
```
Same odd-length enforcement as before — ensures Type I linear phase.

```matlab
b_ksr = fir1(N_ksr, Wn_ksr, ftype_ksr, kaiser(N_ksr+1, beta_ksr));
```
Designs the Kaiser-windowed FIR filter. Note the window length is `N_ksr+1` (number of taps = order+1).

`kaiser(N, beta)` generates the Kaiser window of length N with shape parameter beta.

```matlab
fprintf('Kaiser FIR: N=%d  beta=%.3f\n', N_ksr, beta_ksr);
```
Prints the Kaiser-estimated order and β. Useful to compare against the manual Kaiser formula estimate `N_est`.

---

## 13. FIR Parks-McClellan Filter

```matlab
N_pm       = N_ksr;
```
Uses the same order as the Kaiser filter for a fair comparison.

```matlab
w_ratio    = 10^((Rs-Rp)/20);
```
Computes the **weight ratio** between stopband and passband:  
`w_ratio = 10^((40-1)/20) = 10^(1.95) ≈ 89`

In Parks-McClellan design, you assign a weight to each band — higher weight means the optimizer penalizes errors in that band more heavily. By weighting the stopband ~89× more than the passband, the algorithm allocates most of its approximation power to achieving the required stopband attenuation `Rs`.

```matlab
bands_pm   = [0 f_s1 f_p1 f_p2 f_s2 fs/2]/(fs/2);
amps_pm    = [0 0    1    1    0    0   ];
weights_pm = [w_ratio 1 w_ratio];
```
Defines the **piecewise constant desired frequency response** for `firpm`:
- Bands: `[0→f_s1]`, `[f_p1→f_p2]`, `[f_s2→fs/2]` — lower stopband, passband, upper stopband
- `bands_pm` normalizes all frequencies by `fs/2` (Nyquist)
- `amps_pm` sets desired amplitude: 0 in stopbands, 1 in passband
- `weights_pm`: stopbands weighted `w_ratio` times more than the passband

```matlab
b_pm = firpm(N_pm-1, bands_pm, amps_pm, weights_pm);
```
**Parks-McClellan (Remez Exchange) algorithm** — designs the FIR filter of order `N_pm-1` that **minimizes the weighted Chebyshev (minimax) error** between the actual and desired frequency responses. This is the theoretically optimal equiripple FIR filter for the given order and specification.

Unlike the windowed-sinc approach (which applies a fixed window shape), Parks-McClellan iteratively adjusts the filter coefficients until the ripple is equalized across all bands. The result has the best possible stopband attenuation for the given order.

```matlab
fprintf('Parks-McClellan FIR: N=%d  stopband weight=%.1fx\n', N_pm, w_ratio);
```
Prints the filter order and the stop/passband weight ratio.

---

## 14. Frequency Response Evaluation

```matlab
f_zoom   = linspace(0, min(8, fs/2-0.01), 8192)';
```
Creates a column vector of 8192 evenly-spaced frequency points from 0 Hz to `min(8, fs/2-0.01)` Hz.
- The upper limit is `min(8, fs/2-0.01)`: caps at 8 Hz (well above the 3.5 Hz cardiac band) or just below Nyquist, whichever is smaller.
- `-0.01` avoids evaluating exactly at Nyquist where some filters have undefined behavior.
- 8192 points gives a very smooth frequency response curve for plotting.

```matlab
dB_floor = -80;
safeDB   = @(H) max(20*log10(abs(H)), dB_floor);
```
- `dB_floor = -80` — sets a floor at -80 dB for display purposes.
- `safeDB` is an **anonymous function** (lambda): takes a complex frequency response vector `H`, computes `20·log10(|H|)` (magnitude in dB), then clamps to -80 dB minimum.
- Without the floor, filter nulls (zeros) give `-Inf` dB which breaks the y-axis scale on plots.

```matlab
H_bw = sosfreqz(sos_bw, g_bw, f_zoom, fs);
H_c1 = sosfreqz(sos_c1, g_c1, f_zoom, fs);
H_c2 = sosfreqz(sos_c2, g_c2, f_zoom, fs);
H_el = sosfreqz(sos_el, g_el, f_zoom, fs);
```
Calls the custom helper function `sosfreqz` (defined at the bottom of the file) to evaluate the frequency response of each IIR filter stored in SOS form at the 8192 frequency points. Returns a complex column vector. (Explained in detail in Section 28.)

```matlab
H_ham_lo  = freqz(b_ham_lo,   1,   f_zoom, fs);
H_ham_mid = freqz(b_ham_mid,  1,   f_zoom, fs);
H_ham_hi  = freqz(b_ham_hi,   1,   f_zoom, fs);
H_ksr     = freqz(b_ksr,      1,   f_zoom, fs);
H_pm      = freqz(b_pm,       1,   f_zoom, fs);
```
`freqz(b, a, f, fs)` evaluates the frequency response of a filter with numerator coefficients `b` and denominator `a`. For FIR filters, the denominator is just `[1]` (no feedback). Evaluates at frequencies in `f` given in Hz (because `fs` is provided).

---

## 15. Figure 1 — IIR Magnitude

```matlab
figure('Name','Fig 1 — IIR Magnitude','Position',[30 600 800 360]);
```
Opens a new figure window:
- `'Name'` sets the title bar text.
- `'Position',[30 600 800 360]` sets pixel position and size: `[left bottom width height]` relative to screen bottom-left.

```matlab
plot(f_zoom, safeDB(H_bw),'b', 'LineWidth',1.8); hold on;
plot(f_zoom, safeDB(H_c1),'r', 'LineWidth',1.8);
plot(f_zoom, safeDB(H_c2),'g', 'LineWidth',1.8);
plot(f_zoom, safeDB(H_el),'m', 'LineWidth',1.8);
```
Plots the magnitude response (in dB) of each IIR filter:
- Blue (`'b'`) — Butterworth
- Red (`'r'`) — Chebyshev I
- Green (`'g'`) — Chebyshev II
- Magenta (`'m'`) — Elliptic
- `hold on` — keeps all four curves on the same axes.

```matlab
xline(f_p1,'k--','LineWidth',1.2); xline(f_p2,'k--','LineWidth',1.2);
xline(f_s1,'k:','LineWidth',1.0);  xline(f_s2,'k:','LineWidth',1.0);
```
Vertical reference lines:
- `xline` draws vertical lines at the given x-value.
- Dashed black lines at passband edges (`f_p1=0.7`, `f_p2=3.5` Hz).
- Dotted black lines at stopband edges (`f_s1=0.4`, `f_s2=4.5` Hz).

These form the visual "mask" on the plot: any filter response violating the region outside these lines is failing the spec.

```matlab
yline(-Rp,'b:',sprintf('-%ddB pass',Rp),'LineWidth',0.9);
yline(-Rs,'r--',sprintf('-%ddB stop',Rs),'LineWidth',0.9);
```
Horizontal reference lines:
- `-Rp = -1 dB` — the passband ripple limit (passband must stay above this).
- `-Rs = -40 dB` — the stopband attenuation requirement (stopband must be below this).
- Labels are generated with `sprintf`.

```matlab
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend(sprintf('Butterworth N=%d',N_bw),sprintf('Chebyshev I N=%d',N_c1), ...
       sprintf('Chebyshev II N=%d',N_c2),sprintf('Elliptic N=%d',N_el),'Location','southwest');
title(sprintf('IIR Filters — Magnitude  |  Rp=%.1fdB  Rs=%.0fdB',Rp,Rs));
ylim([dB_floor 5]); xlim([0 min(8,fs/2)]); grid on;
```
- Legend entries include the actual order of each filter (from `N_bw`, `N_c1`, etc.).
- `ylim([dB_floor 5])` = `[-80 5]` dB — shows the full stopband floor with a small margin above 0 dB.
- `xlim([0 min(8,fs/2)])` — x-axis from DC to 8 Hz (or Nyquist if lower).
- `grid on` — adds a background grid.

---

## 16. Figure 2 — FIR Magnitude

```matlab
figure('Name','Fig 2 — FIR Magnitude','Position',[30 180 800 360]);
plot(f_zoom, safeDB(H_ham_lo), 'b--','LineWidth',1.3); hold on;
plot(f_zoom, safeDB(H_ham_mid),'b-.','LineWidth',1.6);
plot(f_zoom, safeDB(H_ham_hi), 'b',  'LineWidth',2.2);
plot(f_zoom, safeDB(H_ksr),    'r',  'LineWidth',1.8);
plot(f_zoom, safeDB(H_pm),     'm',  'LineWidth',1.8);
```
Five FIR filter magnitude responses on one plot:
- Three Hamming filters in different blue line styles (`--`, `-.`, solid) and increasing width — visually showing how higher order sharpens the transition.
- Kaiser in red, Parks-McClellan in magenta.

The same `xline`/`yline` mask markers are applied as in Fig 1. The rest of the formatting is identical in structure.

---

## 17. Figure 3 — Phase Response

```matlab
figure('Name','Fig 3 — Phase','Position',[870 600 800 420]);
subplot(2,1,1);
```
`subplot(2,1,1)` — divides the figure into a 2-row, 1-column grid and selects the first (top) subplot.

```matlab
plot(f_zoom, unwrap(angle(H_bw))*180/pi,'b','LineWidth',1.5); hold on;
plot(f_zoom, unwrap(angle(H_c1))*180/pi,'r','LineWidth',1.5);
plot(f_zoom, unwrap(angle(H_c2))*180/pi,'g','LineWidth',1.5);
plot(f_zoom, unwrap(angle(H_el))*180/pi,'m','LineWidth',1.5);
```
- `angle(H)` — extracts the phase angle of the complex frequency response H(f), in radians. Range: `(-π, π]`.
- `unwrap(...)` — removes 2π phase jumps: when the phase crosses ±π, `unwrap` adds or subtracts 2π to keep the curve continuous. Without unwrapping, the phase appears to "wrap" repeatedly and looks like sawtooth — the true phase trend is hidden.
- `*180/pi` — converts radians to degrees for readability.

IIR filters have **nonlinear phase** — the phase is a nonlinear function of frequency, meaning different frequencies are delayed by different amounts. On the plot, this appears as a curved line rather than a straight line. For ECG/BVP signals, nonlinear phase causes waveform distortion (the harmonic components of the heartbeat pulse arrive at different times, smearing the pulse shape).

```matlab
xline(f_p1,'k--'); xline(f_p2,'k--');
ylabel('Phase (deg)'); title('IIR — nonlinear phase');
legend('Butterworth','Cheby I','Cheby II','Elliptic','Location','best');
xlim([0 min(8,fs/2)]); grid on;
```

```matlab
subplot(2,1,2);
plot(f_zoom, unwrap(angle(H_pm)) *180/pi,'m','LineWidth',1.8); hold on;
plot(f_zoom, unwrap(angle(H_ksr))*180/pi,'r','LineWidth',1.5);
plot(f_zoom, unwrap(angle(H_ham_hi))*180/pi,'b','LineWidth',1.5);
```
Second subplot for FIR phase. FIR filters with symmetric coefficients have **exactly linear phase** — the unwrapped phase is a straight line with slope `-(N/2)` samples (the group delay). This is the key visual distinction from the IIR subplot above: straight lines vs curves.

---

## 18. Figure 4 — Group Delay

Group delay is defined as: `τ(ω) = -dφ/dω`  
It measures how many samples (or seconds) a given frequency component is delayed by the filter. For rPPG, group delay matters because:
1. Too much delay → slow real-time BPM updates
2. Frequency-varying delay (IIR) → waveform distortion

```matlab
figure('Name','Fig 4 — Group Delay','Position',[870 150 800 420]);
subplot(2,1,1);
n_gd = 1024;
[gd_bw, f_gd] = grpdelay(sos_bw, n_gd, fs);
[gd_c1,    ~] = grpdelay(sos_c1, n_gd, fs);
[gd_c2,    ~] = grpdelay(sos_c2, n_gd, fs);
[gd_el,    ~] = grpdelay(sos_el, n_gd, fs);
```
`grpdelay(sos, n, fs)` computes the group delay of an SOS filter at `n=1024` uniformly-spaced frequency points. Returns:
- `gd_bw` — group delay in **samples**
- `f_gd` — corresponding frequency vector in Hz

Note: `grpdelay` accepts SOS form directly (MATLAB R2019b+).

```matlab
plot(f_gd,gd_bw/fs*1000,'b','LineWidth',1.5); hold on;
...
```
`gd_bw/fs*1000` converts group delay from **samples** to **milliseconds**: (samples / (samples/second)) × 1000 ms/s.

IIR group delay varies with frequency — the plot will show peaks near the passband edges where the filter poles cause maximum delay.

```matlab
subplot(2,1,2);
gd_pm  = grpdelay(b_pm,    1,f_zoom,fs);
gd_ksr = grpdelay(b_ksr,   1,f_zoom,fs);
gd_ham = grpdelay(b_ham_hi,1,f_zoom,fs);
```
For FIR filters, `grpdelay` is called with `(b, 1, f, fs)` — numerator only, denominator=1.

```matlab
plot(f_zoom,gd_pm /fs*1000,'m','LineWidth',1.8); hold on;
plot(f_zoom,gd_ksr/fs*1000,'r','LineWidth',1.5);
plot(f_zoom,gd_ham/fs*1000,'b','LineWidth',1.5);
```

```matlab
yline(N_pm/(2*fs)*1000,'k--', ...
    sprintf('%.0f ms = N/2 = %d samples',N_pm/(2*fs)*1000,floor(N_pm/2)), ...
    'LineWidth',1.2,'LabelVerticalAlignment','bottom');
```
Draws a horizontal reference line at the **theoretical FIR group delay**: `N/2` samples = `N/(2fs)` seconds = `N/(2fs)*1000` ms. For a linear-phase FIR of order N, every frequency is delayed by exactly N/2 samples. The `sprintf` label shows both the milliseconds and sample count.

`'LabelVerticalAlignment','bottom'` places the `yline` label below the line.

---

## 19. Applying Filters to Signal

```matlab
S_bw = filtfilt(sos_bw, g_bw, S_det);
S_c1 = filtfilt(sos_c1, g_c1, S_det);
S_c2 = filtfilt(sos_c2, g_c2, S_det);
S_el = filtfilt(sos_el, g_el, S_det);
```
`filtfilt` applies **zero-phase filtering** by processing the signal in both the forward and reverse directions:
1. Filter S_det forward → intermediate signal
2. Reverse the intermediate signal
3. Filter backward → reverse again

The net effect: the phase shift is applied twice in opposite directions, canceling out. Zero-phase filtering is critical here because:
- The heart rate peak in the PSD must be at the correct frequency — phase distortion doesn't shift the peak, so it's less critical for BPM
- But zero-phase prevents waveform distortion visible in time-domain plots

For IIR filters, `filtfilt` also doubles the effective filter order and squares the magnitude response — so a 4th-order Butterworth becomes effectively 8th-order magnitude after filtfilt.

`filtfilt` accepts SOS + gain form directly: `filtfilt(sos, g, x)`.

```matlab
S_ham_lo  = filtfilt(b_ham_lo, 1,   S_det);
S_ham_mid = filtfilt(b_ham_mid,1,   S_det);
S_ham_hi  = filtfilt(b_ham_hi, 1,   S_det);
S_ksr     = filtfilt(b_ksr,    1,   S_det);
S_pm      = filtfilt(b_pm,     1,   S_det);
```
Same for FIR filters. The denominator is `1` (FIR has no feedback polynomial). `filtfilt` on an FIR filter also achieves zero-phase (since the filter is already linear-phase, doubling cancels even that symmetric delay), resulting in zero latency and zero phase distortion.

---

## 20. Figure 5 — Time-Domain BVP

```matlab
figure('Name','Fig 5 — Filtered BVP Signals','Position',[30 50 1200 500]);
subplot(1,2,1);
plot(t_axis,S_det,'Color',[0.8 0.8 0.8],'LineWidth',0.8); hold on;
```
- `subplot(1,2,1)` — left panel of a 1×2 layout.
- Plots the unfiltered detrended signal `S_det` in light grey (`[0.8 0.8 0.8]` RGB) as a background reference.

```matlab
plot(t_axis,S_bw,'b','LineWidth',1.2);
plot(t_axis,S_c1,'r','LineWidth',1.2);
plot(t_axis,S_c2,'g','LineWidth',1.2);
plot(t_axis,S_el,'m','LineWidth',1.2);
xlabel('Time (s)'); ylabel('BVP Amplitude');
legend('S_{det}','Butterworth','Cheby I','Cheby II','Elliptic','Location','best');
title('IIR Filtered BVP'); grid on;
```
Overlays all four IIR-filtered BVP signals. `S_{det}` in the legend uses LaTeX subscript notation for the underscore.

```matlab
subplot(1,2,2);
plot(t_axis,S_det,    'Color',[0.8 0.8 0.8],'LineWidth',0.8); hold on;
plot(t_axis,S_ham_lo, 'b--','LineWidth',1.2);
plot(t_axis,S_ham_hi, 'b',  'LineWidth',1.5);
plot(t_axis,S_ksr,    'r',  'LineWidth',1.5);
plot(t_axis,S_pm,     'm',  'LineWidth',1.5);
```
Right panel: FIR-filtered signals. The Hamming low-order (`N=51`) is shown dashed to distinguish it from the spec-meeting version.

---

## 21. Figure 6 — PSD vs Ground Truth

```matlab
nperseg  = min(round(fs*10), T);
noverlap = floor(nperseg/2);
```
Parameters for Welch PSD estimation:
- `nperseg = min(10 seconds × fs, T)` — each Welch segment is at most 10 seconds long (or the full signal if shorter). Longer segments give better frequency resolution but more variance.
- `noverlap = floor(nperseg/2)` — 50% overlap between adjacent segments. Standard practice: trades some computation for reduced variance in the PSD estimate.

```matlab
[p_bw, f_p] = pwelch(S_bw,     hann(nperseg), noverlap, [], fs);
[p_c2,   ~] = pwelch(S_c2,     hann(nperseg), noverlap, [], fs);
[p_el,   ~] = pwelch(S_el,     hann(nperseg), noverlap, [], fs);
[p_ham,  ~] = pwelch(S_ham_hi, hann(nperseg), noverlap, [], fs);
[p_ksr,  ~] = pwelch(S_ksr,    hann(nperseg), noverlap, [], fs);
[p_pm,   ~] = pwelch(S_pm,     hann(nperseg), noverlap, [], fs);
```
`pwelch(x, window, noverlap, nfft, fs)`:
- `x` — the filtered BVP signal
- `hann(nperseg)` — Hann (Hanning) window applied to each segment. Reduces spectral leakage (energy spreading from one frequency bin to neighbors).
- `noverlap` — samples of overlap between segments
- `[]` — NFFT: empty means use `nperseg` as the FFT length
- `fs` — sampling rate; makes `f_p` return in Hz

Returns `p` (one-sided PSD in units/Hz²) and `f_p` (frequency vector in Hz). `~` discards repeated frequency vectors.

```matlab
band = (f_p >= f_p1) & (f_p <= f_p2);
```
Logical index selecting only the cardiac frequency band `[0.7, 3.5]` Hz from the PSD arrays.

```matlab
figure('Name','Fig 6 — PSD vs Ground Truth','Position',[870 50 900 380]);
plot(f_p(band)*60, 10*log10(p_bw(band)), 'b', 'LineWidth',1.5); hold on;
```
- `f_p(band)*60` — converts frequency from Hz to BPM (multiply by 60 seconds/minute). The x-axis now reads in BPM directly.
- `10*log10(p_bw(band))` — converts power spectral density to dB. (Note: `10*log10` not `20*log10` because PSD is already a *power* quantity, not an amplitude.)

```matlab
xline(bpm_gt_mean,  'k-', sprintf('GT mean %.1f BPM',  bpm_gt_mean), 'LineWidth',2.5,'LabelVerticalAlignment','bottom');
xline(bpm_gt_median,'k--',sprintf('GT med %.1f BPM',bpm_gt_median),  'LineWidth',1.5,'LabelVerticalAlignment','top');
```
Two vertical ground-truth reference lines in BPM. The PSD peak should visually align with these lines if the filter is working correctly. Thick solid for mean, thinner dashed for median.

---

## 22. Sliding-Window BPM Evaluation

This section measures **how accurately** each filter produces a correct BPM estimate over time, using a sliding window approach that mimics real-time operation.

```matlab
sw_len    = round(10*fs);
```
**Window length** = 10 seconds × fs frames. Each BPM estimate is computed from a 10-second segment of filtered signal. 10 seconds gives ~0.1 Hz frequency resolution with Welch PSD, sufficient for ±3 BPM accuracy.

```matlab
sw_step   = round(1*fs);
```
**Step size** = 1 second. The window moves 1 second forward between estimates. Results in 1-Hz temporal resolution of the BPM track.

```matlab
sw_starts = 1:sw_step:T-sw_len+1;
n_sw      = numel(sw_starts);
```
- `sw_starts` — vector of starting frame indices for each window. Starts at frame 1, steps by `sw_step`, stops when the last window would exceed the signal length.
- `n_sw` — total number of windows.

```matlab
sw_time   = zeros(1,n_sw);
```
Pre-allocate a vector to store the center time of each window (for plotting).

```matlab
all_sigs   = {S_bw,S_c1,S_c2,S_el, S_ham_lo,S_ham_mid,S_ham_hi,S_ksr,S_pm};
all_labels = {'Butterworth','Cheby I','Cheby II','Elliptic', ...
              sprintf('Hamming N=%d',N_lo),sprintf('Hamming N=%d',N_mid), ...
              sprintf('Hamming N=%d (spec)',N_hi), ...
              sprintf('Kaiser N=%d',N_ksr),sprintf('Parks-McClellan N=%d',N_pm)};
nF      = numel(all_sigs);
```
Cell arrays grouping all 9 filtered signals and their labels. `nF=9`. This allows the evaluation loop to operate generically over all filters without code duplication.

```matlab
sw_bpm  = zeros(nF, n_sw);
```
Pre-allocate the `[9 × n_sw]` matrix of BPM estimates. Row `fi` = filter `fi`, column `k` = window `k`.

```matlab
nfft_sw = 4096;
```
FFT length for the Welch PSD inside each sliding window. 4096 > the 10-second window length (e.g., 300 samples at 30 fps). **Zero-padding** the FFT to 4096 gives finer frequency interpolation: frequency resolution = `fs/nfft_sw = 30/4096 ≈ 0.0073 Hz ≈ 0.44 BPM`. Better than the unpadded resolution of `fs/300 ≈ 0.1 Hz ≈ 6 BPM`.

```matlab
for k = 1:n_sw
    idx_s      = sw_starts(k);
    idx_e      = idx_s + sw_len - 1;
    sw_time(k) = (idx_s-1)/fs + sw_len/(2*fs);
```
- `idx_s`, `idx_e` — start and end frame indices of window k.
- `sw_time(k)` — the **center time** of window k in seconds:
  - `(idx_s-1)/fs` = time of start frame
  - `+ sw_len/(2*fs)` = half the window duration
  
  This places the BPM estimate at the middle of the window, which is more meaningful for comparing against time-varying GT data.

```matlab
    for fi = 1:nF
        seg = all_sigs{fi}(idx_s:idx_e);
```
Extracts the windowed segment from filtered signal `fi`. `all_sigs{fi}` uses curly-brace indexing to retrieve from a cell array.

```matlab
        np  = length(seg);
        [pw_k,fw_k] = pwelch(seg, hann(np), floor(np/2), nfft_sw, fs);
```
Welch PSD on this 10-second segment:
- `hann(np)` — single Hann window over the full segment (with 50% overlap and the 4096 FFT, internally Welch may subdivide).
- `nfft_sw=4096` — zero-padded FFT for fine frequency resolution.

```matlab
        bk = (fw_k>=f_p1)&(fw_k<=f_p2);
        [~,pi_] = max(pw_k(bk));  fb_ = fw_k(bk);
        sw_bpm(fi,k) = fb_(pi_)*60;
```
- `bk` — logical index for the cardiac band.
- `max(pw_k(bk))` — finds the peak power in the cardiac band. `pi_` is the index within the cardiac-band subvector.
- `fb_(pi_)` — the frequency at that peak in Hz.
- `*60` — converts Hz to BPM.
- Stored in `sw_bpm(fi,k)`.

---

## 23. Error Computation

```matlab
gt_interp = interp1(gt_time, double(gt_hr), sw_time, 'linear','extrap');
```
`interp1` performs **1D interpolation** of the GT heart rate data at the sliding-window timestamps:
- `gt_time` — the known GT timestamps (irregularly sampled by the physiological sensor)
- `double(gt_hr)` — GT BPM values (cast to double for arithmetic)
- `sw_time` — the center times of each sliding window (where we need GT values)
- `'linear'` — linear interpolation between adjacent GT samples
- `'extrap'` — extrapolation outside the GT time range (uses the slope of the last/first segment)

Result: `gt_interp` is a vector of GT BPM values, one per window, synchronized with `sw_bpm`.

```matlab
sw_err    = sw_bpm - gt_interp;
```
**Element-wise subtraction.** `gt_interp` is a row vector of length `n_sw`, and `sw_bpm` is `[nF × n_sw]`. MATLAB **broadcasts** `gt_interp` across all rows of `sw_bpm`, subtracting the same GT value from all filters' estimates at each timestep. Result: `sw_err` is `[9 × n_sw]` — the BPM error for each filter at each time window.

Positive error → filter overestimates heart rate. Negative → underestimates.

---

## 24. Figure 7 — Sliding BPM vs GT, IIR

```matlab
figure('Name','Fig 7 — Sliding BPM vs GT (IIR)','Position',[30 50 1000 400]);
stairs(gt_time,gt_hr,'k','LineWidth',2.5); hold on;
```
`stairs` plots data as a **staircase** (step function), appropriate for GT data because the physiological sensor reports a new BPM as a discrete step at each update, not a smooth transition.

```matlab
iir_c = {'b','r','g','m'};
for fi = 1:4
    est = sw_bpm(fi,:); sigma = movstd(est,11);
```
- Loops over the 4 IIR filters.
- `est` — the BPM time series for filter `fi`.
- `movstd(est, 11)` — **moving standard deviation** with a window of 11 samples. Measures local variability (jitter) in the BPM estimate. Used to draw a confidence band.

```matlab
    fill([sw_time,fliplr(sw_time)],[est+sigma,fliplr(est-sigma)],iir_c{fi},'FaceAlpha',0.12,'EdgeColor','none');
```
`fill` draws a filled polygon representing the ±1σ confidence band:
- `[sw_time, fliplr(sw_time)]` — x-coordinates: go forward along time, then backward (to close the polygon)
- `[est+sigma, fliplr(est-sigma)]` — y-coordinates: top of band, then bottom in reverse
- `'FaceAlpha',0.12` — 12% opacity, so bands are visible but don't hide the lines
- `'EdgeColor','none'` — no outline on the filled polygon

```matlab
    plot(sw_time,est,iir_c{fi},'LineWidth',1.5);
end
```
Overlays the actual BPM estimate line on top of the confidence band.

```matlab
legend('Ground Truth','','Butterworth','','Cheby I','','Cheby II','','Elliptic','Location','best');
```
The empty strings `''` are placeholder legend entries for the `fill` patches (which would otherwise get auto-labeled). This keeps only the `plot` lines in the legend.

---

## 25. Figure 8 — Sliding BPM vs GT, FIR

```matlab
figure('Name','Fig 8 — Sliding BPM vs GT (FIR)','Position',[30 50 1000 400]);
stairs(gt_time,gt_hr,'k','LineWidth',2.5); hold on;
fir_c  = {'b','b','b','r','m'};
fir_ls = {'--','-.', '-','-','-'};
for ii = 1:5
    fi = ii+4;
    plot(sw_time,sw_bpm(fi,:),fir_c{ii},'LineStyle',fir_ls{ii},'LineWidth',1.5);
end
```
- FIR filters are indices 5–9 in `all_sigs`. `fi = ii+4` maps loop index `ii` (1–5) to filter index `fi` (5–9).
- No confidence bands here — just the BPM lines for cleaner comparison of multiple FIR designs.
- Three blue variants for the three Hamming filters (line style differentiates them).

---

## 26. Figure 9 — Per-Window Error

```matlab
figure('Name','Fig 9 — Per-window Error','Position',[30 50 1200 1000]);
all_c    = {'b','r','g','m','c',[0 0.5 0],[0 0.4 0.8],[0.85 0.33 0],[0.5 0 0.5]};
```
Nine colors (one per filter): blue, red, green, magenta, cyan, dark green, steel blue, orange, purple. The last four are specified as `[R G B]` triplets because MATLAB's single-character color codes don't have enough variety for 9 distinct colors.

```matlab
mae_all  = mean(abs(sw_err), 2);
```
**Mean Absolute Error (MAE)** for each filter:
- `abs(sw_err)` — absolute value element-wise
- `mean(..., 2)` — mean along dimension 2 (across time windows)
- Result: `[9×1]` vector, one MAE scalar per filter

MAE = average absolute deviation from GT in BPM. Lower is better.

```matlab
rmse_all = sqrt(mean(sw_err.^2, 2));
```
**Root Mean Square Error (RMSE)** for each filter:
- `sw_err.^2` — element-wise square
- `mean(..., 2)` — mean over time
- `sqrt` — square root

RMSE penalizes large errors more than MAE (because squaring amplifies big deviations). If RMSE >> MAE, the filter has occasional large spikes even if typically accurate.

```matlab
rho_all  = arrayfun(@(fi) corr(sw_bpm(fi,:)',gt_interp'), 1:nF)';
```
**Pearson correlation coefficient** between each filter's BPM track and the GT.

- `@(fi) corr(sw_bpm(fi,:)', gt_interp')` — anonymous function computing `corr` for filter `fi`. `.T` transposes to column vectors as required by `corr`.
- `arrayfun(..., 1:nF)` — applies the function to each filter index 1 through 9.
- `'` at the end transposes the result to a column vector.

`corr` ∈ [-1, 1]: 1 = perfect positive correlation, 0 = no correlation. A filter can have low MAE but also low correlation if its errors are systematic (biased). Conversely, high correlation with high MAE indicates the filter tracks heart rate trends correctly but with a constant offset.

```matlab
for fi = 1:nF
    err_i = sw_err(fi,:);
    subplot(nF,1,fi);
```
Creates 9 vertically stacked subplots, one per filter.

```matlab
    fill([sw_time,fliplr(sw_time)], ...
         [err_i+movstd(err_i,11),fliplr(err_i-movstd(err_i,11))], ...
         all_c{fi},'FaceAlpha',0.18,'EdgeColor','none'); hold on;
    plot(sw_time,err_i,'Color',all_c{fi},'LineWidth',1.5);
```
Same fill + plot pattern as Fig 7, but now showing the **error** (not the BPM value).

```matlab
    yline(0,'k--','LineWidth',1.1);
    yline( 5,'k:','LineWidth',0.9); yline(-5,'k:','LineWidth',0.9);
    ylabel('Err (BPM)'); ylim([-60 60]); xlim([0 max(gt_time)+2]); grid on;
```
- `yline(0)` — zero error reference (perfect accuracy).
- `yline(±5)` — dotted ±5 BPM lines. Clinical guideline: ±5 BPM is often considered acceptable error for rPPG.
- `ylim([-60 60])` — fixed y-axis so all 9 subplots are visually comparable.

```matlab
    title(sprintf('%s  |  MAE=%.1f  RMSE=%.1f  r=%.3f', ...
        all_labels{fi},mae_all(fi),rmse_all(fi),rho_all(fi)),'FontSize',8);
```
Each subplot title shows the filter name plus its three performance metrics. `FontSize',8` keeps the title compact in the small subplot.

```matlab
sgtitle('Per-window Error = rPPG BPM - GT  |  dotted=+-5 BPM  |  shaded=+-1sigma','FontWeight','bold');
```
`sgtitle` (super-title) adds a single title above all subplots in the figure.

---

## 27. Summary Table

```matlab
fprintf('\n%s\n', repmat('=',1,96));
fprintf('%-32s  %4s  %7s  %7s  %+7s  %7s  %7s  %5s\n','Filter','N','BPM','GT','Error','MAE','RMSE','r');
fprintf('%s\n', repmat('-',1,96));
```
Prints a formatted ASCII table header:
- `repmat('=',1,96)` — creates a string of 96 `=` characters for the border.
- `%-32s` — left-aligned string in a 32-character field.
- `%4s`, `%7s` — right-aligned strings in 4 and 7-character fields.
- `%+7s` — string with explicit sign prefix, for the Error column.

```matlab
psd_list = {p_bw,[],p_c2,p_el,p_ham,[],[],p_ksr,p_pm};
```
Cell array mapping each filter index to its pre-computed Welch PSD (from Fig 6). `[]` (empty) for Chebyshev I, Hamming mid, and Hamming low — their PSDs weren't computed globally, so they'll be computed on the fly in the loop.

```matlab
for fi = 1:nF
    if ~isempty(psd_list{fi})
        pb_band = psd_list{fi}(band);
    else
        [pfi,~] = pwelch(all_sigs{fi}, hann(nperseg), noverlap, [], fs);
        pb_band = pfi(band);
    end
```
For each filter: if a global PSD was pre-computed, use it; otherwise compute it now.

```matlab
    [~,pi_] = max(pb_band); fp_=f_p(band); fp=fp_(pi_);
```
Finds the peak frequency in the cardiac band: `fp` is the frequency in Hz where the global PSD peaks. This is the "single-number" BPM estimate from the full-duration signal.

```matlab
    if fi==5; fprintf('%s\n',repmat('-',1,96)); end
```
Prints a horizontal separator between IIR (filters 1–4) and FIR (filters 5–9) sections of the table.

```matlab
    fprintf('%-32s  %4d  %7.1f  %7.1f  %+7.1f  %7.1f  %7.1f  %5.3f\n', ...
        all_labels{fi}, numel(all_sigs{fi})-1, fp*60, bpm_gt_mean, ...
        fp*60-bpm_gt_mean, mae_all(fi), rmse_all(fi), rho_all(fi));
```
One row per filter:
- `all_labels{fi}` — filter name
- `numel(all_sigs{fi})-1` — filter order (number of taps minus 1 for FIR; for IIR, this prints the signal length minus 1, which is not the filter order — a minor inconsistency in the script)
- `fp*60` — detected BPM from global PSD
- `bpm_gt_mean` — ground truth mean BPM
- `fp*60-bpm_gt_mean` — single-number BPM error
- `mae_all(fi)`, `rmse_all(fi)`, `rho_all(fi)` — sliding-window metrics

```matlab
fprintf('GT: mean=%.1f  median=%.1f  std=%.1f  range=[%d %d]  n=%d\n', ...
    bpm_gt_mean, bpm_gt_median, std(double(gt_hr)), min(gt_hr), max(gt_hr), numel(gt_hr));
```
Prints complete GT statistics below the table for reference.

```matlab
[~,bi]=min(mae_all(1:4));
[~,bf]=min(mae_all(5:9)); bf=bf+4;
fprintf('\nBest IIR: %s  MAE=%.1f BPM  N=%d\n', all_labels{bi},mae_all(bi),numel(all_sigs{bi})-1);
fprintf('Best FIR: %s  MAE=%.1f BPM  N=%d\n',  all_labels{bf},mae_all(bf),numel(all_sigs{bf})-1);
```
- `min(mae_all(1:4))` — finds the IIR filter with the lowest MAE.
- `min(mae_all(5:9))` — finds the best FIR filter. `bf=bf+4` corrects the index back to the full array numbering.
- Prints a "winner" summary for IIR and FIR separately.

---

## 28. Helper Function: sosfreqz

```matlab
function H = sosfreqz(sos, g, f, fs)
% Cascade freqz section-by-section; freqz always returns a column vector.
    H = ones(numel(f), 1);
    for k = 1:size(sos, 1)
        H = H .* freqz(sos(k,1:3), sos(k,4:6), f(:), fs);
    end
    H = g .* H;
end
```

This function computes the frequency response of an IIR filter stored in **Second-Order Sections (SOS)** form at arbitrary frequencies.

**Why is this function needed?**  
MATLAB's built-in `freqz` function can accept SOS form, but only evaluates at uniformly-spaced frequencies (specified as an integer count, not a vector). Here we need the response at 8192 specific frequency points stored in `f_zoom`. So this helper manually cascades the biquad sections.

**Line-by-line:**

```matlab
H = ones(numel(f), 1);
```
Initialize H as a column vector of ones, same length as `f`. This is the identity for multiplication — `H` will accumulate the cascade product.

```matlab
for k = 1:size(sos, 1)
```
Loop over each second-order section (row) of the SOS matrix. `size(sos,1)` = number of biquad sections.

```matlab
    H = H .* freqz(sos(k,1:3), sos(k,4:6), f(:), fs);
```
For section `k`:
- `sos(k,1:3)` — numerator coefficients `[b0, b1, b2]` of the k-th biquad
- `sos(k,4:6)` — denominator coefficients `[a0, a1, a2]` of the k-th biquad
- `freqz(..., f(:), fs)` — evaluates this biquad's frequency response at the frequencies in `f` (in Hz, since `fs` is provided)
- `f(:)` ensures `f` is a column vector, since `freqz` returns a column — element-wise multiplication requires matching shapes
- `.*` — element-wise (pointwise) complex multiplication accumulates the cascade

The mathematical basis: for a cascade of filters H₁(f)·H₂(f)·...·Hₘ(f), the total response at each frequency is the product of individual responses. In complex arithmetic this multiplies both magnitudes and rotates phases.

```matlab
H = g .* H;
```
Applies the overall gain scalar `g` to the cascaded response. In SOS form, any gain that doesn't fit into the biquad coefficients is stored separately in `g`. This final multiplication gives the complete, correctly scaled frequency response.

---

## Summary of the Full Script Flow

```
                    ┌──────────────────────────────────────────────────┐
                    │              bpm_control_filterdesign.m           │
                    └──────────────────────────────────────────────────┘
                                          │
         ┌────────────────────────────────┼────────────────────────────┐
         ▼                                ▼                            ▼
   Load GT vitals              Read video (frame-by-frame)        Define filter
   (CSV → gt_time,             → Face ROI → skin mask             specification
   gt_hr, stats)               → YCbCr → spatial mean             (f_p1,f_p2,
                                → R_t, G_t, B_t                   Rp, Rs)
                                          │
                               CHROM projection
                               → S = Xs - α·Ys
                               → Linear detrend
                               → S_det
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
        IIR Filters                  FIR Filters (Hamming)      FIR Filters
     (Butterworth,                (N=51, 151, N_est)          (Kaiser, Parks-
      Cheby I, II,                                             McClellan)
      Elliptic)
      SOS form                     b_ham_lo/mid/hi             b_ksr, b_pm
              │                           │                           │
              └───────────────────────────┼───────────────────────────┘
                                          ▼
                              Frequency response plots
                              (Magnitude, Phase, Group Delay)
                              [Figures 1–4]
                                          │
                              filtfilt (zero-phase) applied
                              → 9 filtered BVP signals
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                    Time domain       PSD (global)    Sliding window
                    [Fig 5]           [Fig 6]         BPM (10s, 1s step)
                                                      → MAE, RMSE, r
                                                      [Figs 7, 8, 9]
                                                      [Table]
```

---

## Key Design Concepts Reference

| Concept | What it means in this script |
|---------|------------------------------|
| **SOS form** | IIR filters stored as cascaded biquads — avoids numerical overflow in high-order polynomial coefficients |
| **filtfilt** | Zero-phase filtering via forward + reverse pass — doubles the effective order, cancels all phase |
| **Welch PSD** | Averaged, windowed periodogram — reduces variance in spectral estimate; used both globally (Fig 6) and in each sliding window |
| **Kaiser formula** | Analytic estimate of required FIR order given transition bandwidth and stopband attenuation |
| **Parks-McClellan** | Minimax-optimal equiripple FIR design — best stopband performance for a given order |
| **Group delay** | Frequency-domain latency: FIR = constant = N/2 samples; IIR = varies, peaks near filter edges |
| **MAE vs RMSE** | MAE = average error; RMSE = penalizes large outliers more; both in BPM |
| **Pearson r** | Correlation with GT — high r means filter tracks trends even if offset is present |

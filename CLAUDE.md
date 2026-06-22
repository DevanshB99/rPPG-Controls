# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

<<<<<<< HEAD
This is a **MATLAB research implementation** of a Remote Photoplethysmography (rPPG) pipeline — non-contact heart rate extraction from face video recordings. Developed at the MACS Lab (University of Washington) as part of the DocBOT project. All algorithms are derived from and keyed to equations in `rPPG_Mathematical_Foundation.pdf` / `.tex`.

## Running the Code

All scripts run in MATLAB R2019a+. Open MATLAB, `cd` to this directory, then run a script by name:

```matlab
bpm_controls               % Full pipeline: raw video → BPM estimate
bpm_control_filterdesign   % IIR filter comparison with ground truth validation
bpm_controls_FDA           % Frequency domain analysis (FFT / Welch / STFT)
% Face Segmentation/
FaceSegment_HoG_LBP        % Advanced adaptive face segmentation pipeline
```

**Before running**, edit the hardcoded paths near the top of each script:
- `IMG_PATH` — input JPEG image (used for Bayer demosaicing visualization)
- `VID_PATH` — input `.mov` video file (30+ fps QuickTime)
- `CSV_PATH` — DocBOT ground-truth vitals CSV (`offset_seconds`, `heart_rate` columns)

### Required MATLAB Toolboxes
- Computer Vision Toolbox (face detection, landmark extraction)
- Image Processing Toolbox
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox (for MTCNN in `FaceSegment_HoG_LBP.m`)

## Architecture

### Main Pipeline (`bpm_controls.m`)
Implements the full rPPG signal chain in order:
1. **ADC analysis** — DN value histograms for raw RGB channels (Eqs. 8–9)
2. **Bayer demosaicing** — RGGB pattern channel separation (Eqs. 10–21)
3. **YCbCr conversion** — ITU-R BT.601 color space transform (Eqs. 22–23)
4. **Skin detection** — YCbCr threshold mask: `77≤Cb≤127, 133≤Cr≤173, Y>40` (Eq. 25)
5. **Face ROI** — Viola-Jones cascade detector; final mask = face ROI AND YCbCr skin mask
6. **Signal extraction** — Frame-by-frame spatial mean over skin pixels (Eqs. 37–38)
7. **Temporal normalization** — DC removal via division by channel mean (Eqs. 39–41)
8. **CHROM projection** — Xs/Ys signals → BVP waveform (Eqs. 42–49)
9. **Detrending** — Least-squares linear drift removal (Eqs. 51–54)
10. **Bandpass filter** — 4th-order Butterworth, 0.67–3.0 Hz cardiac band (Eqs. 55–57)
11. **Welch PSD** — Power spectral density estimation (Eqs. 58–62)
12. **Peak detection + harmonic correction** — Identifies dominant cardiac frequency (Eqs. 61–64)

### Filter Design Analysis (`bpm_control_filterdesign.m`)
Compares Butterworth, Chebyshev I, Chebyshev II, and Elliptic filters across: frequency response (magnitude/phase/group delay), sliding-window BPM estimates (10 s windows, 1 s step), and error metrics vs ground truth (MAE, RMSE, correlation). Includes **frame-by-frame luminance normalization** to compensate camera AGC/auto-exposure drift.

### Frequency Domain Analysis (`bpm_controls_FDA.m`)
Explores time–frequency resolution trade-offs via FFT, Welch, and STFT at varying window lengths (5 s / 10 s / full signal). Motivates MUSIC subspace method for short-window high-resolution estimation.

### Face Segmentation (`Face Segmentation/FaceSegment_HoG_LBP.m`)
Multi-stage adaptive segmentation beyond simple YCbCr thresholding:
- MTCNN face detection + facial landmark localization
- Feature region exclusion (eyes, brows, nose, lips)
- **Voting across 4 color spaces:** adaptive YCbCr (left/right split), RGB rules, HSV hue range, shadow compensation — pixel counted as skin if 2+ methods agree
- **Texture analysis:** HoG gradient energy + LBP variance to reject hair/background
- **GMM clustering** (K=2–4 selected by BIC) to separate skin/hair/mixed clusters
- Outputs labeled ROIs: Forehead, Left/Right Temple, Left/Right Cheek

## Key Data Files

Measurement data lives in `Measurement Data/` — JSON outputs from the DocBOT system containing per-frame camera metadata (ISO, aperture, shutter speed, white balance), ROI luminance statistics, and face detection confidence. The CSV files provide ground-truth heart rate for validation.

## Mathematical Reference

All equation numbers in comments (`% Eq. XX`) refer to `rPPG_Mathematical_Foundation.pdf` in the project root. The `.tex` source is also present. When modifying signal processing logic, cross-reference the relevant equations to maintain traceability.
=======
This is a MATLAB research project implementing **Remote Photoplethysmography (rPPG)** — contactless heart rate measurement from RGB video. The project compares multiple frequency estimation methods (FFT, Welch PSD, MUSIC, ESPRIT) for extracting heart rate (BPM) from facial video.

This is a collaboration between Devansh Bajwala and Prof. Xu Chen, who is upgrading the system with advanced subspace-based frequency estimation methods (MUSIC/ESPRIT) from controls/precision engineering theory.

## Running the Code

All scripts are standalone MATLAB `.m` files — open MATLAB and run them directly. No build system or dependencies to install.

- **Main pipeline:** `bpm_controls.m` (or the live script `bpm_controls.mlx`)
- **ESPRIT variants:** `bpm_control_ESPRIT.m`
- **Filter design comparison:** `bpm_control_filterdesign.m`
- **FDA (FFT/Welch/STFT) comparison:** `bpm_controls_FDA.m`
- **MUSIC algorithm:** `bpm_estimate_MUSIC.m`

**Required MATLAB Toolboxes:** Signal Processing, Image Processing, Computer Vision (Vision)

**Data paths** in scripts are hardcoded — update paths to point at videos/images in `Measurement Data/` before running.

## Signal Processing Pipeline

The core 7-stage pipeline (implemented in `bpm_controls.m`):

1. **Face detection & ROI** — Viola-Jones detector; falls back to manual bounding box
2. **RGB extraction** — Per-frame spatial mean over skin pixels → R(t), G(t), B(t)
3. **Skin detection** — YCbCr thresholds: Cb∈[77,127], Cr∈[133,173], Y>40
4. **CHROM projection** — `Xs = 3R̂ - 2Ĝ`, `Ys = 1.5R̂ + Ĝ - 1.5B̂`, `S = Xs - α·Ys` (α = std(Xs)/std(Ys))
5. **Detrending** — Linear drift removal via least squares
6. **Bandpass filter** — 4th-order Butterworth, 0.67–3.5 Hz, zero-phase (`filtfilt`)
7. **Frequency estimation** — Peak of PSD in cardiac band → BPM

## Frequency Estimation Methods

| Method | File | Notes |
|--------|------|-------|
| Welch PSD | `bpm_controls.m` | Production baseline; Hann window, 50% overlap |
| FFT / STFT | `bpm_controls_FDA.m` | Multiple window length comparisons |
| MUSIC | `bpm_estimate_MUSIC.m` | Subspace method; higher resolution with fewer samples |
| ESPRIT (3 variants) | `bpm_control_ESPRIT.m` | K=1 analytic (Hilbert), K=2 LS, K=2 TLS |

**Core research question (Prof. Chen):** MUSIC/ESPRIT can extract frequency accurately from much shorter windows (2–3 sec vs 10 sec for Welch), potentially reducing measurement latency significantly.

## Research Roadmap (Prof. Chen's 4 Parts)

1. **Filter Design** — Compare FIR vs IIR, Butterworth vs Chebyshev II; target 0.7–3.5 Hz
2. **Frequency Domain Analysis** — Minimize window length while maintaining accuracy (FFT/Welch/STFT)
3. **MUSIC/ESPRIT** — High-resolution subspace frequency estimation; quantify minimum data requirements
4. **Confidence Scoring** — SNR + temporal stability + skin pixel count; integrate with DocBot control loop

## Key Reference Files

- `rPPG_Mathematical_Foundation.tex/.pdf` — Full mathematical derivation of entire signal chain (image sensor physics through spectral estimation, ~700 equations)
- `Technical Documents/rPPG_MUSIC_Analysis_and_Roadmap.md` — Prof. Chen's roadmap and MUSIC/ESPRIT explanation
- `Technical Documents/` — Per-trial analysis results for each method variant
- `Measurement Data/DocBOT_2026-04-07_*/` — Video recordings + ground truth vitals (CSV/JSON)

## Harmonic Correction

All frequency estimators include a **half-frequency artifact check**: if the detected peak is below ~1.0 Hz, check whether 2× that frequency has a strong peak (indicating the true BPM was detected at its first harmonic). This prevents systematic 2× underestimation errors.
>>>>>>> 8f086f4 (WIP)

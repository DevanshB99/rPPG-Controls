# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

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

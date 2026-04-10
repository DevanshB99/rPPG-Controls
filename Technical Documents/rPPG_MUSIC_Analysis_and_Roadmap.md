# rPPG — MUSIC Algorithm Analysis, Document Review & Implementation Roadmap

**Author:** Devansh Bajwala  
**Date:** 2026-04-05  
**Professor:** Prof. Xu Chen  
**Context:** DocBot rPPG Heart Rate Extraction — Research Upgrade

---

## Table of Contents

1. [Big Picture — What Professor Chen Is Asking](#1-big-picture)
2. [Document 1 — Professor's 4-Part Research Roadmap](#2-document-1-professors-roadmap)
3. [Document 2 — MUSIC Algorithm from Absolute First Principles](#3-document-2-music-algorithm)
   - [Step 0: Why Not Just Use FFT?](#step-0-why-not-just-use-fft)
   - [Step 1: The Signal Model](#step-1-the-signal-model)
   - [Step 2: Where Does K=1 Come From?](#step-2-where-does-k1-come-from)
   - [Step 3: The Data Vector and Steering Matrix](#step-3-the-data-vector-and-steering-matrix)
   - [Step 4: The Covariance Matrix](#step-4-the-covariance-matrix)
   - [Step 5: Eigendecomposition — Separating Signal from Noise](#step-5-eigendecomposition)
   - [Step 6: The MUSIC Pseudo-Spectrum](#step-6-the-music-pseudo-spectrum)
   - [Step 7: Complete MUSIC Algorithm](#step-7-complete-music-algorithm)
   - [Step 8: Design Parameter M](#step-8-design-parameter-m)
   - [Step 9: MUSIC vs Welch Comparison](#step-9-music-vs-welch-comparison)
   - [Step 10: The Controls Connection](#step-10-the-controls-connection)
4. [Document 3 — Mathematical Foundation (rPPG_Mathematical_Foundation.pdf)](#4-document-3-mathematical-foundation)
5. [Current Implementation — bpm_estimate.py](#5-current-implementation-bpm_estimatepy)
6. [Gap Analysis — Current vs Professor's Requirements](#6-gap-analysis)
7. [MATLAB Implementation Roadmap — bpm_controls.mlx](#7-matlab-implementation-roadmap)
8. [Full Pipeline Summary Map](#8-full-pipeline-summary-map)

---

## 1. Big Picture

Prof. Xu Chen is doing two things simultaneously:

1. **Auditing** the current rPPG implementation and giving it a research-grade upgrade roadmap
2. **Introducing MUSIC** (Multiple Signal Classification) — a superior frequency identification algorithm from his own research in controls/precision engineering — as a replacement and comparison for Welch's PSD method

### The Core Critique of the Current System

> "You are using Welch's method to find the heart rate frequency. This works, but it is a blunt instrument. It needs lots of data (long windows), has poor frequency resolution, and gets confused by noise. MUSIC can extract the correct frequency from far fewer samples with far higher resolution, even in noisy conditions."

### The Framing: A Controls Problem

Prof. Chen frames rPPG as a **controls/signal identification problem**. The heartbeat is a narrow-band periodic signal (like a sinusoid) buried in a noisy channel — exactly like mechanical vibration disturbances in precision engineering systems (his domain: hard disk drives, active suspensions).

The same mathematical tools used to identify and reject vibrations in precision mechatronics can extract the heart rate from a noisy camera signal. This is the intellectual bridge he is drawing.

---

## 2. Document 1 — Professor's 4-Part Research Roadmap

### Section 1: Filter Design and Signal Pre-Processing

**Core Objective:** Design bandpass filters to isolate physiological heart rate range.  
**Target Range:** **0.7 Hz to 3.5 Hz** (42 to 210 BPM)

#### Current Implementation
```python
b, a = butter(4, [low, high], btype="band")   # IIR Butterworth, order 4
h = filtfilt(b, a, h)                          # zero-phase forward-backward filtering
```

#### What Professor Wants
Compare two classes of filters:

| Property | FIR Filters | IIR Filters |
|---|---|---|
| Examples | Windowed sinc, Parks-McClellan | Butterworth, Chebyshev, Elliptic |
| Phase response | Linear (no temporal distortion) | Non-linear (can distort waveform) |
| Filter order needed | High (100s of taps) | Low (4–8) |
| Computational cost | High | Low |
| Transition band sharpness | Parks-McClellan = optimal | Elliptic = sharpest |
| Stability | Always stable | Can be unstable if poorly designed |

**For rPPG:** Linear phase matters because the temporal shape of the blood volume pulse (BVP) waveform carries diagnostic information. IIR `filtfilt` achieves zero-phase by running the filter twice (forward + backward), at the cost of double computation.

**Tool:** MATLAB Filter Design Toolbox. Reference: Prof. Chen's EJC 2013 paper, Figure 4.

---

### Section 2: Frequency Domain Analysis

**Core Objective:** Transform the time-domain rPPG signal into frequency domain to extract heart rate.

**Methods to Compare:**

1. **FFT (Fast Fourier Transform)** — fast, simple, poor resolution for short signals
2. **Welch's Method (PSD estimation)** — averages multiple overlapping FFTs, more stable but still limited resolution
3. **STFT (Short-Time Fourier Transform)** — time-frequency representation, shows how the frequency changes over time

**Key Research Question:** How short can the analysis window be while still achieving accurate BPM?

- Current implementation: 10-second Welch window → ~6 BPM frequency resolution
- Goal: Can MUSIC achieve the same accuracy with a 2–3 second window?

A shorter window means **lower latency** in the BPM reading. This is clinically significant.

---

### Section 3: Rapid Peak Detection — MUSIC

**Core Objective:** Extract heart rate rapidly using much less data than conventional methods.

**Method:** MUSIC and ESPRIT — subspace-based frequency estimators.

**Key Claims:**
- Works with far fewer frames than Welch's method
- Dramatically higher frequency resolution
- Robust to noise through explicit signal/noise subspace separation
- Identifies the minimum data window needed for accurate frequency extraction

> "Frequency identification algorithms such as MUSIC and ESPRIT are excellent at the job."  
> — Prof. Xu Chen

---

### Section 4: Confidence Scoring and Sensor Orchestration

**Core Objective:** DocBot must self-evaluate the reliability of the heart rate it measures.

**Multi-Parameter Confidence Score (all three required):**

1. **SNR** — Signal-to-Noise Ratio of the dominant cardiac peak relative to surrounding frequency bins
2. **Temporal Stability** — Does the BPM estimate stay consistent across successive sliding windows?
3. **Skin Pixel Count** — How many confirmed skin pixels contributed to the spatial average?

**System Integration:** The confidence score feeds into the **DocBot robot control loop**. If confidence falls below a threshold, the robot physically repositions its camera to improve the line-of-sight to the patient's face — a closed-loop control architecture where signal quality drives actuator commands.

---

## 3. Document 2 — MUSIC Algorithm from Absolute First Principles

---

### Step 0: Why Not Just Use FFT?

The DFT/FFT has a fundamental limitation: **frequency resolution is inversely proportional to signal length.**

```
Frequency resolution: Δf = fs / N
```

For a 30 fps camera with 150 frames (5 seconds):
```
Δf = 30 / 150 = 0.2 Hz = 12 BPM resolution
```

This means you can only distinguish heart rates that differ by more than 12 BPM — clinically unacceptable.

**Furthermore, DFT fails with:**
- Short signals (few samples)
- Two frequencies that are close together
- Significant noise (peaks shift, false peaks appear)

**Prof. Chen's example from the document:**
- True frequencies: 504 Hz and 648 Hz
- Signal length: 440 samples

| Method | Frequency 1 | Frequency 2 |
|---|---|---|
| True values | 504 Hz | 648 Hz |
| DFT (clean signal) | 517.6 Hz | 621.2 Hz |
| DFT (noisy signal) | 517.6 Hz | 672.9 Hz |
| MUSIC (noisy signal) | ~504 Hz | ~648 Hz |

**Conclusion:** DFT fails badly in noise. MUSIC recovers the true frequencies even from the noisy signal. A high-resolution technique is required.

---

### Step 1: The Signal Model

MUSIC begins by modeling the signal mathematically. The fundamental equation is:

```
y[n] = Σᵢ₌₁ᴷ αᵢ · e^(j(ωᵢn + φᵢ)) + v[n]
```

**Every term unpacked:**

| Symbol | Meaning |
|---|---|
| `y[n]` | Measured signal at time step n (the rPPG signal after CHROM processing) |
| `K` | Number of sinusoidal components (frequencies) buried in the signal |
| `αᵢ` | Amplitude of the i-th sinusoid (how strong is that frequency component?) |
| `ωᵢ` | Frequency of the i-th sinusoid in **radians per sample** |
| `φᵢ` | Initial phase (random, unknown — we don't need to know it) |
| `v[n]` | Additive white Gaussian noise |

**Physical frequency to ω conversion:**
```
ω = 2π · f_Hz · Tₛ = 2π · f_Hz / fs
```

Example for 72 BPM (1.2 Hz) at 30 fps:
```
ω = 2π × 1.2 / 30 = 0.251 rad/sample
```

**Why complex exponentials instead of cosines?**  
Because `e^(jωn) = cos(ωn) + j·sin(ωn)`. The complex form makes all the matrix algebra clean and tractable. For a real-valued rPPG signal, you either work with the analytic signal (Hilbert transform) or simply acknowledge the math is presented in complex form.

---

### Step 2: Where Does K=1 Come From?

**K is the number of sinusoidal frequency components you expect in your signal.**

For rPPG:
- The heart beats at **one fundamental frequency** — your heart rate (e.g., 1.2 Hz = 72 BPM)
- This is ONE narrow-band sinusoidal component
- Therefore **K = 1**

**Why not K=2 or K=3?** Harmonics (2f, 3f of the cardiac signal) exist in the raw BVP signal, but after bandpass filtering to 0.7–3.5 Hz, you are specifically hunting for the single dominant cardiac fundamental frequency. Even if harmonics are present, setting K=1 directs MUSIC to find the one dominant frequency.

**In Prof. Chen's original context (hard disk drives):** K=2 because there were two mechanical vibration frequencies (504 Hz and 648 Hz) to identify simultaneously.

**How to estimate K if unknown:** Look at the eigenvalues of the covariance matrix. There will be K clearly large eigenvalues (signal) and M-K small, nearly equal eigenvalues (noise). The "gap" between eigenvalue K and eigenvalue K+1 reveals the true K.

---

### Step 3: The Data Vector and Steering Matrix

MUSIC does not process one sample at a time. It stacks M consecutive samples into a **column vector**:

```
y[n] = [y[n], y[n-1], y[n-2], ..., y[n-M+1]]ᵀ   ∈ ℂᴹˣ¹
```

Think of M as a small "snapshot" window sliding through your signal. At each position n, you have an M-dimensional observation vector.

**The Vandermonde (Steering) Matrix A:**

```
        ┌  1        1       ...   1       ┐
        │  e^(-jω₁) e^(-jω₂) ... e^(-jωₖ)│
A =     │  ⋮         ⋮      ⋱    ⋮        │    ∈ ℂᴹˣᴷ
        └  e^(-j(M-1)ω₁)  ...  e^(-j(M-1)ωₖ) ┘
```

Each column is a **steering vector** for one frequency:
```
a(ωᵢ) = [1, e^(-jωᵢ), e^(-j2ωᵢ), ..., e^(-j(M-1)ωᵢ)]ᵀ
```

The steering vector is the "fingerprint" of how a sinusoid at frequency ωᵢ appears across the M dimensions of the observation vector. Every frequency has a unique steering vector — this uniqueness is what MUSIC exploits.

**The full signal model becomes:**
```
y[n]    =   A    ·   x[n]   +   v[n]
(M×1)     (M×K)    (K×1)      (M×1)
```

---

### Step 4: The Covariance Matrix

**This is the single most important mathematical step.**

We compute the **covariance matrix** of the observation vector y[n]:

```
Rᵧ = E{y[n] · y[n]*} = A · Rₓ · A* + σ²·I
```

**Where:**
- `E{·}` = statistical expectation (average over time)
- `y[n]*` = conjugate transpose of y[n]
- `Rₓ = diag(α₁², α₂², ..., αₖ²)` = diagonal matrix of signal powers
- `σ²` = noise power (variance of the white noise v[n])
- `I` = M×M identity matrix

**Why is Rₓ diagonal?**  
Because the phases φᵢ are assumed to be independently and uniformly distributed on [-π, π]. This causes the cross-correlation between different sinusoids to average to zero:

```
E{αᵢ·e^(j(ωᵢn+φᵢ)) · (αₗ·e^(j(ωₗn+φₗ)))*} = 0   when i ≠ l
```

Only same-frequency terms survive (i = l), giving power αᵢ². So Rₓ is diagonal.

**Physical meaning of Rᵧ:**

The covariance matrix Rᵧ has two contributions:
1. `A·Rₓ·A*` — the **signal contribution** (rank K — only K directions in M-space are energized by the signals)
2. `σ²·I` — the **noise contribution** (adds σ² equally and uniformly to all M directions)

---

### Step 5: Eigendecomposition — Separating Signal from Noise

We compute the eigenvalues of Rᵧ. Due to the structure above:

```
Eigenvalues of Rᵧ:
  λᵢ = μᵢ + σ²   for i = 1, 2, ..., K     ← SIGNAL eigenvalues (large, distinct)
  λᵢ = σ²        for i = K+1, K+2, ..., M  ← NOISE eigenvalues (small, equal)
```

**The key insight:** White noise adds σ² equally to every eigenvalue. The signal part only "pumps up" K of them. So:

- **K large, distinct eigenvalues** → signal subspace
- **M-K small, approximately equal eigenvalues** → noise subspace (all ≈ σ²)

The eigenvector matrix is partitioned:
```
Q = [q₁ q₂ ... qₖ | qₖ₊₁ ... q_M]
         Qₛ (M×K)       Qₙ (M×(M-K))
```

- **Qₛ** = signal eigenvectors — span the signal subspace
- **Qₙ** = noise eigenvectors — span the noise subspace

**These two subspaces are orthogonal to each other.**

**The crucial geometric fact:**  
The steering vector `a(ωᵢ)` for any true signal frequency lies EXACTLY in the signal subspace, and is therefore PERPENDICULAR to the noise subspace:

```
Qₙ* · a(ωᵢ) = 0   for any true signal frequency ωᵢ
```

This orthogonality is the mathematical engine of MUSIC.

---

### Step 6: The MUSIC Pseudo-Spectrum

Since `Qₙ* · a(ω) = 0` at true signal frequencies, the quantity `‖Qₙ* · a(ω)‖²` will be:
- **Near zero** at the true signal frequencies ωᵢ
- **Large and positive** everywhere else

Therefore define the **MUSIC pseudo-spectrum**:

```
P_MUSIC(ω) =         1
              ─────────────────────────
              a(ω)* · Qₙ · Qₙ* · a(ω)

           =         1
              ─────────────────────────
              ‖Qₙ* · a(ω)‖²
```

This function has **sharp, narrow peaks exactly at the true signal frequencies.**

**Why is this better than DFT?**  
DFT smears each frequency into a wide sinc-shaped lobe. MUSIC uses the geometric fact of subspace orthogonality to produce infinitely-sharp (in theory) peaks. The resolution is limited only by the amount of data used to estimate Rᵧ, not by a fixed FFT bin spacing.

---

### Step 7: Complete MUSIC Algorithm

```
═══════════════════════════════════════════════════════════════
MUSIC ALGORITHM — STEP BY STEP
═══════════════════════════════════════════════════════════════

INPUT:  N samples of signal y[n]
        Design parameter M (choose M > K; rule of thumb M = K³ or 
        larger, e.g. M = 10–30 for rPPG with K=1)
        Number of signals K (K=1 for rPPG)
        Frequency sweep range [ω_low, ω_high]

─────────────────────────────────────────────────────────────
STEP 1: Collect N samples
        y[0], y[1], ..., y[N-1]
─────────────────────────────────────────────────────────────
STEP 2: Form the M×M sample covariance matrix

        R̂ᵧ = 1/(N-M+1) · Σ y[n]·y[n]*
                           n=M-1 to N-1

        where each y[n] = [y[n], y[n-1], ..., y[n-M+1]]ᵀ

─────────────────────────────────────────────────────────────
STEP 3: Eigendecompose R̂ᵧ

        R̂ᵧ = Q · Λ · Q*

        Sort eigenvalues: λ₁ ≥ λ₂ ≥ ... ≥ λ_M
        Extract noise eigenvectors: Qₙ = [qₖ₊₁, qₖ₊₂, ..., q_M]

─────────────────────────────────────────────────────────────
STEP 4: Estimate K (if unknown)

        Plot eigenvalues in order.
        The "elbow" — where eigenvalues drop sharply from large 
        to small — reveals K.
        For rPPG: K=1. Expect 1 large eigenvalue, M-1 small ones.

─────────────────────────────────────────────────────────────
STEP 5: Sweep ω across physiological band

        For each candidate ω in [ω_low, ω_high]:
            Construct steering vector:
                a(ω) = [1, e^(-jω), e^(-j2ω), ..., e^(-j(M-1)ω)]ᵀ

            Compute MUSIC pseudo-spectrum:
                P_MUSIC(ω) = 1 / (a(ω)* · Qₙ · Qₙ* · a(ω))

        Find K highest peaks of P_MUSIC(ω).

─────────────────────────────────────────────────────────────
OUTPUT: Estimated frequencies ω̂₁, ..., ω̂ₖ

        Convert to Hz:    f̂ᵢ = ω̂ᵢ / (2π · Tₛ) = ω̂ᵢ · fs / (2π)
        Convert to BPM:   BPM = f̂₁ × 60
═══════════════════════════════════════════════════════════════
```

**rPPG-specific parameters:**
- K = 1
- Sweep range: ω ∈ [2π × 0.7/fs, 2π × 3.5/fs] (physiological band in rad/sample)
- fs = actual frame rate (computed from timestamps, not assumed 30)

---

### Step 8: Design Parameter M

The footnote in the document states: **M = K³ gives good performance** for high-sampling-rate systems.

For rPPG with K=1:
- M = 1³ = 1 is trivially too small
- The tradeoff is:

| Smaller M | Larger M |
|---|---|
| Better statistical estimate of R̂ᵧ | Better signal/noise subspace separation |
| Less data required | Sharper MUSIC peaks |
| Less computation | More computation |

**Recommended starting range for rPPG:** M = 10 to 30. Experiment in MATLAB to find the optimal value that gives consistent BPM with minimal data.

The relationship between M and minimum required data N:
```
Need at least N > M samples (more is better for covariance estimation)
Rule of thumb: N ≥ 10 × M
```

---

### Step 9: MUSIC vs Welch Comparison

| Property | Welch PSD | MUSIC |
|---|---|---|
| Frequency resolution | Limited by window length: Δf = fs/L | Super-resolution — not FFT-bin limited |
| Minimum data needed | ~10 seconds (300 frames at 30fps) | ~2–3 seconds (60–90 frames) |
| Noise handling | Averages noise (reduces but cannot separate) | Explicitly separates signal/noise subspaces |
| Frequency accuracy | Quantized to FFT bins | Continuous — finds sub-bin exact frequencies |
| Requires knowing K | No — finds all peaks | Yes — must specify K (or estimate it) |
| Computational cost | O(N log N) — very fast | O(M³) for eigendecomposition |
| Handles colored noise | Poorly | Better (if noise model is accurate) |
| Implementation complexity | Simple | Moderate |

**Bottom line:** MUSIC is the right tool when you need fast response (short windows) and high precision. Welch is simpler and sufficient for long, clean signals.

---

### Step 10: The Controls Connection

Prof. Chen's controller diagram shows:

```
r=0 ──→ [C_FB] ──u(k)──→ [Plant] ──+──→ z(k)
              ↑                      |
             [DOB]←──────────────────|
              |                      |
         [IMP-based              [Bandpass]
          frequency                Filter
          estimation]
              |
            d̂(k)
```

**In precision engineering:**
- `d(k)` = narrow-band vibration disturbance at unknown frequencies
- `DOB` estimates `d(k)` from sensor readings  
- MUSIC identifies the exact disturbance frequencies from `d̂(k)`
- IMP (Internal Model Principle) synthesizes a perfect rejection controller tuned to those exact frequencies

**The rPPG analogy:**

| Controls Domain | rPPG Domain |
|---|---|
| Narrow-band disturbance d(k) | Blood volume pulse signal (what we want) |
| White noise v(k) | Camera noise, motion artifacts, lighting flicker |
| DOB estimate d̂(k) | CHROM-extracted rPPG signal |
| MUSIC frequency identification | Cardiac frequency extraction |
| IMP controller | BPM display + confidence score |
| Actuator repositioning | DocBot camera repositioning |

The robot's camera becomes a **feedback actuator** in a closed-loop system: poor signal quality (low confidence) drives the robot to reposition, improving the signal, closing the loop.

---

## 4. Document 3 — Mathematical Foundation (rPPG_Mathematical_Foundation.pdf)

This is the existing comprehensive technical documentation covering the complete rPPG pipeline from first principles. It is mathematically rigorous and correct.

### Pipeline Covered

```
Photons → Image Sensor → Bayer CFA → Demosaicing → RGB Image
       → YCbCr Conversion → Skin Detection → BiSeNet Segmentation
       → Combined Mask → CHROM Algorithm → Detrending
       → Butterworth Bandpass → Welch PSD → Peak Detection
       → Harmonic Correction → BPM
```

### Key Equations

**Quantum efficiency and photon-to-electron:**
```
η(λ): ηblue ≈ 0.35,  ηgreen ≈ 0.50,  ηred ≈ 0.25
```

**ADC quantization (8-bit):**
```
DN = floor((Vpixel - Voffset) / Vref × 255)
```

**YCbCr skin detection:**
```
is_skin(Y, Cb, Cr) = 1   if:  77 ≤ Cb ≤ 127
                               133 ≤ Cr ≤ 173
                               Y > 40
```

**CHROM algorithm:**
```
Xₛ(t) = 3R̂(t) - 2Ĝ(t)
Yₛ(t) = 1.5R̂(t) + Ĝ(t) - 1.5B̂(t)
α = σ(Xₛ) / σ(Yₛ)
S(t) = Xₛ(t) - α·Yₛ(t)
```

**Welch PSD peak detection:**
```
fpeak = argmax   PSD(f)
        f∈[0.67, 3.0]

BPM = 60 × fpeak
```

**Harmonic correction:**
```
If PSD(2fpeak) / PSD(fpeak) > 0.3:  use fcorrected = 2fpeak
```

### Upgrade Point

Section 8 (Welch's Method) is where MUSIC will be added. The filter design in Section 7 (Butterworth only) will be expanded with FIR comparison. The confidence score does not yet incorporate temporal stability or skin pixel count.

---

## 5. Current Implementation — bpm_estimate.py

### Architecture

```
CameraThread → frames → InferenceWorker (YOLO + BiSeNet + CHROM) → MainWindow (UI)
```

### Signal Processing Pipeline in Code

```python
# Step 1: Per-channel linear detrend (removes slow illumination drift)
for c in range(3):
    rgb[:, c] = detrend(rgb[:, c], type="linear")

# Step 2: CHROM algorithm (BGR → RGB, then project)
rgb_signal = rgb[:, ::-1].copy()
h = chrom_rppg(rgb_signal)

# Step 3: Detrend + bandpass (4th-order Butterworth, 0.67–3.0 Hz)
h = detrend(h, type="linear")
b, a = butter(4, [low, high], btype="band")
h = filtfilt(b, a, h)

# Step 4: Welch PSD (10-second window, 50% overlap)
f, pxx = welch(h, fs=actual_fs, nperseg=nperseg, noverlap=nperseg//2)

# Step 5: Peak detection with sub-harmonic correction
# Step 6: SNR-based confidence (SNR 1→conf 0.0, SNR 10→conf 1.0)
confidence = float(np.clip((snr - 1.0) / 9.0, 0.0, 1.0))

# Step 7: EMA stabilization
self._bpm_ema = alpha * bpm + (1 - alpha) * self._bpm_ema
```

### Key Parameters

| Parameter | Current Value | Notes |
|---|---|---|
| Buffer length | 15 seconds (450 frames at 30fps) | Rolling deque |
| Minimum buffer to start | 5 seconds (150 frames) | Before any BPM output |
| Welch window | 10 seconds | Gives ~6 BPM frequency resolution |
| Bandpass range | 0.67–3.0 Hz | Professor wants 0.7–3.5 Hz |
| Filter order | 4th-order Butterworth IIR | No FIR comparison |
| EMA alpha | 0.35 | Smoothing factor |
| BPM reject threshold | ±20 BPM from EMA | Outlier rejection |
| Calibration offset | +15 BPM | Hardcoded — set to 0 for research |

---

## 6. Gap Analysis — Current vs Professor's Requirements

| Feature | Status | Notes |
|---|---|---|
| YOLO face detection | ✅ Done | |
| BiSeNet + YCbCr skin segmentation | ✅ Done | |
| CHROM algorithm | ✅ Done | |
| Linear detrending | ✅ Done | |
| IIR Butterworth bandpass (0.67–3.0 Hz) | ✅ Done | Range needs update to 0.7–3.5 Hz |
| Welch PSD | ✅ Done | |
| Sub-harmonic correction | ✅ Done | |
| SNR confidence score | ⚠️ Partial | Missing temporal stability + skin count |
| EMA BPM stabilization | ✅ Done | |
| FIR filter design and comparison | ❌ Missing | Needs MATLAB implementation |
| STFT analysis | ❌ Missing | Not implemented |
| Sliding window comparison (different lengths) | ❌ Missing | Not implemented |
| MUSIC algorithm (K=1) | ❌ Missing | Core research contribution needed |
| ESPRIT algorithm | ❌ Missing | Alternative to MUSIC |
| Eigenvalue plot for K estimation | ❌ Missing | |
| Temporal stability in confidence | ❌ Missing | |
| Skin pixel count in confidence | ❌ Missing | Data available, not used |
| Minimum data benchmark (MUSIC vs Welch) | ❌ Missing | Key research question |
| DocBot robot sensor repositioning | ❌ Missing | Requires ROS2 integration |
| MATLAB live script implementation | ❌ Missing | Target: bpm_controls.mlx |

---

## 7. MATLAB Implementation Roadmap — bpm_controls.mlx

The plan is to implement everything in MATLAB first (with a static image, then a recorded video, then live stream) before porting back to Python/ROS2.

### Phase 1: Replicate Current Python Pipeline (Baseline)

Mirror exactly what `bpm_estimate.py` does, with full visualization at every step:

```
Section 1.1  Load video / image
Section 1.2  Simulate face crop (manual ROI for image, YOLO for video)
Section 1.3  YCbCr skin detection — visualize mask at each frame
Section 1.4  Spatial averaging — extract R̄(t), Ḡ(t), B̄(t) time series
Section 1.5  Per-channel detrend — plot before/after
Section 1.6  CHROM projection — plot Xₛ(t), Yₛ(t), S(t)
Section 1.7  Butterworth bandpass (current: 0.67–3.0 Hz)
Section 1.8  Welch PSD — plot full spectrum with cardiac band highlighted
Section 1.9  Peak detection — show dominant frequency
Section 1.10 Harmonic correction — show decision
Section 1.11 Final BPM output
```

### Phase 2: Filter Design Comparison

```
Section 2.1  Butterworth IIR — frequency response, phase response
Section 2.2  Chebyshev Type I IIR — comparison
Section 2.3  Elliptic IIR — comparison (sharpest transition)
Section 2.4  Parks-McClellan FIR — optimal equiripple
Section 2.5  Windowed sinc FIR — Kaiser window
Section 2.6  Side-by-side comparison: magnitude, phase, group delay
Section 2.7  Apply each filter to rPPG signal and compare results
```

### Phase 3: Frequency Analysis Comparison

```
Section 3.1  FFT at various window lengths (1s, 2s, 5s, 10s, 15s)
Section 3.2  Welch at various window lengths — compare BPM accuracy
Section 3.3  STFT — time-frequency heatmap showing BPM evolution over time
Section 3.4  Frequency resolution comparison table
```

### Phase 4: MUSIC Implementation — Core Research

```
Section 4.1  Simulate test signal: 1.2 Hz sinusoid + white noise
             (synthetic rPPG to validate MUSIC before real data)

Section 4.2  MUSIC with K=1, M=15, N=90 (3 seconds at 30fps)
             - Build covariance matrix R̂ᵧ
             - Eigendecompose: plot ALL eigenvalues (show 1 large + M-1 small)
             - Identify signal/noise subspace split
             - Build steering vectors
             - Sweep ω and compute P_MUSIC(ω)
             - Plot MUSIC pseudo-spectrum
             - Find peak → BPM

Section 4.3  Real rPPG signal → MUSIC
             - Apply to actual extracted S(t) signal

Section 4.4  Side-by-side: MUSIC spectrum vs Welch PSD

Section 4.5  Minimum data experiment:
             - Vary N: 30, 45, 60, 90, 120, 150, 300 samples
             - Plot MUSIC BPM accuracy vs N
             - Find minimum N for <5 BPM error
             - Compare to minimum N for Welch at same accuracy

Section 4.6  Noise robustness experiment:
             - Add increasing noise levels (SNR: 20dB, 10dB, 5dB, 0dB)
             - Compare MUSIC vs Welch BPM accuracy at each SNR
```

### Phase 5: Enhanced Confidence Score

```
Section 5.1  SNR component — current formula
Section 5.2  Temporal stability component:
             - Run analysis on overlapping 2-second windows
             - Compute std of BPM estimates across last 5 windows
             - Normalize to [0,1]
Section 5.3  Skin pixel count component:
             - Normalize skin pixel count to [0,1] based on face area
Section 5.4  Combined confidence score:
             C = w₁·SNR_score + w₂·Stability_score + w₃·Skin_score
             (tune weights w₁, w₂, w₃ experimentally)
Section 5.5  Plot confidence over time for a test video
Section 5.6  Threshold analysis — what confidence level gives reliable BPM?
```

### Phase 6: Image → Video → Live Stream Progression

```
Phase 6a:  Static image — manually define ROI, compute spatial mean,
           test all frequency methods on single-frame data (synthetic temporal)

Phase 6b:  Recorded video — full pipeline, compare MUSIC vs Welch,
           plot BPM over time from both methods

Phase 6c:  Live webcam stream — real-time MATLAB implementation

Phase 6d:  ROS2 integration — port MATLAB results back to Python,
           add robot repositioning trigger based on confidence threshold
```

---

## 8. Full Pipeline Summary Map

```
╔══════════════════════════════════════════════════════════════════╗
║              COMPLETE rPPG PIPELINE WITH MUSIC UPGRADE           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  CAMERA FEED                                                     ║
║       │                                                          ║
║       ▼                                                          ║
║  [1] FACE DETECTION (YOLO) ──────────────────────────────────   ║
║       │                                                          ║
║       ▼                                                          ║
║  [2] SKIN SEGMENTATION                                           ║
║       ├── BiSeNet semantic segmentation                          ║
║       ├── YCbCr color filter (77≤Cb≤127, 133≤Cr≤173, Y>40)     ║
║       └── Combined mask = BiSeNet AND YCbCr                      ║
║       │                                                          ║
║       ▼                                                          ║
║  [3] RGB SIGNAL EXTRACTION                                       ║
║       └── Spatial mean: R̄(t), Ḡ(t), B̄(t)                      ║
║       │                                                          ║
║       ▼                                                          ║
║  [4] CHROM ALGORITHM                                             ║
║       ├── Temporal normalization (DC removal)                    ║
║       ├── Xₛ(t) = 3R̂(t) - 2Ĝ(t)                               ║
║       ├── Yₛ(t) = 1.5R̂(t) + Ĝ(t) - 1.5B̂(t)                   ║
║       └── S(t) = Xₛ(t) - α·Yₛ(t)                               ║
║       │                                                          ║
║       ▼                                                          ║
║  [5] DETRENDING                                                  ║
║       └── Remove linear drift                                    ║
║       │                                                          ║
║       ▼                                                          ║
║  [6] BANDPASS FILTER (0.7 – 3.5 Hz)  ← PROF: compare FIR/IIR   ║
║       │                                                          ║
║       ▼                                                          ║
║  [7] FREQUENCY ANALYSIS (SLIDING WINDOW)                         ║
║       ├── Welch PSD (current method)                             ║
║       ├── FFT (comparison)                                       ║
║       ├── STFT (time-frequency view)              ← PROF: NEW   ║
║       └── MUSIC K=1 (rapid peak detection)        ← PROF: KEY   ║
║               ├── Build covariance matrix R̂ᵧ                    ║
║               ├── Eigendecompose → Qₛ, Qₙ                       ║
║               ├── Sweep ω, compute P_MUSIC(ω)                   ║
║               └── Find peak → cardiac frequency                  ║
║       │                                                          ║
║       ▼                                                          ║
║  [8] HARMONIC CORRECTION                                         ║
║       └── Check if 2f has >30% power of f                       ║
║       │                                                          ║
║       ▼                                                          ║
║  [9] BPM OUTPUT                                                  ║
║       └── BPM = fpeak × 60                                       ║
║       │                                                          ║
║       ▼                                                          ║
║  [10] CONFIDENCE SCORE   ← PROF: ALL THREE components           ║
║       ├── SNR of cardiac peak             (current: ✅)          ║
║       ├── Temporal stability across windows (current: ❌)        ║
║       └── Skin pixel count                (current: ❌)          ║
║       │                                                          ║
║       ▼                                                          ║
║  [11] ROBOT CONTROL (DocBot)   ← PROF: Closed-loop integration  ║
║       ├── confidence ≥ threshold → report BPM                   ║
║       └── confidence < threshold → reposition camera (ROS2)     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Key Equations Reference

### MUSIC Pseudo-Spectrum
```
P_MUSIC(ω) = 1 / (a(ω)* · Qₙ · Qₙ* · a(ω))
```

### Steering Vector
```
a(ω) = [1, e^(-jω), e^(-j2ω), ..., e^(-j(M-1)ω)]ᵀ
```

### Sample Covariance Matrix
```
R̂ᵧ = 1/(N-M+1) · Σₙ y[n]·y[n]*
```

### CHROM Signal
```
S(t) = Xₛ(t) - α·Yₛ(t),   α = σ(Xₛ)/σ(Yₛ)
```

### Confidence Score (Target)
```
C(t) = w₁·SNR_norm + w₂·Stability_norm + w₃·SkinCount_norm
```

---

## Next Steps

1. **Implement `bpm_controls.mlx`** — start with Phase 1 (replicate Python pipeline in MATLAB with full visualization)
2. **Validate on a static image** — test each step independently
3. **Add MUSIC** (Phase 4) — implement from scratch in MATLAB, validate on synthetic sinusoid first
4. **Benchmark** — compare MUSIC vs Welch accuracy at different window lengths
5. **Extended confidence score** (Phase 5) — all three components
6. **Port to Python + ROS2** — integrate into DocBot live system

---

*Document compiled from:*
- *Prof. Xu Chen, "Implementation of rPPG Signal Processing and Frequency Analysis for DocBot Heart Rate Extraction", 2026-04-01*
- *Prof. Xu Chen, "Multiple Narrow-Band Signal Identification with MUSIC", 2026-04-05*
- *DocBot Technical Documentation, "Mathematical Foundations of Remote Photoplethysmography", 2026-03-10*
- *`bpm_estimate.py` — Current rPPG implementation (CHROM + Welch)*

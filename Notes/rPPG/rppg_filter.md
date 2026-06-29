# rppg_filter — Adaptive Filter Design & Selection

> **File:** `Updated Pipeline/scripts/analysis/bpm_filterdesign.m`
> **Role in pipeline:** Stage 2 — consumes `rppg_output.csv` from the acquisition stage, designs and evaluates a full suite of IIR and FIR bandpass filters against the extracted BVP signal, ranks them by a composite quality metric, and exports the best-performing filtered signals to `results/filter_results/` for Stage 3 (FDA).
> **Input:** `results/input_results/<timestamp>/rppg_output.csv`
> **Output:** `results/filter_results/filterdesign_<timestamp>.csv` + console summary table + 10 MATLAB figures

---

## Table of Contents

1. [What This File Actually Does](#1-what-this-file-actually-does)
2. [High-Level Architecture & Design Philosophy](#2-high-level-architecture--design-philosophy)
3. [Source Selection — Two Signal Paths](#3-source-selection--two-signal-paths)
   - 3.1 [Path A — Python CSV (primary)](#31-path-a--python-csv-primary)
   - 3.2 [Path B — MATLAB Raw Extraction (fallback)](#32-path-b--matlab-raw-extraction-fallback)
4. [Ground-Truth Loading](#4-ground-truth-loading)
5. [Adaptive Passband Estimation](#5-adaptive-passband-estimation)
   - 5.1 [Step 1 — Broad Welch PSD](#51-step-1--broad-welch-psd)
   - 5.2 [Step 2 — Rough Cardiac Estimate](#52-step-2--rough-cardiac-estimate)
   - 5.3 [Step 3 — Half-Harmonic Artifact Detection](#53-step-3--half-harmonic-artifact-detection)
   - 5.4 [Step 4 — Signal-to-Artifact Ratio](#54-step-4--signal-to-artifact-ratio)
   - 5.5 [Step 5 — Upper Passband & Stopband Edges](#55-step-5--upper-passband--stopband-edges)
   - 5.6 [Step 6 — Adaptive Sliding Window Length](#56-step-6--adaptive-sliding-window-length)
6. [IIR Filter Design — SOS Form](#6-iir-filter-design--sos-form)
7. [FIR Filter Design](#7-fir-filter-design)
   - 7.1 [Kaiser Order Estimate](#71-kaiser-order-estimate)
   - 7.2 [Three Hamming-Windowed Sinc Filters](#72-three-hamming-windowed-sinc-filters)
   - 7.3 [Kaiser-Window FIR](#73-kaiser-window-fir)
   - 7.4 [Parks-McClellan Equiripple FIR](#74-parks-mcclellan-equiripple-fir)
8. [Figures — What They Show and Why](#8-figures--what-they-show-and-why)
9. [Sliding-Window BPM Evaluation](#9-sliding-window-bpm-evaluation)
   - 9.1 [Harmonic Correction](#91-harmonic-correction)
   - 9.2 [Per-Window SNR](#92-per-window-snr)
10. [Quality Metrics & Composite Score](#10-quality-metrics--composite-score)
    - 10.1 [Quality-Weighted MAE](#101-quality-weighted-mae)
    - 10.2 [Lin's Concordance Correlation Coefficient (CCC)](#102-lins-concordance-correlation-coefficient-ccc)
    - 10.3 [Composite Score Formula](#103-composite-score-formula)
11. [Tight Lower-Passband Comparison](#11-tight-lower-passband-comparison)
12. [Export to Filter Results](#12-export-to-filter-results)
13. [Helper Functions](#13-helper-functions)
    - 13.1 [`sosfreqz()`](#131-sosfreqz)
    - 13.2 [`concordance_cc()`](#132-concordance_cc)
14. [Complete Signal Flow Diagram](#14-complete-signal-flow-diagram)
15. [Why These Parameter Values?](#15-why-these-parameter-values)
16. [Appendix — References](#16-appendix--references)

---

## 1. What This File Actually Does

`bpm_filterdesign.m` answers a fundamental question: **which bandpass filter design gives the most accurate BPM estimates for this specific rPPG signal?**

The file operates entirely downstream of the acquisition stage — it never reads the video. It receives the already-extracted, CHROM-projected, detrended BVP signal (`BVP_detrended` from `rppg_output.csv`) and subjects it to nine different filter designs: four IIR designs (Butterworth, Chebyshev I, Chebyshev II, Elliptic) and five FIR designs (Hamming at 3 orders + Kaiser + Parks-McClellan). Each filter is evaluated in a sliding-window BPM estimation loop, compared against the contact-sensor ground truth, and scored on four axes (accuracy, stability, SNR, and agreement).

Crucially, the script does **not** use a fixed bandpass specification. It first analyses the actual spectrum of the input BVP signal to detect where motion/respiration artifacts are, then derives the passband edges adaptively. This is important because the artifact location varies between recordings.

At the end, the best-performing filtered signals are exported for the next stage (FDA), which compares Welch, MUSIC, and ESPRIT frequency estimators applied to those same signals.

---

## 2. High-Level Architecture & Design Philosophy

**Decision 1 — Adaptive specification, not hardcoded constants.**
The passband `[f_p1, f_p2]` is derived from the signal's own spectrum in every run. A fixed `[0.7, 3.5]` Hz spec is used as a fall-through only. This is architecturally significant: a filter designed for a specific artifact location will outperform a generic filter on that recording, and the adaptive approach makes this automatic.

**Decision 2 — SOS (Second-Order Sections) for IIR, never direct-form.**
High-order bandpass IIR filters in direct form (numerator/denominator polynomial) suffer from catastrophic numerical rounding errors at high orders. SOS representation decomposes the filter into a cascade of 2nd-order sections, each of which is numerically stable. The `butter()`, `cheby1()` etc. functions are called with the `[sos, g]` output form, and `filtfilt(sos, g, x)` is used — not `filtfilt(b, a, x)`.

**Decision 3 — Order selected by the filter's own `*ord` function, not by hand.**
`buttord()`, `cheb1ord()`, `cheb2ord()`, `ellipord()` compute the minimum order that satisfies the specified passband ripple (Rp) and stopband attenuation (Rs) requirements. The order is then **capped at 6** for Butterworth (because `buttord` can return very high orders when the transition band is tight, leading to unstable high-order designs). For Chebyshev and Elliptic, the lower equiripple order is inherently self-limiting.

**Decision 4 — `filtfilt` for zero-phase filtering.**
All filters are applied with `filtfilt` (forward-backward pass). This doubles the effective order but eliminates all phase distortion — critical for rPPG because phase delay would shift the BVP waveform in time, corrupting any time-domain analysis. For the sliding-window BPM evaluation via Welch PSD, phase matters less — but for waveform-shape analysis (future work) it is essential.

---

## 3. Source Selection — Two Signal Paths

```matlab
USE_PYTHON_CSV = true;
```

### 3.1 Path A — Python CSV (primary)

```matlab
py_data = readtable(PYTHON_CSV);
t_axis  = py_data.time_s;
fs      = 1 / median(diff(t_axis));
S_det   = py_data.BVP_detrended;
```

When `USE_PYTHON_CSV = true`, the CHROM-projected, detrended BVP signal is read directly from the Python pipeline's output. The sampling rate `fs` is inferred from the median frame-to-frame time difference rather than read from a metadata field — this handles minor VFR (variable frame rate) jitter in the timestamps without assuming exact integer fps.

**Why `median(diff(t_axis))` instead of `mean(diff(t_axis))`?** Median is robust to dropped frames (where one `diff` value is ~2/fps) and duplicate timestamps (where one `diff` value is ~0). Mean would be biased by these outliers.

Using the Python pipeline's BVP signal is preferred because it incorporates BiSeNet-based skin segmentation, adaptive colour skin detection, GMM clustering, hair rejection, and probability-weighted averaging — far superior to the simple MATLAB fallback.

### 3.2 Path B — MATLAB Raw Extraction (fallback)

When `USE_PYTHON_CSV = false`, the script reads the raw video directly and runs a simplified Viola-Jones + fixed-YCbCr skin pipeline:

```matlab
detector = vision.CascadeObjectDetector('MinSize', [80 80]);
...
frame = rot90(readFrame(vid), 3);
```

- `rot90(..., 3)` applies a 270° counter-clockwise (= 90° clockwise) rotation to compensate for the portrait-mode recording orientation of a phone camera stored in landscape container.
- `MinSize = [80 80]` is the minimum face bounding-box size in pixels — prevents detecting small false-positive faces in the background.
- The YCbCr conversion uses the exact ITU-R BT.601 matrix coefficients: `Y = 0.299R + 0.587G + 0.114B`, `Cb = -0.168736R - 0.331264G + 0.5B + 128`, `Cr = 0.5R - 0.418688G - 0.081312B + 128`.
- Fixed thresholds `Cb∈[77,127], Cr∈[133,173], Y>40` — no adaptivity.
- Luminance normalisation: `fcd = fcd/lum*128` — same approach as the Python pipeline.
- CHROM and linear detrend are computed inline using the same equations as the Python pipeline.

This path exists as a **debugging fallback** — if the Python CSV is not available or suspect, you can check whether the MATLAB extraction gives a consistent BVP signal.

---

## 4. Ground-Truth Loading

```matlab
csv_data    = readtable(CSV_PATH);
gt_time     = csv_data.offset_seconds;
gt_hr       = csv_data.heart_rate;
valid       = ~isnan(gt_hr);
gt_time     = gt_time(valid);  gt_hr = gt_hr(valid);
bpm_gt_mean  = mean(gt_hr);
bpm_gt_median= median(gt_hr);
```

The DocBOT vitals CSV is loaded regardless of which signal path is used. NaN rows are stripped immediately. Both mean and median GT BPM are computed:
- **Mean** is used as the reference BPM for relative error percentages (`pct_err = mae/bpm_gt_mean * 100`).
- **Median** is displayed on PSD plots as a more robust reference since GT HR distributions during a resting recording are often slightly skewed by one or two anomalous sensor readings.

---

## 5. Adaptive Passband Estimation

This is the most architecturally significant block in the file. The passband is derived from the signal's own spectrum in six steps.

### 5.1 Step 1 — Broad Welch PSD

```matlab
N_rough            = min(round(fs * 30), T);
[P_rough, F_rough] = pwelch(S_det, hann(N_rough), floor(N_rough/2), 8192, fs);
```

A Welch PSD is computed with a window of up to 30 seconds (or the full signal if shorter). **Why 30 s?** Long windows give high frequency resolution (`df = fs / N_rough`), allowing accurate detection of the cardiac peak and any artifact peaks that may be close to it. The 8192-point FFT provides sub-0.005 Hz resolution at typical fps (~30fps → df ≈ 30/8192 ≈ 0.0037 Hz ≈ 0.22 BPM), far better than needed to distinguish cardiac (~1.2 Hz) from half-harmonic (~0.6 Hz).

The Hann window is used (rather than rectangular) to control spectral leakage — a rectangular window has -13 dB sidelobes that can obscure a weak cardiac peak near a strong artifact. Hann has -31 dB sidelobes.

### 5.2 Step 2 — Rough Cardiac Estimate

```matlab
broad_mask  = (F_rough >= 0.67) & (F_rough <= 3.5);
[~, hf_i]  = max(P_rough(broad_mask));
f_card_raw  = F_hf(hf_i);
bpm_rough   = f_card_raw * 60;
if bpm_rough < 60 && (f_card_raw * 2) <= 3.5
    bpm_rough = bpm_rough * 2;
end
```

The dominant peak in the physiological heart rate range (0.67–3.5 Hz, 40–210 BPM) is found. An initial harmonic correction is applied: if the detected peak is below 60 BPM and doubling it stays within the search range, the doubled frequency is taken as the true cardiac estimate. This corrects the common half-harmonic artifact (see Section 5.3) before the artifact itself is located.

**Why 0.67 Hz as the lower bound here?** 0.67 Hz = 40 BPM — the minimum possible human heart rate (well-trained athletes at rest). Using a wider window (e.g. 0.3 Hz) would capture respiratory peaks (0.1–0.5 Hz) as false cardiac detections.

### 5.3 Step 3 — Half-Harmonic Artifact Detection

```matlab
f_half  = (bpm_rough / 2) / 60;
hh_lo   = max(0.40, f_half - 0.30);
hh_hi   = min(1.50, f_half + 0.30);
hh_mask = (F_rough >= hh_lo) & (F_rough <= hh_hi);
...
if pv_hh >= 0.20 * pv_card
    f_artifact = F_hh(hh_i);
    artifact_found = true;
end
```

The **half-harmonic artifact** is the most dangerous interference for rPPG. It arises when the cardiac fundamental is detected at double the true rate — but the more common problem is the reverse: a strong low-frequency artifact (respiration at ~0.3 Hz, or slow body sway at ~0.5 Hz) sits at exactly half the cardiac frequency and gets mistaken for the cardiac frequency. This makes the estimated BPM half the true value — a systematic 2× error.

The detection strategy:
1. **Targeted search (Strategy A):** Look within ±0.30 Hz of `bpm_rough/2` (i.e., around the expected half-frequency location). Accept as artifact if its power ≥ 20% of the cardiac peak power. The 20% threshold is deliberately loose — a peak at 20% of cardiac power can still dominate when passed through a filter's passband if it coincides with a filter ripple.
2. **Fallback (Strategy B):** If no half-harmonic is found, take the highest peak in the general motion zone (0.5–1.5 Hz). This is always defined — there is always some lowest-frequency content.

**Signal-to-artifact ratio (S/A):**
```matlab
sa_ratio_dB = 10 * log10(pv_card / (pv_art + 1e-30));
```
This is computed in dB. If the artifact is **below** `f_p1` (the lower passband edge), it will be filtered out — even a strong artifact below the passband is not dangerous. The script reports this context explicitly. If the artifact is **inside** the passband (`f_artifact >= f_p1`), all filters will struggle to remove it, and the console prints a `WARNING`.

### 5.4 Step 4 — Signal-to-Artifact Ratio

The S/A ratio is primarily informational — it tells you whether the adaptive passband is working. A recording with S/A < 0 dB (artifact stronger than cardiac) and the artifact inside the passband indicates that no filter will help; the rPPG signal quality from Stage 1 is insufficient, and you should return to the acquisition stage and investigate why (poor lighting, motion, skin detection failures).

### 5.5 Step 5 — Upper Passband & Stopband Edges

```matlab
f_p1 = max(0.67, f_artifact + 0.25);
f_s1 = max(0.30, f_p1 - 0.35);
f_p2 = min(fs/2 - 0.5, max(bpm_rough/60 + 1.50, 3.0));
f_s2 = min(fs/2 - 0.1, f_p2 + 0.80);
```

**Lower passband edge `f_p1`:** Set 0.25 Hz above the detected artifact, with a floor at 0.67 Hz. The 0.25 Hz margin ensures the artifact is in the transition band (attenuated but not at 0 dB) rather than inside the passband. The `max(0.67, ...)` floor ensures we never exclude the cardiac band even if the artifact is very close to 0.67 Hz.

**Lower stopband edge `f_s1`:** 0.35 Hz below `f_p1`, with a floor at 0.30 Hz. The `f_p1 - f_s1 = 0.35 Hz` transition bandwidth. This is the constraint that controls filter order: a narrower transition band requires a higher-order filter. 0.35 Hz is wide enough that IIR orders stay manageable (≤6) for most recordings.

**Upper passband edge `f_p2`:** At least 1.5 Hz above the cardiac estimate (to capture the harmonic) and at least 3.0 Hz (physiological maximum), capped at 0.5 Hz below Nyquist to avoid filter instability near fs/2.

**Upper stopband edge `f_s2`:** 0.80 Hz above `f_p2`.

### 5.6 Step 6 — Adaptive Sliding Window Length

```matlab
sw_len_s = 10;
if T/fs < 30;  sw_len_s = 5;  end
```

The BPM evaluation sliding window is 10 seconds by default, reduced to 5 seconds for short recordings (< 30 seconds total). This ensures at least 3 non-overlapping windows even for very short recordings. A 10-second Welch window at 30fps gives `10 × 30 = 300` samples and `df = 30/300 = 0.1 Hz = 6 BPM` frequency resolution — acceptable for BPM estimation.

---

## 6. IIR Filter Design — SOS Form

```matlab
Rp = 1.0;   % dB  max passband ripple
Rs = 40;    % dB  min stopband attenuation

[N_bw, Wn_bw]  = buttord(Wp, Ws, Rp, Rs);
N_bw = min(N_bw, 6);
[sos_bw, g_bw] = butter(N_bw, Wn_bw, 'bandpass');

[N_c1, Wn_c1]  = cheb1ord(Wp, Ws, Rp, Rs);
[sos_c1, g_c1] = cheby1(N_c1, Rp, Wn_c1, 'bandpass');

[N_c2, Wn_c2]  = cheb2ord(Wp, Ws, Rp, Rs);
[sos_c2, g_c2] = cheby2(N_c2, Rs, Wn_c2, 'bandpass');

[N_el, Wn_el]  = ellipord(Wp, Ws, Rp, Rs);
[sos_el, g_el] = ellip(N_el, Rp, Rs, Wn_el, 'bandpass');
```

**Normalised frequencies:** `Wp = [f_p1 f_p2]/(fs/2)` and `Ws = [f_s1 f_s2]/(fs/2)`. Dividing by `fs/2` (Nyquist) converts Hz to normalised digital frequency in [0, 1] as required by MATLAB's filter design functions.

**The four IIR families:**

| Filter | Key property | `*ord` result | Note |
|---|---|---|---|
| Butterworth | Maximally flat passband (no ripple) | Highest order for given Rp/Rs | Order capped at 6 |
| Chebyshev I | Equiripple in passband (≤Rp), monotone stopband | Lower order than Butterworth | Rp=1dB is modest ripple |
| Chebyshev II | Monotone passband, equiripple stopband (≥Rs) | Same order as Cheby I | Better stopband than Butterworth |
| Elliptic | Equiripple in both bands | Minimum order | Highest selectivity for given order |

**Why SOS form?** The second argument form `[sos, g] = butter(...)` returns a matrix where each row is `[b0 b1 b2 1 a1 a2]` — the coefficients of one 2nd-order section. Filtering with `filtfilt(sos, g, x)` processes the signal through each section in sequence, keeping numerical precision. At order 6, a bandpass filter has `6` poles per band edge → `12` poles total → 6 second-order sections. Direct-form polynomial coefficients for a 12th-order filter would have values spanning many orders of magnitude, causing catastrophic rounding in `double` precision.

**Butterworth order cap:** The `buttord` formula is conservative — for tight transition bands it returns high orders. Order 6 bandpass = 12 poles, which is already at the edge of numerical stability for `filtfilt`. The cap at 6 trades a small amount of stopband attenuation (below the 40dB spec) for numerical safety.

**References:** See [Appendix A.1](#a1-iir-filter-design)

---

## 7. FIR Filter Design

FIR (Finite Impulse Response) filters are linear-phase by construction when symmetric — every frequency component is delayed by exactly N/2 samples, with no frequency-dependent phase distortion. This is fundamentally impossible for IIR filters. The trade-off is that FIR filters require much higher orders (more coefficients) to achieve the same stopband attenuation.

### 7.1 Kaiser Order Estimate

```matlab
delta_f     = min(f_p1-f_s1, f_s2-f_p2);
delta_omega = 2*pi*delta_f/fs;
N_est       = ceil((Rs-7.95)/(2.285*delta_omega));
```

This is the **Kaiser approximation formula** for minimum FIR filter order:

`N ≈ (Rs - 7.95) / (2.285 × Δω)`

where `Δω = 2π × Δf / fs` is the normalised transition bandwidth. The formula is derived from the Kaiser window's shape parameter relation to attenuation. `Rs = 40 dB` and `Δf = min(f_p1-f_s1, f_s2-f_p2)` — the smaller of the two transition bands (the bottleneck).

**Odd order enforcement:** `if mod(N_est,2)==0; N_est=N_est+1; end`. An FIR filter of odd order has an even number of coefficients → Type I linear phase (symmetric, no half-sample delay). Type II (even order) has a zero at ω=π (Nyquist), which is undesirable for a bandpass filter. The `+1` ensures odd order.

### 7.2 Three Hamming-Windowed Sinc Filters

```matlab
N_lo  = 2*round(0.5*fs/2)+1;   % ~0.5 s latency
N_mid = 2*round(2.0*fs/2)+1;   % ~2.0 s latency
N_hi  = min(N_est, round(sw_len_s*fs*0.4));
b_ham_lo  = fir1(N_lo -1, Wp, 'bandpass', hamming(N_lo));
b_ham_mid = fir1(N_mid-1, Wp, 'bandpass', hamming(N_mid));
b_ham_hi  = fir1(N_hi -1, Wp, 'bandpass', hamming(N_hi));
```

`fir1` designs a windowed-sinc FIR filter. The ideal bandpass impulse response is an infinite sinc — the window truncates it to N samples. The Hamming window (`0.54 - 0.46cos(2πn/N)`) provides -42 dB peak sidelobe attenuation, sufficient for the 40 dB Rs spec with some margin.

**Three orders serve different purposes:**
- `N_lo` (~0.5s latency): Fast response, wide transition band. Acceptable for real-time streaming where latency budget is tight.
- `N_mid` (~2.0s latency): Balanced design — the one ultimately exported as `BVP_ham_tight` and `BVP_ham_adapt`.
- `N_hi` (meets spec, capped at 40% of window length): Shows maximum achievable sharpness. Capped because a filter longer than 40% of the analysis window would spend most of each window in its own startup transient.

**Order formulas using `round(..., 2)` pattern:** `2*round(0.5*fs/2)+1` computes `round(0.5*fs/2)` → nearest integer to 0.25×fs samples → doubles it → adds 1 to guarantee odd. The double-and-add-1 construction ensures the group delay `N/2` is always an integer number of samples (no half-sample delays).

### 7.3 Kaiser-Window FIR

```matlab
dev_stop = 10^(-Rs/20);
dev_pass = (10^(Rp/10)-1)/(10^(Rp/10)+1);
[N_ksr, Wn_ksr, beta_ksr, ftype_ksr] = kaiserord(...
    [f_s1 f_p1 f_p2 f_s2], [0 1 0], [dev_stop dev_pass dev_stop], fs);
b_ksr = fir1(N_ksr, Wn_ksr, ftype_ksr, kaiser(N_ksr+1, beta_ksr));
```

`kaiserord` is MATLAB's implementation of the Kaiser window order estimation, taking the full specification in terms of ripple deviations rather than dB:
- `dev_stop = 10^(-40/20)` = 0.01 (maximum stopband amplitude error — 1% of passband)
- `dev_pass = (10^(1/10)-1)/(10^(1/10)+1)` ≈ 0.0559 (maximum passband deviation from unity)

`kaiserord` returns the filter order `N_ksr`, the cutoff frequencies `Wn_ksr`, the Kaiser β parameter `beta_ksr` (controls the trade-off between main-lobe width and sidelobe level), and the filter type string `ftype_ksr`. The β parameter is computed from the desired attenuation using Kaiser's empirical formula.

The key advantage of Kaiser over plain Hamming: β is **tuned to exactly meet the spec** rather than using a fixed window shape. Hamming's sidelobe behaviour is fixed at -42 dB regardless of what you asked for.

### 7.4 Parks-McClellan Equiripple FIR

```matlab
N_pm       = N_ksr;
w_ratio    = 10^((Rs-Rp)/20);
bands_pm   = [0 f_s1 f_p1 f_p2 f_s2 fs/2]/(fs/2);
amps_pm    = [0 0    1    1    0    0   ];
weights_pm = [w_ratio 1 w_ratio];
b_pm = firpm(N_pm-1, bands_pm, amps_pm, weights_pm);
```

Parks-McClellan (PM) uses the Remez exchange algorithm to design the **optimal equiripple FIR filter** — for a given order N, no other linear-phase FIR has lower maximum error across both pass and stop bands simultaneously. The Chebyshev equiripple property means all sidelobes are equally high, and all passband ripples are equally small.

**Weight vector `[w_ratio 1 w_ratio]`:** The error in each band is weighted by this vector during optimisation. `w_ratio = 10^((40-1)/20)` ≈ 89 — the stopband is penalised 89× more heavily than the passband. This trades some extra passband ripple for better stopband attenuation, matching the Rp=1dB / Rs=40dB asymmetric requirement.

**Why the same order as Kaiser (`N_pm = N_ksr`)?** PM is being compared to Kaiser at equal order. Since PM is optimal at any given order, this is a fair comparison — both need the same number of coefficients; PM will have better overall error distribution.

**References:** See [Appendix A.2](#a2-fir-filter-design)

---

## 8. Figures — What They Show and Why

| Figure | Name | Key Information |
|---|---|---|
| Fig 0 | Adaptive Spec Diagnostic | Raw BVP PSD with detected artifact, GT BPM, and derived passband edges overlaid. **Check this first on a new recording.** |
| Fig 1 | IIR Magnitude | Magnitude responses of all 4 IIR filters overlaid with passband/stopband spec lines. Shows which filter most cleanly meets the spec. |
| Fig 2 | FIR Magnitude | Same for all 5 FIR designs. Higher-order Hamming and PM/Kaiser clearly show sharper transitions. |
| Fig 3 | Phase | Top: IIR nonlinear phase (each filter has a different frequency-dependent phase curve). Bottom: FIR perfectly linear phase (straight lines). |
| Fig 4 | Group Delay | Top: IIR group delay varies within the passband (up to seconds of variation). Bottom: FIR constant group delay = N/2 samples across all frequencies. |
| Fig 5 | Filtered BVP | Time-domain waveforms of S_det before and after each filter. Visual check for ringing, edge effects, and waveform fidelity. |
| Fig 6 | PSD vs GT | Welch PSD of each filter's output in the cardiac band, with GT BPM marked. The dominant peak should align with the vertical GT line. |
| Fig 7 | IIR Sliding BPM | Per-window BPM estimates for all 4 IIR filters vs GT over time, with ±1σ shading. |
| Fig 8 | FIR Sliding BPM | Same for all 5 FIR designs. |
| Fig 9 | Per-Window Error | 9-panel error plot (one per filter) with ±5 BPM dotted reference and ±1σ shading. Best filter marked with ◄. |
| Fig 10 | Composite Score Bar | Single bar chart ranking all 9 filters by composite score. Best IIR (orange), best FIR (green), overall best (dark green). |

**On Fig 3 and Fig 4 (Phase and Group Delay):** These are not just cosmetic — they demonstrate a fundamental IIR vs FIR trade-off. An IIR filter with group delay that varies by 500 ms across the passband means the 1.0 Hz cardiac component and the 2.0 Hz harmonic arrive at different times after filtering. `filtfilt`'s forward-backward pass cancels the phase exactly — so for offline processing, this matters less. But it motivates why **FIR filters are preferred for real-time rPPG** where `filtfilt` cannot be used (it requires the full signal).

---

## 9. Sliding-Window BPM Evaluation

```matlab
sw_len    = round(sw_len_s * fs);
sw_step   = round(1*fs);
sw_starts = 1:sw_step:T-sw_len+1;
```

Windows step by 1 second with a 10-second (or 5-second) analysis window. For each window:

```matlab
seg = all_sigs{fi}(idx_s:idx_e);
np  = length(seg);
[pw_k, fw_k] = pwelch(seg, hann(np), floor(np/2), nfft_sw, fs);
bk = (fw_k >= f_p1) & (fw_k <= f_p2);
[pk_val, pi_] = max(pw_band);
bpm_est = fw_band(pi_) * 60;
```

The Welch PSD is computed on the window using a single full-window Hann segment (no sub-windows within). This is equivalent to a zero-padded FFT with a Hann window — it gives smooth spectral interpolation around the cardiac peak, reducing grid-quantisation errors. The `nfft_sw = 4096` zero-padding further smooths the spectral peak.

### 9.1 Harmonic Correction

```matlab
if bpm_est < 60
    f_double = (bpm_est * 2) / 60;
    if f_double <= f_p2
        [~, pi2] = min(abs(fw_k - f_double));
        if pw_k(pi2) >= 0.30 * pk_val
            bpm_est = bpm_est * 2;
        end
    end
end
```

If the peak BPM estimate is below 60 BPM (i.e., the detected peak is likely a sub-harmonic artifact), the code checks whether doubling the frequency produces a peak with ≥ 30% of the sub-harmonic's power. If so, the doubled frequency is taken as the true BPM estimate.

**Why 30% threshold?** If the cardiac signal were at 2× the artifact, the 2:1 power ratio of the cardiac:artifact depends heavily on signal quality. In clean signals, the cardiac peak dominates. In noisy/low-quality windows, the cardiac may be only 5–10× weaker than the artifact — so even 5% would suffice. 30% is a moderate threshold that avoids doubling good sub-1-Hz estimates (genuine low-HR cases) while correcting the common artifact.

**The 60 BPM gate:** A normal resting adult never has HR < 40 BPM (0.67 Hz). An estimate below 60 BPM is suspicious — it may be the half-harmonic. The harmonic correction only fires for estimates < 60 BPM, preventing it from doubling a legitimate low-HR measurement.

### 9.2 Per-Window SNR

```matlab
[~, pi_c] = min(abs(fw_k - bpm_est/60));
sw_snr(fi, k) = 10 * log10(pw_k(pi_c) / (mean(pw_band) + 1e-30));
```

SNR is computed as the cardiac peak power (at the harmonic-corrected BPM) divided by the mean in-band power. This measures how cleanly the cardiac peak stands above the noise floor — not just whether the peak is large in absolute terms. A window where all frequencies have high power (noisy signal) will have low SNR even if the cardiac peak is detectable. The SNR is evaluated **at the corrected BPM**, not the raw dominant peak — this prevents a spurious sharp artifact from claiming a high SNR.

---

## 10. Quality Metrics & Composite Score

### 10.1 Quality-Weighted MAE

```matlab
w = (sw_quality(:)' / (sum(sw_quality) + 1e-9)) * n_sw;
mae_w_all = (abs(sw_err) * w') / sum(w);
```

Each window is weighted by the mean `quality_score` from the Python pipeline's CSV for that window's frames. High-quality frames (good skin detection, high BiSeNet confidence) contribute more to the MAE. The weights are normalised so they sum to `n_sw` — this preserves the absolute BPM scale (an unweighted MAE of 5 BPM means "on average 5 BPM off", and a weighted MAE of the same magnitude means the same thing after accounting for frame quality). Without normalisation, a recording with uniformly low quality scores would produce an artificially low weighted MAE.

**Why weight by frame quality?** Without quality weighting, a window with good skin detection but unlucky framing (subject blinks, turns) has the same influence as a window with poor skin detection throughout. The quality score already encodes the certainty of the skin signal — downweighting low-quality windows gives a more honest estimate of the filter's true BPM accuracy.

### 10.2 Lin's Concordance Correlation Coefficient (CCC)

```matlab
function ccc = concordance_cc(x, y)
    mu_x = mean(x);  mu_y = mean(y);
    s2_x = mean((x - mu_x).^2);
    s2_y = mean((y - mu_y).^2);
    s_xy = mean((x - mu_x) .* (y - mu_y));
    ccc  = 2 * s_xy / (s2_x + s2_y + (mu_x - mu_y)^2 + 1e-9);
end
```

CCC measures **both precision (correlation) and accuracy (bias)** simultaneously. A Pearson r of 1.0 is achievable even when all estimates are systematically 20 BPM off — r measures correlation shape, not bias. CCC = 1.0 only when estimates perfectly match GT values.

**Why CCC instead of Pearson r for this data?** In a resting-subject recording, the true heart rate barely varies (±2–3 BPM around a mean of ~72 BPM). Pearson r on a nearly-constant GT signal is numerically unstable — any noisy estimator will show low r even if its absolute errors are tiny, because there is nothing to correlate against. CCC handles this correctly: the `(mu_x - mu_y)²` term in the denominator penalises mean bias even when there is no temporal variation to correlate against.

**References:** See [Appendix A.3](#a3-lins-ccc)

### 10.3 Composite Score Formula

```matlab
norm01  = @(x) (x - min(x)) ./ (max(x) - min(x) + 1e-9);
mae_s   = 1 - norm01(mae_w_all);   % lower MAE → higher score
cons_s  = 1 - norm01(consistency); % lower std → higher score
snr_s   =     norm01(snr_all);     % higher SNR → higher score
ccc_s   =     norm01(ccc_all);     % higher CCC → higher score
composite = 0.40*mae_s + 0.25*cons_s + 0.20*snr_s + 0.15*ccc_s;
```

Each metric is min-max normalised across all 9 filters to [0, 1], then weighted. The normalisation is **relative**: the best filter on each metric gets 1.0, the worst gets 0.0. This means composite score is a ranking tool, not an absolute quality measure — it tells you which filter is best for this recording, not whether any filter is good enough.

**Why these weights?**
- `0.40 × wMAE`: Accuracy (BPM closeness to GT) is the primary objective.
- `0.25 × consistency`: A filter that varies wildly between windows is useless for real deployment even if its mean is accurate.
- `0.20 × SNR`: High cardiac-band SNR means the BPM estimate is robust — a sharp peak is easier to locate than a broad hump.
- `0.15 × CCC`: CCC provides an independent check on accuracy+bias that handles the stable-GT case Pearson r cannot.

---

## 11. Tight Lower-Passband Comparison

```matlab
f_p1_t = 1.0;   f_s1_t = 0.65;
[N_el_t, Wn_el_t] = ellipord(Wp_t, Ws_t, Rp, Rs);
[sos_el_t, g_el_t]= ellip(N_el_t, Rp, Rs, Wn_el_t, 'bandpass');
b_ham_t = fir1(N_mid-1, Wp_t, 'bandpass', hamming(N_mid));
```

After the full 9-filter evaluation, a **second comparison** tests whether raising the lower passband from the adaptive `f_p1` to a fixed `1.0 Hz (60 BPM)` improves performance. 

**Rationale:** The adaptive `f_p1` is set 0.25 Hz above the detected artifact. If the artifact is at 0.55 Hz, `f_p1 ≈ 0.80 Hz` — which still passes respiration (0.1–0.5 Hz) and residual low-frequency motion artifacts. A fixed `1.0 Hz` lower bound is physiologically justified for adult subjects at rest (no healthy adult has a resting HR < 60 BPM) and removes an entire octave of potential interference.

The tight comparison tests only the two best-candidate designs (Elliptic IIR and Hamming-mid FIR) rather than all 9, to keep the output concise. The metrics (wMAE, consistency, SNR, CCC) are printed and compared against the adaptive-passband results above.

**The console output explicitly states the interpretation:**
> "fp1=0.7Hz passes 0.7-1.0 Hz motion/respiration artifacts. fp1=1.0Hz removes that band."

---

## 12. Export to Filter Results

```matlab
out_tbl               = py_data;
out_tbl.BVP_ham_tight = S_ham_t;    % PRIMARY — best performer
out_tbl.BVP_ham_adapt = S_ham_mid;
out_tbl.BVP_el_tight  = S_el_t;
out_tbl.BVP_el_adapt  = S_el;
out_tbl.f_p1_adapt    = repmat(f_p1, T, 1);
out_tbl.f_p2_adapt    = repmat(f_p2, T, 1);
out_tbl.N_ham_mid     = repmat(N_mid, T, 1);
writetable(out_tbl, out_csv);
```

The entire `py_data` table (all original columns from Stage 1) is copied and four new BVP signal columns are appended:
- `BVP_ham_tight` — Hamming N_mid, fp1=1.0 Hz (declared primary/best)
- `BVP_ham_adapt` — Hamming N_mid, adaptive fp1
- `BVP_el_tight` — Elliptic tight fp1=1.0 Hz
- `BVP_el_adapt` — Elliptic adaptive fp1

Plus scalar metadata (replicated to all rows for convenience): `f_p1_adapt`, `f_p2_adapt`, `N_ham_mid`. The FDA stage reads these adaptive cutoff values to correctly bound its frequency estimation search.

Copying `py_data` as the base table preserves all quality metrics, per-region signals, and GT BPM columns — the FDA stage can access all of them without reading Stage 1's CSV separately.

---

## 13. Helper Functions

### 13.1 `sosfreqz()`

```matlab
function H = sosfreqz(sos, g, f, fs)
    H = ones(numel(f), 1);
    for k = 1:size(sos, 1)
        H = H .* freqz(sos(k,1:3), sos(k,4:6), f(:), fs);
    end
    H = g .* H;
end
```

MATLAB's built-in `freqz` does not directly accept SOS matrices — it expects `[b, a]` polynomial form. This helper cascades `freqz` across each 2nd-order section, multiplying the complex frequency responses, then applies the overall gain scalar `g`. The result is the exact same frequency response that `filtfilt(sos, g, x)` implements. Used for Figs 1–4.

### 13.2 `concordance_cc()`

See Section 10.2. Lin's CCC formula is implemented directly from the definition: `ccc = 2·Cov(x,y) / (Var(x) + Var(y) + (μx-μy)²)`. The `1e-9` denominator guard prevents division by zero when x and y are both constant.

---

## 14. Complete Signal Flow Diagram

```
rppg_output.csv (from acquisition stage)
           │
           ▼
┌────────────────────────────────────────────────────┐
│  Source selection                                   │
│  ┌─────────────────┐   ┌──────────────────────────┐│
│  │ Path A (primary)│   │ Path B (fallback)        ││
│  │ BVP_detrended   │   │ Raw video → Viola-Jones  ││
│  │ from Python CSV │   │ → YCbCr → CHROM → detrend││
│  └────────┬────────┘   └─────────────┬────────────┘│
│           └───────────┬──────────────┘              │
│                       ▼                             │
│                   S_det (BVP signal)                │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│  ADAPTIVE PASSBAND ESTIMATION                      │
│                                                    │
│  Welch PSD (30s window, 8192-pt FFT)               │
│       ↓                                            │
│  Rough cardiac estimate (0.67–3.5 Hz search)       │
│       ↓                                            │
│  Half-harmonic artifact detection (±0.30 Hz)       │
│       ↓                                            │
│  S/A ratio (dB) — warn if artifact in passband     │
│       ↓                                            │
│  f_p1 = artifact + 0.25 Hz (floor 0.67)           │
│  f_p2 = cardiac + 1.50 Hz (floor 3.0)             │
│  f_s1 = f_p1 - 0.35  │  f_s2 = f_p2 + 0.80       │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│  IIR DESIGN (SOS form)                             │
│  Butterworth N=auto(≤6)  Cheby I N=auto            │
│  Cheby II N=auto         Elliptic N=auto           │
│  All: Rp=1dB, Rs=40dB                             │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────┐
│  FIR DESIGN                                        │
│  Hamming N_lo (0.5s)  Hamming N_mid (2.0s)        │
│  Hamming N_hi (spec)  Kaiser (kaiserord)           │
│  Parks-McClellan (firpm, weighted equiripple)      │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────┐
│  filtfilt → 9 filtered BVP signals                 │
│                                                    │
│  Figs 1–5: frequency response + time domain        │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────┐
│  SLIDING-WINDOW BPM EVALUATION (10s, 1s step)     │
│                                                    │
│  For each window, each filter:                     │
│  Welch PSD → peak detection → harmonic correction  │
│  → BPM estimate + per-window SNR                   │
│                                                    │
│  GT interpolation → error = BPM_est - GT           │
│                                                    │
│  Metrics:                                          │
│  wMAE, RMSE, consistency, SNR_mean, CCC            │
│  Composite = 0.40·wMAE + 0.25·cons + 0.20·SNR     │
│            + 0.15·CCC                              │
│                                                    │
│  Figs 6–10: BPM tracks, errors, ranking            │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────┐
│  TIGHT PASSBAND COMPARISON (fp1=1.0 Hz)            │
│  Elliptic tight + Hamming N_mid tight              │
│  Full metrics printed; compare vs adaptive          │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
           filter_results/filterdesign_<ts>.csv
           (primary: BVP_ham_tight, fp1=1.0 Hz)
```

---

## 15. Why These Parameter Values?

| Parameter | Value | Rationale |
|---|---|---|
| `Rp = 1.0 dB` | Max passband ripple | At -1dB the signal is attenuated to 89% of its input value. This is physiologically irrelevant for rPPG (we care about relative oscillations, not absolute levels). A tighter Rp would demand higher filter orders. |
| `Rs = 40 dB` | Min stopband attenuation | At -40dB the artifact is reduced to 1% of its original amplitude. At typical S/A ratios of +6dB to +15dB, this gives a post-filter S/A of 46–55dB — more than sufficient. |
| `f_artifact + 0.25 Hz` | Lower passband margin | 0.25 Hz ensures the artifact is well into the transition band. At 0.10 Hz margin the filter would still have significant response at the artifact; at 0.50 Hz we lose too much of the low-HR range. |
| `f_p1 - 0.35 Hz` | Transition bandwidth | Larger bandwidth = lower IIR order (stable designs). 0.35 Hz transition keeps Butterworth at ≤6. A 0.10 Hz transition band would demand N>20 for Butterworth. |
| Butterworth cap `N ≤ 6` | Order cap | Order-6 bandpass = 12 poles. Double-precision floating-point arithmetic becomes unreliable for coefficients at order >8. The cap trades a small attenuation shortfall for numerical stability. |
| `N_lo = 2*round(0.5*fs/2)+1` | ~0.5s FIR | At 30fps, `N_lo ≈ 31`. Provides the fastest-response FIR for comparison. |
| `N_mid = 2*round(2.0*fs/2)+1` | ~2.0s FIR | At 30fps, `N_mid ≈ 121`. Balanced: sharp enough for good stopband rejection, short enough to not dominate 10s windows. Declared as PRIMARY. |
| `N_hi ≤ 40% of window` | Cap on hi-order FIR | A filter longer than 40% of the analysis window causes >40% of each window to be a startup transient. Beyond this point, longer is counterproductive. |
| Harmonic correction `≥ 30%` | Power threshold for doubling | Conservative: avoids doubling legitimate sub-1-Hz estimates while correcting strong half-harmonic artifacts. 50% would miss weak cardiac signals; 5% would cause too many false doublings. |
| `f_p1_tight = 1.0 Hz` | Tight lower bound | Adults at rest: HR > 60 BPM guaranteed. Removing 0.67–1.0 Hz eliminates the respiration band (0.1–0.5 Hz) plus a 0.5 Hz margin. |
| Composite weights `0.40/0.25/0.20/0.15` | Accuracy priority | BPM accuracy (wMAE) is the primary clinical requirement. Consistency and SNR matter for deployment reliability. CCC rounds out the statistical picture. |

---

## 16. Appendix — References

### A.1 IIR Filter Design

- **Proakis & Manolakis:** Proakis, J. G., & Manolakis, D. K. (2006). *Digital Signal Processing: Principles, Algorithms, and Applications.* 4th ed. Prentice Hall. Chapters 8–9 (Butterworth, Chebyshev, Elliptic design theory).
- **MATLAB `buttord` / `butter` docs:** [https://www.mathworks.com/help/signal/ref/butter.html](https://www.mathworks.com/help/signal/ref/butter.html)
- **SOS stability:** Deczky, A. G. (1972). *Synthesis of Recursive Digital Filters Using the Minimum p-Error Criterion.* IEEE TASP. The numerical advantage of SOS over direct-form is well-established; MATLAB's `filtfilt` documentation explicitly recommends SOS for orders ≥ 4.
- **`filtfilt` zero-phase:** [https://www.mathworks.com/help/signal/ref/filtfilt.html](https://www.mathworks.com/help/signal/ref/filtfilt.html)

### A.2 FIR Filter Design

- **Kaiser window formula:** Kaiser, J. F. (1974). *Nonrecursive digital filter design using the Io-sinh window function.* Proc. 1974 IEEE ISCAS, 20–23.
- **Parks-McClellan algorithm:** Parks, T. W., & McClellan, J. H. (1972). *Chebyshev Approximation for Nonrecursive Digital Filters with Linear Phase.* IEEE TCAS, 19(2), 189–194. [https://doi.org/10.1109/TCT.1972.1083419](https://doi.org/10.1109/TCT.1972.1083419)
- **MATLAB `firpm` docs:** [https://www.mathworks.com/help/signal/ref/firpm.html](https://www.mathworks.com/help/signal/ref/firpm.html)
- **MATLAB `kaiserord` docs:** [https://www.mathworks.com/help/signal/ref/kaiserord.html](https://www.mathworks.com/help/signal/ref/kaiserord.html)

### A.3 Lin's CCC

- **Lin, L. I. (1989).** *A Concordance Correlation Coefficient to Evaluate Reproducibility.* Biometrics, 45(1), 255–268. [https://doi.org/10.2307/2532051](https://doi.org/10.2307/2532051)
- CCC was chosen over Pearson r specifically because the GT heart rate barely varies during a resting recording. CCC correctly handles low-variance GT signals where Pearson r degrades.

### A.4 Welch PSD

- **Welch, P. D. (1967).** *The Use of Fast Fourier Transform for the Estimation of Power Spectra.* IEEE TAES, 15(2), 70–73. [https://doi.org/10.1109/TAES.1967.5408896](https://doi.org/10.1109/TAES.1967.5408896)
- **MATLAB `pwelch` docs:** [https://www.mathworks.com/help/signal/ref/pwelch.html](https://www.mathworks.com/help/signal/ref/pwelch.html)

### A.5 Harmonic Correction in rPPG

- The half-harmonic artifact is discussed extensively in: de Haan, G., & Jeanne, V. (2013). *Robust Pulse Rate From Chrominance-Based rPPG.* IEEE TBME, 60(10). [https://doi.org/10.1109/TBME.2013.2266196](https://doi.org/10.1109/TBME.2013.2266196)
- Wang, W. et al. (2017). *Algorithmic Principles of Remote PPG.* IEEE TBME, 64(7). [https://doi.org/10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282)

### A.6 Viola-Jones Face Detector (Path B fallback)

- **Viola, P., & Jones, M. (2001).** *Rapid Object Detection Using a Boosted Cascade of Simple Features.* CVPR 2001. [https://doi.org/10.1109/CVPR.2001.990517](https://doi.org/10.1109/CVPR.2001.990517)

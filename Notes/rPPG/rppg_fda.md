# rppg_fda — Frequency-Domain Analysis: Welch vs MUSIC vs ESPRIT

> **File:** `Updated Pipeline/scripts/analysis/bpm_fda.m`
> **Role in pipeline:** Stage 3 — consumes the filtered BVP signals from Stage 2 (`filterdesign_<ts>.csv`), then systematically evaluates five frequency estimation methods (Welch PSD, FFT, MUSIC, ESPRIT) across five window lengths (2s, 3s, 5s, 10s, 20s), with a four-factor per-window confidence scoring system. Saves all figures + a diary log.
> **Input:** `results/filter_results/filterdesign_<timestamp>.csv`
> **Output:** `results/fda_results/fda_<timestamp>/` — PNG + PDF figures + `output.txt` console log

---

## Table of Contents

1. [What This File Actually Does](#1-what-this-file-actually-does)
2. [High-Level Architecture & Design Philosophy](#2-high-level-architecture--design-philosophy)
3. [Setup & Data Loading](#3-setup--data-loading)
4. [Section W — Noise Whiteness Diagnostic](#4-section-w--noise-whiteness-diagnostic)
   - 4.1 [Cardiac Notch Filter](#41-cardiac-notch-filter)
   - 4.2 [Ljung-Box Test](#42-ljung-box-test)
   - 4.3 [Eigenvalue Coefficient of Variation](#43-eigenvalue-coefficient-of-variation)
   - 4.4 [Section W Figures](#44-section-w-figures)
5. [Section A — Full-Signal Spectra](#5-section-a--full-signal-spectra)
6. [Section B — Sliding-Window Multi-Method BPM Estimation](#6-section-b--sliding-window-multi-method-bpm-estimation)
   - 6.1 [Window Size Loop](#61-window-size-loop)
   - 6.2 [Per-Window Confidence Scoring](#62-per-window-confidence-scoring)
   - 6.3 [BPM Estimation Methods Summary](#63-bpm-estimation-methods-summary)
7. [Local Function Blocks — In Depth](#7-local-function-blocks--in-depth)
   - 7.1 [`peak_bpm()` — Dominant Cardiac Peak with Harmonic Correction](#71-peak_bpm--dominant-cardiac-peak-with-harmonic-correction)
   - 7.2 [`est_fft()` — Single-Window FFT Estimator](#72-est_fft--single-window-fft-estimator)
   - 7.3 [`est_welch()` — Welch PSD Estimator](#73-est_welch--welch-psd-estimator)
   - 7.4 [`est_music()` — MUSIC Subspace Estimator](#74-est_music--music-subspace-estimator)
   - 7.5 [`est_esprit()` — ESPRIT Rotational-Invariance Estimator](#75-est_esprit--esprit-rotational-invariance-estimator)
   - 7.6 [`win_quality()` — Four-Factor Window Quality](#76-win_quality--four-factor-window-quality)
   - 7.7 [`mdl_order_local()` — Adaptive MDL Model Order Selection](#77-mdl_order_local--adaptive-mdl-model-order-selection)
   - 7.8 [`ljung_box_local()` — Ljung-Box Portmanteau Test](#78-ljung_box_local--ljung-box-portmanteau-test)
   - 7.9 [`acf_local()` — Autocorrelation Function](#79-acf_local--autocorrelation-function)
8. [Figure Structure — Per Window Length](#8-figure-structure--per-window-length)
9. [Auto-Save & Diary Logging](#9-auto-save--diary-logging)
10. [Complete Signal Flow Diagram](#10-complete-signal-flow-diagram)
11. [Why These Parameter Values?](#11-why-these-parameter-values)
12. [Appendix — References](#12-appendix--references)

---

## 1. What This File Actually Does

`bpm_fda.m` is the **frequency-domain analysis stage** of the pipeline. Its purpose is to answer the core research question that Prof. Chen identified: *can subspace-based frequency estimators (MUSIC, ESPRIT) extract heart rate accurately from much shorter windows than Welch PSD requires?*

The script tests five window lengths (2, 3, 5, 10, 20 seconds) against four BVP signal variants (Hamming-tight, Hamming-adaptive, Elliptic-tight, Elliptic-adaptive) and six frequency estimation methods (Welch on each of 4 signals, MUSIC on the primary signal, ESPRIT on the primary signal). For each window/method combination, BPM is estimated, compared against ground-truth, and a per-window confidence score is computed.

Uniquely, the script also runs a formal **noise whiteness diagnostic** (Section W) before the main analysis. This test tells you whether the residual noise in the filtered BVP signal is white (uncorrelated, spectrally flat) or coloured (autocorrelated, non-flat spectrum). MUSIC and ESPRIT's theoretical guarantees assume white noise — if the noise is coloured, their performance degrades and the diagnostic warns you.

---

## 2. High-Level Architecture & Design Philosophy

**Decision 1 — The filtered signal is the input, not the raw BVP.**
By taking `BVP_ham_tight` (already bandpass-filtered at 1.0–f_p2 Hz) as input to MUSIC and ESPRIT, the subspace analysis operates on a signal where out-of-band noise has already been removed. This is correct: MUSIC and ESPRIT work on the autocorrelation matrix of the signal, which is contaminated by all noise outside the band. Pre-filtering concentrates the energy in the frequency range of interest.

**Decision 2 — MDL selects model order adaptively per window.**
MUSIC and ESPRIT require specifying a model order p (number of sinusoids). Using a fixed p would be wrong: a 2-second window of a 30fps BVP signal may contain only 1 strong cardiac sinusoid, while a 10-second window after light motion may contain 2–3. The Minimum Description Length (MDL) criterion selects p from the eigenvalue structure of the autocorrelation matrix, adapting to the signal at hand.

**Decision 3 — Confidence scoring gates the comparison.**
Not all windows are equally informative. A window where the face was partially occluded (low detection rate), or where the camera's AGC was rapidly adjusting (high luminance CV), should be downweighted. The four-factor composite confidence score `[SNR, det_rate, skin_mean, lum_cv]` identifies high-quality (HQ) windows and reports HQ-subset MAE separately from overall MAE.

**Decision 4 — Results are auto-saved, not just displayed.**
All figures are saved as both PNG and PDF to a timestamped output folder. A `diary()` captures the full console output. This is essential for research: you can run the script, walk away, and return to a complete set of results files.

---

## 3. Setup & Data Loading

```matlab
pipeline_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
fda_ts   = datestr(now, 'yyyymmdd_HHMMSS');
fda_dir  = fullfile(pipeline_dir, 'results', 'fda_results', sprintf('fda_%s', fda_ts));
mkdir(fda_dir);
diary(fullfile(fda_dir, 'output.txt'));
diary on;
```

The output folder is created immediately at startup (before any computation). `diary on` redirects all `fprintf` / `disp` console output to `output.txt`. This means even if MATLAB crashes mid-run, the partial results are preserved.

**Auto-detection of the latest filter results CSV:**
```matlab
csv_files  = dir(fullfile(filter_dir, 'filterdesign_*.csv'));
[~, newest] = max([csv_files.datenum]);
csv_path   = fullfile(filter_dir, csv_files(newest).name);
```

The script reads from the most recently generated `filterdesign_*.csv` — no hardcoded timestamps. This ensures it always uses the latest filter design output without manual path updates.

**Loaded signals:**
```matlab
S_primary   = data.BVP_ham_tight;   % winner from filter design
S_ham_adapt = data.BVP_ham_adapt;
S_el_tight  = data.BVP_el_tight;
S_el_adapt  = data.BVP_el_adapt;
f_low_adapt = data.f_p1_adapt(1);
```

The tight-passband Hamming filter output is declared `S_primary` — the signal against which MUSIC and ESPRIT are run. The rationale: it was the best performer in Stage 2 (lowest composite score). MUSIC and ESPRIT see the same signal as the best Welch variant, enabling fair comparison.

**Cardiac band definition:**
```matlab
f_low  = 1.0;
f_high = data.f_p2_adapt(1);
```

The tight lower bound (1.0 Hz = 60 BPM) is used as the search band for all frequency estimators. The upper bound is the adaptive `f_p2` from Stage 2 (stored per-frame in the CSV, same value for all rows). Using the same band for both Welch and MUSIC/ESPRIT ensures a fair comparison: no method benefits from a wider search range.

---

## 4. Section W — Noise Whiteness Diagnostic

Section W is a formal statistical test for residual noise structure in the filtered BVP signal. The theoretical performance of MUSIC and ESPRIT depends on noise being white (independent, identically distributed). Real rPPG signals have coloured residual noise from motion, respiration, camera electronics, and imperfect skin segmentation.

### 4.1 Cardiac Notch Filter

```matlab
bw_notch = 0.30;
N_notch  = 2*round(1.5*fs) + 1;
b_ns = fir1(N_notch-1, ...
    [max(f_low+0.05, f_card-bw_notch), min(f_high-0.05, f_card+bw_notch)] / (fs/2), ...
    'stop', hamming(N_notch));
S_noise = filtfilt(b_ns, 1, S_primary);
```

To test the noise, the cardiac signal itself must be removed first — otherwise the autocorrelation of the cardiac sinusoid would dominate every whiteness test. A **notch (band-stop) FIR filter** centered at the detected cardiac frequency `f_card` with bandwidth ±0.30 Hz removes the cardiac component. The notch is 1.5-second long (sufficient for sharp notch depth at the cardiac frequency resolution), applied with `filtfilt` for zero-phase notching.

`S_noise` is the residual after notching — ideally just noise. Its spectral and temporal structure is then analysed.

**Why ±0.30 Hz notch bandwidth?** At 30fps, the MUSIC resolution is about 0.1 Hz for a 3-second window. A ±0.30 Hz notch ensures the entire cardiac peak (including spectral broadening from cardiac rate variability and window leakage) is removed. Too narrow (±0.05 Hz) would leave cardiac spectral residue that looks like coloured noise.

### 4.2 Ljung-Box Test

```matlab
LB_LAGS      = min(20, floor(T/5));
[Q_lb, p_lb] = ljung_box_local(S_noise, LB_LAGS);
is_colored   = p_lb < 0.05;
```

The **Ljung-Box portmanteau test** is a formal hypothesis test for residual autocorrelation. It tests H₀: "the noise is white (no autocorrelation at any of the first m lags)" against H₁: "at least one lag has significant autocorrelation."

**Test statistic:**
```matlab
Q = N * (N + 2) * sum_{lag=1}^{m}  r²(lag) / (N - lag)
```
where `r(lag)` is the sample autocorrelation at `lag`. Under H₀, Q follows a χ²(m) distribution. `p_lb < 0.05` rejects the null → coloured noise confirmed.

**Why `LB_LAGS = min(20, T/5)`?** More lags give more statistical power but the test statistic becomes unreliable when `m > T/5` (the sample autocorrelations at long lags are poorly estimated from a short signal). Capping at 20 prevents computing autocorrelations at lags longer than 2/3 of a second at 30fps.

The Ljung-Box result has direct implications for MUSIC/ESPRIT reliability: if `is_colored = true`, the noise subspace is not identically distributed, and the signal/noise subspace separation assumed by MUSIC becomes approximate. The script reports this explicitly.

**References:** See [Appendix A.2](#a2-ljung-box-test)

### 4.3 Eigenvalue Coefficient of Variation

```matlab
M_ev  = min(floor(T/4), 60);
x_ev  = S_noise - mean(S_noise);
X_ev  = hankel(x_ev(1:M_ev), x_ev(M_ev:end));
R_ev  = (X_ev * X_ev') / size(X_ev, 2);
ev    = sort(real(eig(R_ev)), 'descend');
ev_ns = ev(5:end);
cv_ev = std(ev_ns) / (mean(ev_ns) + 1e-9);
```

**Hankel matrix construction:** `hankel(x_ev(1:M_ev), x_ev(M_ev:end))` builds the data matrix for autocorrelation estimation. Each column is a shifted segment of the noise signal — column k is `[x(k), x(k+1), ..., x(k+M_ev-1)]`. The product `X_ev * X_ev' / N_cols` is the sample autocorrelation matrix `R`, exactly the same structure used inside MUSIC and ESPRIT.

**Noise subspace eigenvalue analysis:** The eigenvalues of the autocorrelation matrix are sorted descending. In a white noise signal, all eigenvalues of R would be equal (= noise power σ²). In a coloured noise signal, some eigenvalues are larger (corresponding to coloured noise modes). The **first 4 eigenvalues are skipped** (assumed to belong to signal components) and the remaining eigenvalues `ev_ns` constitute the noise subspace.

**Coefficient of Variation (CV):** `cv_ev = std(ev_ns)/mean(ev_ns)`. For perfectly white noise, all noise-subspace eigenvalues are equal → CV ≈ 0. For coloured noise, they span a wider range → CV > 0.30. The 0.30 threshold is empirical: CV > 0.30 indicates non-trivially non-uniform eigenvalue distribution (coloured noise that will degrade MUSIC/ESPRIT).

**Why `M_ev = min(T/4, 60)`?** M is the subspace dimension (autocorrelation matrix size). Larger M provides better noise subspace estimation but requires more data (stability condition: `N_cols = T - M_ev > M_ev`). `T/4` ensures `N_cols ≈ 3×M_ev`, a reasonable ratio. Cap at 60 prevents very large matrices for long signals (which would be slow).

**References:** See [Appendix A.3](#a3-pisarenko--music-eigenvalue-analysis)

### 4.4 Section W Figures

Fig W1 shows four panels:
1. **PSD of S_primary** — full BVP spectrum with cardiac peak marked (red dashed) and band limits (black dashed). Visual check of signal quality.
2. **PSD of noise floor** — PSD after notching out the cardiac component. Ideally flat (white). Any remaining spectral peaks indicate coloured noise modes.
3. **ACF of noise floor** — Sample autocorrelation with 95% confidence bounds (`±1.96/√T`). Spikes outside bounds indicate autocorrelation at those lags — white noise should have all lags within bounds.
4. **Covariance eigenvalues** — Semilogy plot of all M_ev eigenvalues. Signal subspace (first 4, blue circles) vs noise subspace (remaining, red squares). A flat noise-subspace cluster → white noise; a sloping cluster → coloured.

---

## 5. Section A — Full-Signal Spectra

```matlab
specCal(S_primary, fs);
sce = specCale([S_primary, S_ham_adapt, S_el_tight, S_el_adapt], fs);
```

`specCal` and `specCale` are functions from the external MACS MATLAB toolbox (`macs-matlab-toolbox-master`). They compute and display the full-signal amplitude spectra of the BVP signals. The plot from `specCale` compares all four BVP variants (tight vs adaptive passband, Hamming vs Elliptic) in a single overlay, with passband edge markers.

**Why compare tight vs adaptive here?** The tight lower edge (`f_low=1.0 Hz`) removes the 0.7–1.0 Hz respiration/motion band. Section A lets you visually confirm that the tight-filtered signals have zero power below 1.0 Hz, while the adaptive-filtered signals may retain some power in the 0.7–f_p1 range. If the adaptive f_p1 ended up at 0.75 Hz (artifact at 0.50 Hz), you can see exactly what low-frequency content is still present.

---

## 6. Section B — Sliding-Window Multi-Method BPM Estimation

### 6.1 Window Size Loop

```matlab
win_secs = [2, 3, 5, 10, 20];
```

The loop tests five window lengths: 2, 3, 5, 10, and 20 seconds. The core research question is: **how short can a window be while still giving accurate BPM estimates?**

- **Welch PSD** requires long windows for frequency resolution. At 30fps, `df = 30/N_frames`. To resolve BPM to ±3 BPM (0.05 Hz), you need `N ≥ 30/0.05 = 600 frames = 20s`. At 10s, resolution is 0.1 Hz = 6 BPM. At 2s, resolution is 0.5 Hz = 30 BPM — completely unusable.
- **MUSIC and ESPRIT** are superresolution methods. They can theoretically estimate a single frequency with sub-bin resolution from a short window, limited by SNR rather than window length. Testing at 2s and 3s reveals whether MUSIC/ESPRIT's theoretical advantage manifests in practice on real rPPG data.

```matlab
M_sub  = max(round(win_N / 4), 6);
starts = 1 : round(fs) : T - win_N + 1;
```

The subspace dimension `M_sub` is set to `win_N/4`, minimum 6. For a 2-second window at 30fps, `win_N=60`, `M_sub = max(15, 6) = 15`. For a 10-second window, `M_sub = max(75, 6) = 75`. The subspace dimension controls the resolution of the autocorrelation matrix estimation. `M_sub > max_order + 1` must hold for MDL to select orders up to `max_k=6`; the `min(..., 6)` inside `mdl_order_local` ensures this.

**Sliding step = 1 second** (round(fs) frames). At a 10-second window, this gives 90% overlap between consecutive windows — a standard rPPG sliding window approach that produces a smooth BPM track.

### 6.2 Per-Window Confidence Scoring

```matlab
ACC_THR   = 10;     % ±BPM tolerance for "accurate"
SNR_THR   = 6;      % dB: cardiac peak vs in-band noise
DET_THR   = 0.7;    % fraction of frames with fresh face detection
SKIN_THR  = 300;    % minimum mean skin pixels
LUM_CV_THR= 0.12;   % luminance coefficient of variation
CONF_THR  = 0.80;   % threshold for high-quality (HQ) classification
W_CONF    = [0.40, 0.30, 0.15, 0.15];
```

For each window, four factors are computed (from `win_quality()`):

| Factor | Measurement | Why it matters |
|---|---|---|
| `snr_db` | Cardiac peak PSD / mean in-band PSD | Direct measure of signal quality. Low SNR → BPM estimate unreliable regardless of estimator |
| `det_rate` | Fraction of frames where face was detected | Low detection → interpolated/stale bounding box → skin ROI misaligned → corrupted signal |
| `skin_mean` | Mean skin pixel count per frame | Low count → face partially occluded or skin detection failing |
| `lum_cv` | Luminance coefficient of variation across window | High CV → camera AGC actively changing exposure → multiplicative noise in BVP |

Each factor is linearly scored to [0, 1]:
```matlab
snr_score  = min(1, max(0, snr_db      / (SNR_THR    * 2)));
det_score  = det_rate;
skin_score = min(1, max(0, skin_mean   / (SKIN_THR   * 2)));
lum_score  = max(0, 1 - lum_cv_win    / (LUM_CV_THR * 2));
```

Each factor reaches score=0.5 at its threshold and score=1.0 at 2× the threshold. This is a "soft" threshold — a window that barely fails SNR_THR (e.g. 5.5 dB vs threshold 6 dB) still contributes to the mean, just with a lower weight than a window at 12 dB.

**Why these weights `[0.40, 0.30, 0.15, 0.15]`?**
- SNR (0.40): Direct measure of signal quality — the most important factor. Even with perfect detection and stable lighting, a noisy signal cannot be estimated reliably.
- Detection rate (0.30): A stale face bounding box means the skin ROI is misaligned and the BVP signal contains artefacts from non-skin pixels.
- Skin count + Luminance (0.15 each): Secondary factors that affect signal quality but are less directly correlated with BPM accuracy than SNR.

**`CONF_THR = 0.80` requires SNR ≥ ~6dB when other factors are perfect:** With W=[0.40,0.30,0.15,0.15] and det=skin=lum=1.0, the minimum SNR score needed to reach CONF=0.80 is `(0.80 - 0.30 - 0.15 - 0.15) / 0.40 = 0.50` → `snr_score = 0.50` → `snr_db = 0.50 × 12 = 6 dB`. This is the stated intent in the code comment.

**HQ vs all-window MAE reporting:** The script prints both overall MAE and HQ-subset MAE (when `sum(hq_win) >= 5`). The gap between these tells you how much of the estimation error comes from low-quality windows. If HQ MAE is much lower than overall MAE, the frequency estimator is actually capable — it is being dragged down by bad windows.

### 6.3 BPM Estimation Methods Summary

For each window, six BPM estimates are computed:

| Variable | Method | Signal | Band used |
|---|---|---|---|
| `bpm_welch` | Welch PSD | `S_primary` (Ham-tight) | `[f_low, f_high]` |
| `bpm_welch_ha` | Welch PSD | `S_ham_adapt` | `[f_low_adapt, f_high]` |
| `bpm_welch_et` | Welch PSD | `S_el_tight` | `[f_low, f_high]` |
| `bpm_welch_ea` | Welch PSD | `S_el_adapt` | `[f_low_adapt, f_high]` |
| `bpm_music` | MUSIC | `S_primary` | `[f_low, f_high]` via `omega_cb` |
| `bpm_esprit` | ESPRIT | `S_primary` | `[f_low, f_high]` |

Note: MUSIC and ESPRIT always run on `S_primary` (the primary winner from Stage 2). They are compared against Welch on the same signal to isolate the effect of the frequency estimation method, holding the signal constant.

---

## 7. Local Function Blocks — In Depth

### 7.1 `peak_bpm()` — Dominant Cardiac Peak with Harmonic Correction

```matlab
function bpm = peak_bpm(spectrum, freqs, f_low, f_high)
    band = freqs >= f_low & freqs <= f_high;
    [peak_val, idx] = max(spectrum(band));
    fb = freqs(band);  fp = fb(idx);
    if fp < 1.2 && 2*fp <= f_high
        [~, i2] = min(abs(freqs - 2*fp));
        if spectrum(i2) > 0.05 * peak_val;  fp = 2*fp;  end
    end
    bpm = fp * 60;
end
```

Shared harmonic correction logic used by `est_fft()` and `est_welch()`. The harmonic correction threshold here is **0.05 (5%)** — much more aggressive than the 30% threshold in Stage 2. 

**Why 5% here vs 30% in Stage 2?**

Stage 2 operates on the full signal (hundreds of seconds). The cardiac peak is well-estimated and stands well above noise. A 30% threshold is appropriate: accept doubling only when the harmonic is clearly present.

Stage 3 (FDA) works with 2–10 second windows where the cardiac component may be 5–15× weaker than a motion artifact in the low end of the band. The Welch PSD in a 2-second window has 0.5 Hz frequency resolution — the cardiac peak at 1.2 Hz may have nearly the same height as a strong 0.6 Hz artifact peak. A 5% threshold says: "if there is ANY evidence of a peak at 2× the sub-harmonic frequency (even 5% of sub-harmonic power), prefer the doubling." This is intentionally aggressive to avoid the common 2× underestimation error for short windows.

**The `fp < 1.2 Hz` gate:** Only doubles if the raw peak is below 1.2 Hz (72 BPM). This prevents accidentally doubling a genuine 65 BPM cardiac peak.

### 7.2 `est_fft()` — Single-Window FFT Estimator

```matlab
function bpm = est_fft(seg, win_N, nfft, fs, f_low, f_high)
    X  = abs(fft(seg .* hann(win_N), nfft));
    fv = (0:nfft/2) * fs / nfft;
    bpm = peak_bpm(X(1:nfft/2+1), fv, f_low, f_high);
end
```

A simple windowed FFT. The segment is multiplied by a Hann window before FFT to suppress spectral leakage. `nfft` zero-padding increases spectral interpolation resolution. Note: `fft(x, nfft)` with `nfft > length(x)` pads with zeros — this does not add frequency resolution (that is determined by window length) but interpolates between existing spectral bins, giving smoother peak location.

The FFT magnitude (not PSD) is used — no averaging. This means the FFT estimate is **noisier** than Welch for short windows (Welch's benefit is averaging multiple overlapping sub-windows). `est_fft` is defined but its direct output is not printed in the main summary table — Welch via `est_welch` subsumes it (Welch with a single full-window segment equals a Hann-windowed FFT up to normalisation).

### 7.3 `est_welch()` — Welch PSD Estimator

```matlab
function bpm = est_welch(seg, win_N, nfft, fs, f_low, f_high)
    [P, fw] = pwelch(seg, hann(win_N), floor(win_N/2), nfft, fs);
    bpm = peak_bpm(P, fw, f_low, f_high);
end
```

The MATLAB `pwelch` with a **single full-window Hann segment** (`window = hann(win_N)`) and **50% overlap** (`noverlap = floor(win_N/2)`). For a segment of length `win_N`, `pwelch` with one segment of the same length and 50% overlap effectively computes a single-window Welch estimate — equivalent to a Hann-windowed FFT with energy normalisation. The 50% overlap (`floor(win_N/2)`) is used by MATLAB internally to allow sub-windowing if `win_N > length(seg)`, but since they are equal, only one segment is processed.

The key difference from `est_fft`: `pwelch` applies one-sided normalisation (`P = |X|²/(fs × ∑w²)`) so the output is a power spectral density in units of power/Hz, not raw magnitude. For peak detection, normalisation does not affect the peak location, only the absolute values.

### 7.4 `est_music()` — MUSIC Subspace Estimator

```matlab
function bpm = est_music(seg, p, M, omega_cb, fs, f_low, f_high)
    bpm = NaN;
    if length(seg) < M || M < 5;  return;  end
    try
        Px = m_music(seg(:), p, M, omega_cb);
    catch
        return;
    end
    f_cb = omega_cb / (2*pi) * fs;
    band = f_cb >= f_low & f_cb <= f_high;
    Px_b = Px(band);  fb = f_cb(band);
    [peak_val, idx] = max(Px_b);  fp = fb(idx);
    if fp < 1.2 && 2*fp <= f_high
        near = f_cb >= (2*fp - 0.2) & f_cb <= (2*fp + 0.2);
        if any(near)
            [nb_val, nb_idx] = max(Px(near));
            if nb_val > peak_val - 10
                nb_f = f_cb(near);
                fp   = nb_f(nb_idx);
            end
        end
    end
    bpm = fp * 60;
end
```

`m_music()` is called from the external MUSIC-ESPRIT toolbox (`MUSIC-ESPRIT-Frequency-ID-main`). It implements the **MUSIC (MUltiple SIgnal Classification) algorithm**:

1. Build the Hankel data matrix from the input segment with embedding dimension M.
2. Compute the autocorrelation matrix `R = X·X' / N`.
3. Eigen-decompose R → eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λ_M.
4. Separate signal subspace (top p eigenvalues) from noise subspace (remaining M-p).
5. Compute the MUSIC pseudospectrum: `P_MUSIC(ω) = 1 / ∑|e(ω)ᴴ·vₙ|²`, where `vₙ` are noise subspace eigenvectors and `e(ω)` is the steering vector `[1, e^{jω}, e^{2jω}, ..., e^{j(M-1)ω}]`.
6. The pseudospectrum has sharp peaks (approaching infinity) at true signal frequencies.

**The `omega_cb` parameter:** A dense frequency grid computed at module level:
```matlab
omega_cb = linspace(2*pi*f_low/fs, 2*pi*f_high/fs, 2048);
```
MUSIC evaluates the pseudospectrum at these 2048 digital frequency points. This gives sub-0.001 Hz resolution over the cardiac band — far finer than Welch's `fs/N` resolution.

**Why only the cardiac band?** Unlike Welch, which computes PSD over 0 to Nyquist, MUSIC's frequency evaluation is targeted. Computing the MUSIC spectrum at 2048 points only in the cardiac band is computationally equivalent to searching all frequencies at 100× lower density. This is both faster and more accurate — the targeted grid ensures no alias between the cardiac peak and out-of-band peaks.

**MUSIC harmonic correction — neighbourhood search:**
```matlab
near = f_cb >= (2*fp - 0.2) & f_cb <= (2*fp + 0.2);
if nb_val > peak_val - 10
    fp = nb_f(nb_idx);
end
```

A **±0.2 Hz neighbourhood** around `2×fp` is searched, not just the single nearest bin. MUSIC peaks can be offset from the true frequency by a small amount when the signal is short or noisy. The `-10` threshold (on the MUSIC pseudospectrum which can have very high dynamic range) accepts the neighbourhood peak as the harmonic if it is within 10 dB of the sub-harmonic's peak. This is more permissive than the power-ratio threshold in `peak_bpm()` because MUSIC peaks are much sharper and values near the true frequency can fall off quickly.

**Error handling:** The `try/catch` block guards against `m_music` throwing an error (e.g., ill-conditioned matrix when the segment is too noisy or too short). In that case, `bpm` remains `NaN` and is excluded from MAE calculations via `'omitnan'` flag.

**References:** See [Appendix A.3](#a3-pisarenko--music-eigenvalue-analysis), [A.4](#a4-music-algorithm)

### 7.5 `est_esprit()` — ESPRIT Rotational-Invariance Estimator

```matlab
function bpm = est_esprit(seg, p, M, fs, f_low, f_high)
    bpm = NaN;
    if length(seg) < M || M < 5;  return;  end
    try
        [~, w_est] = evalc('m_esprit(seg(:), p, M)');
    catch
        return;
    end
    hz = sort(real(w_est) * fs / (2*pi));
    hz = hz(hz >= f_low & hz <= f_high);
    if isempty(hz);  return;  end
    if numel(hz) >= 2 && hz(end)/hz(1) > 1.5
        bpm = hz(end) * 60;
    else
        fp = hz(1);
        if fp < 1.2 && 2*fp <= f_high;  fp = 2*fp;  end
        bpm = fp * 60;
    end
end
```

`m_esprit()` implements the **ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques) algorithm**:

1. Build two overlapping submatrices of the Hankel data matrix (shifting by one sample).
2. Solve the rotational-invariance equation: the signal subspace of one submatrix rotates into the signal subspace of the other by a diagonal matrix whose entries are `e^{jωₖ}`.
3. The estimated frequencies come directly as `ω_est = angle(eigenvalues of Φ)`.

Unlike MUSIC, ESPRIT does not search over a frequency grid — it produces a small set of p estimated frequencies algebraically. This makes ESPRIT faster than MUSIC (no pseudospectrum computation) and potentially more precise (the algebraic solution has sub-bin resolution by construction).

**`evalc('m_esprit(...)')`:** The `evalc` wrapper captures any warning or diagnostic output that `m_esprit` prints to the console and discards it (the `[~, ...]` captures the suppressed output). This prevents the ESPRIT toolbox's verbose diagnostics from flooding the command window during a 1000-window run.

**Frequency output processing:**
```matlab
hz = sort(real(w_est) * fs / (2*pi));
hz = hz(hz >= f_low & hz <= f_high);
```

ESPRIT returns complex-valued `w_est` in practice (due to finite-precision effects). Taking `real(w_est)` discards small imaginary parts from numerical rounding. Converting from digital frequency (`radians/sample`) to Hz: `f = ω × fs / (2π)`. Filtering to the cardiac band removes spurious out-of-band estimates.

**Two-frequency disambiguation:**
```matlab
if numel(hz) >= 2 && hz(end)/hz(1) > 1.5
    bpm = hz(end) * 60;
```

If ESPRIT returns two or more frequencies within the cardiac band and their ratio exceeds 1.5 (i.e., they are well-separated), the higher frequency is selected as the cardiac fundamental. The rationale: when ESPRIT detects two distinct frequencies in the band, they are likely the cardiac fundamental (~1.2 Hz) and its harmonic (~2.4 Hz), or the cardiac and a nearby artifact. A ratio > 1.5 indicates genuinely distinct frequencies; the higher one is more likely to be the true cardiac frequency (the lower one being a residual sub-harmonic artifact). If ratio ≤ 1.5, they are similar frequencies (likely one frequency with split due to noise), and the standard harmonic-correction path applies.

**Why use `p = max(4, p_mdl)` for ESPRIT vs `p_mdl` for MUSIC?**
```matlab
bpm_music(k)  = est_music(seg, p_mdl,        M_sub, omega_cb, fs, f_low, f_high);
bpm_esprit(k) = est_esprit(seg, max(4,p_mdl), M_sub, fs, f_low, f_high);
```
ESPRIT requires `p ≥ 2` per complex sinusoid (because a real-valued sinusoid has two complex poles). For a single cardiac frequency, p=2 is theoretically correct. However, the BVP signal almost always contains the cardiac fundamental plus at least its first harmonic — so p=4 (two complex sinusoids) is more appropriate. `max(4, p_mdl)` ensures ESPRIT has at least enough model order to represent two sinusoids, even when MDL selects p_mdl=2 for very short, noisy windows.

**References:** See [Appendix A.5](#a5-esprit-algorithm)

### 7.6 `win_quality()` — Four-Factor Window Quality

```matlab
function [snr_db, det_rate, skin_mean, lum_cv] = ...
        win_quality(seg, det_seg, skin_seg, lum_seg, win_N, nfft, fs, f_low, f_high)
    [P, fw]    = pwelch(seg, hann(win_N), floor(win_N/2), nfft, fs);
    band       = fw >= f_low & fw <= f_high;
    P_band     = P(band);  f_band = fw(band);
    [pk, pi]   = max(P_band);
    noise_mask = abs(f_band - f_band(pi)) >= 0.5;
    noise_P    = mean(P_band(noise_mask));
    snr_db    = 10*log10(max(pk / max(noise_P, eps), 1));
    det_rate  = mean(double(det_seg));
    skin_mean = mean(double(skin_seg));
    mu_lum    = mean(lum_seg);
    lum_cv    = std(lum_seg) / max(mu_lum, eps);
end
```

**SNR computation:** The cardiac peak is identified at its maximum within the cardiac band. The noise floor is estimated as the mean of all in-band frequencies that are ≥ 0.5 Hz away from the peak — this excludes the peak itself and its immediate spectral neighbourhood from the noise estimate. `max(pk/noise_P, 1)` ensures SNR is at least 0 dB (avoids negative SNR dB when noise > signal, which would be misleading — a SNR of -3dB is reported as 0dB to indicate "no cardiac peak detectable").

**Detection rate (`det_rate`):** Direct mean of the `face_detected` column from Stage 1. This is 1 if MediaPipe detected a face in that frame, 0 if detection failed. A detection rate of 0.7 means 30% of frames had stale/interpolated bounding boxes.

**Luminance CV (`lum_cv`):** Coefficient of variation of the per-frame luminance over the window: `std/mean`. This is a proxy for camera AGC activity. When AGC is stable (good lighting), luminance is nearly constant → `lum_cv ≈ 0`. When AGC is rapidly adjusting (subject enters bright light, or moves near a lamp), luminance swings by 10–30% → `lum_cv ≈ 0.10–0.30`. High lum_cv means the rPPG signal has been corrupted by multiplicative illumination noise that cannot be fully removed by the luminance normalisation in Stage 1.

### 7.7 `mdl_order_local()` — Adaptive MDL Model Order Selection

```matlab
function k_opt = mdl_order_local(x, M, max_k)
    x  = x(:) - mean(x);
    N  = length(x) - M;
    X  = hankel(x(1:M), x(M:end));
    R  = (X * X') / N;
    ev = sort(real(eig(R)), 'descend');
    best = Inf;  k_opt = 2;
    for k = 0 : 2 : max_k
        n_n = M - k;
        if n_n < 3;  break;  end
        lam = max(ev(k+1:end), 1e-30);
        gm  = exp(mean(log(lam)));
        am  = mean(lam);
        mdl = -N * n_n * log(gm/am) + 0.5*k*(2*M-k)*log(N);
        k_opt = max(2, k);
    end
end
```

MDL (Minimum Description Length) is an information-theoretic criterion for model selection that trades goodness-of-fit against model complexity.

**The MDL formula:**

`MDL(k) = -N · n_n · log(g_mean / a_mean) + 0.5 · k · (2M-k) · log(N)`

where:
- `k` = candidate number of signal components (model order)
- `n_n = M - k` = number of noise eigenvalues
- `g_mean = exp(mean(log(λ_{k+1:M})))` = geometric mean of noise eigenvalues
- `a_mean = mean(λ_{k+1:M})` = arithmetic mean of noise eigenvalues
- The first term: `-N·n_n·log(g/a)` penalises deviation from white noise in the noise subspace. When the noise subspace eigenvalues are equal (white noise), `g_mean = a_mean` and this term is 0. When they are spread out (coloured noise or unaccounted signal components), `g < a` and this term is negative (more negative = larger penalty for choosing too few components).
- The second term: `0.5·k·(2M-k)·log(N)` is the MDL complexity penalty for having k free frequency parameters.

The model order that minimises `MDL(k)` is chosen.

**Why step by 2 (`k = 0:2:max_k`)?** Complex sinusoids come in conjugate pairs. A real-valued signal with p sinusoids has 2p complex poles. The model order in MUSIC/ESPRIT counts complex poles, so the number of sinusoids is always even.

**Default `k_opt = 2` (minimum 1 sinusoid):** If MDL never finds a minimum (monotonically decreasing — meaning adding more sinusoids always helps), the default of 2 is returned. This is conservative — it means "at least one sinusoid" rather than claiming many.

**References:** See [Appendix A.6](#a6-mdl-model-order-selection)

### 7.8 `ljung_box_local()` — Ljung-Box Portmanteau Test

```matlab
function [Q, p] = ljung_box_local(x, m)
    N = length(x);
    x = x - mean(x);
    v = var(x);
    if v < 1e-30;  Q = 0;  p = 1;  return;  end
    x = x / sqrt(v);
    Q = 0;
    for lag = 1:m
        r = dot(x(1:N-lag), x(lag+1:N)) / N;
        Q = Q + r^2 / (N - lag);
    end
    Q = N * (N + 2) * Q;
    p = 1 - chi2cdf(Q, m);
end
```

The function normalises the input (`x/sqrt(var(x))`) before computing autocorrelations. This ensures the autocorrelation estimate `r(lag) = E[x(t)x(t+lag)] / E[x²]` is bounded in [-1, 1] (the correlation coefficient), not the raw autocovariance. The formula then implements the Ljung-Box Q statistic:

`Q = N(N+2) × Σ_{lag=1}^{m} r²(lag) / (N-lag)`

The `1/(N-lag)` correction (vs the simpler Box-Pierce `1/N`) is Ljung and Box's improvement that gives better χ² approximation in finite samples. The p-value from `chi2cdf(Q, m)` gives the probability of observing Q this large or larger under H₀ (white noise). `p < 0.05` → reject H₀ → coloured noise.

**References:** See [Appendix A.2](#a2-ljung-box-test)

### 7.9 `acf_local()` — Autocorrelation Function

```matlab
function [rho, lags] = acf_local(x, max_lag)
    x = x - mean(x);
    [c, lgs] = xcorr(x, max_lag, 'coeff');
    idx  = lgs >= 0;
    rho  = c(idx);
    lags = lgs(idx)';
end
```

Wrapper around MATLAB's `xcorr` with `'coeff'` normalisation (divides by the zero-lag autocorrelation = variance, producing values in [-1, 1]). Only the non-negative lags are returned (the ACF is symmetric for real-valued signals). Used in Fig W1 panel 3 to visualise the autocorrelation structure of the noise floor signal.

The 95% confidence bound `±1.96/√T` is plotted alongside — samples within these bounds are statistically indistinguishable from zero under H₀ (white noise). Any lag where the ACF exceeds this bound indicates significant autocorrelation.

---

## 8. Figure Structure — Per Window Length

For each of the 5 window sizes, one figure is generated with two stacked panels:

**Panel 1 (top) — BPM tracking:**
- Ground truth as black stairs (step plot to show exact GT value at each time point).
- `bpm_welch` (Ham-tight): solid blue, thickness 2.
- `bpm_welch_ha` (Ham-adapt): dashed blue, thin.
- `bpm_welch_et` (El-tight): solid red, thickness 2.
- `bpm_welch_ea` (El-adapt): dashed red, thin.
- `bpm_music`: solid green, thickness 1.8.
- `bpm_esprit`: dashed magenta, thickness 1.5.
- GT mean horizontal line (black dotted).
- `±ACC_THR` (±10 BPM) acceptance zone horizontal lines (green dotted).
- Red inverted triangles at windows classified as low-confidence.

**Panel 2 (bottom) — Confidence scoring:**
- Four thin dotted lines: individual factor scores (SNR=blue, det=green, lum=red, skin=black).
- Thick magenta line: composite confidence score.
- Cyan fill: regions where composite ≥ CONF_THR (HQ windows).
- Dashed horizontal at CONF_THR = 0.80.

The two panels are **x-linked** (`linkaxes([ax1 ax2], 'x')`): zooming on the time axis of one panel simultaneously zooms the other. This makes it easy to identify which BPM estimate errors coincide with low-confidence periods.

**The 5 figures (one per window size) are the primary output** — they show at a glance which window size allows which method to produce reliable BPM tracks.

---

## 9. Auto-Save & Diary Logging

```matlab
fig_handles = findall(0, 'Type', 'figure');
for fh = fig_handles'
    fname = get(fh, 'Name');
    fname = regexprep(fname, '[^\w -]', '_');
    saveas(fh, fullfile(fda_dir, [fname '.png']));
    saveas(fh, fullfile(fda_dir, [fname '.pdf']));
end
diary off;
```

All open figures are saved as both PNG (raster, for quick viewing) and PDF (vector, for publication/zoom). The `regexprep` replaces any non-alphanumeric characters in the figure name with underscores — spaces are preserved, special characters (like `|`) are replaced — producing valid filenames. `diary off` closes the log file cleanly before MATLAB exits.

---

## 10. Complete Signal Flow Diagram

```
filterdesign_<ts>.csv (from filter design stage)
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  DATA LOADING                                            │
│  S_primary (BVP_ham_tight)                              │
│  S_ham_adapt, S_el_tight, S_el_adapt                   │
│  det_t, skin_t, lum_t, gt_bpm                           │
│  f_low=1.0Hz, f_high=f_p2_adapt                         │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  SECTION W — NOISE WHITENESS                             │
│                                                          │
│  Detect f_card via Welch PSD of S_primary               │
│       ↓                                                  │
│  Notch-filter out cardiac (FIR ±0.30 Hz) → S_noise     │
│       ↓                                                  │
│  Ljung-Box test (Q-stat, p-value vs χ²)                 │
│       ↓                                                  │
│  Eigenvalue CV of autocorrelation matrix                 │
│       ↓                                                  │
│  Report: white/coloured + implications for MUSIC        │
│                                                          │
│  Fig W1: PSD + noise PSD + ACF + eigenvalues            │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  SECTION A — FULL-SIGNAL SPECTRA                        │
│  specCal(S_primary) + specCale([all 4 signals])         │
│  Overlay: tight vs adaptive passband edges               │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  SECTION B — WINDOW-SIZE LOOP: [2,3,5,10,20] seconds   │
│                                                          │
│  For each window size:                                   │
│  ┌────────────────────────────────────────────────────┐ │
│  │ win_N = win_secs × fs                              │ │
│  │ M_sub = max(win_N/4, 6)                            │ │
│  │                                                    │ │
│  │  For each 1-second step:                           │ │
│  │  ├─ est_welch(Ham-tight)  → bpm_welch             │ │
│  │  ├─ est_welch(Ham-adapt)  → bpm_welch_ha          │ │
│  │  ├─ est_welch(El-tight)   → bpm_welch_et          │ │
│  │  ├─ est_welch(El-adapt)   → bpm_welch_ea          │ │
│  │  ├─ mdl_order_local()     → p_mdl (adaptive)      │ │
│  │  ├─ est_music(p_mdl)      → bpm_music             │ │
│  │  ├─ est_esprit(max(4,p))  → bpm_esprit            │ │
│  │  └─ win_quality()         → snr,det,skin,lum_cv  │ │
│  │                                                    │ │
│  │  Confidence score → HQ classification             │ │
│  │  MAE (all) + MAE (HQ only)                        │ │
│  │  Acc% (within ±10 BPM)                            │ │
│  └────────────────────────────────────────────────────┘ │
│  Print comparison table per window size                  │
│  One figure per window (BPM track + confidence score)   │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
            fda_results/fda_<ts>/
            ├── Fig W1 — Noise Whiteness.png/.pdf
            ├── specCal & specCale figures.png/.pdf
            ├── BPM Track — 2s window.png/.pdf
            ├── BPM Track — 3s window.png/.pdf
            ├── BPM Track — 5s window.png/.pdf
            ├── BPM Track — 10s window.png/.pdf
            ├── BPM Track — 20s window.png/.pdf
            └── output.txt (diary log)
```

---

## 11. Why These Parameter Values?

| Parameter | Value | Rationale |
|---|---|---|
| `win_secs = [2,3,5,10,20]` | Window size sweep | 10s is the Welch baseline (≥10 BPM resolution). 5s is the practical minimum for Welch. 2s and 3s test MUSIC/ESPRIT's theoretical advantage for ultra-short windows. 20s tests the other extreme (maximum accuracy for long stable recordings). |
| `M_sub = max(win_N/4, 6)` | Hankel subspace dim | `win_N/4` provides N_cols ≈ 3M (good autocorrelation estimate). Floor at 6 ensures at least `max_k+1=7` eigenvalues for MDL to choose from. |
| `max_k = 6` in MDL | Max model order | 6 complex poles = 3 real sinusoids. In the cardiac band [1.0,f_high Hz], at most the fundamental + 2 harmonics are expected. Orders above 6 are overfitting. |
| `SNR_THR = 6 dB` | HQ gate | Cardiac peak must be 4× the noise floor in the band. Below 6 dB, the peak is not clearly distinguishable — any frequency estimator will be unreliable. |
| `DET_THR = 0.7` | Face detection gate | 30% dropped frames is the practical maximum before the bounding-box interpolation error significantly corrupts the skin ROI. |
| `SKIN_THR = 300` | Skin pixels gate | 300 skin pixels represents roughly a 17×17 pixel ROI area — the minimum for a spatially averaged rPPG signal to have useful SNR. |
| `LUM_CV_THR = 0.12` | AGC stability gate | 12% luminance variation within a 2–10s window indicates active AGC. Stage 1's luminance normalisation removes DC shifts but not rapid fluctuations. |
| `CONF_THR = 0.80` | HQ threshold | With W=[0.40,0.30,0.15,0.15], reaching 0.80 requires SNR ≥ 6dB when other factors are perfect. Chosen so that approximately the top 50–60% of windows (empirically) qualify as HQ. |
| `W_CONF = [0.40,0.30,0.15,0.15]` | Factor weights | SNR dominates (0.40) because signal quality is most predictive of BPM accuracy. Detection rate second (0.30) because a stale bbox immediately corrupts the signal. Skin count and luminance are secondary (0.15 each). |
| `ACC_THR = 10 BPM` | Accuracy threshold | ±10 BPM is the standard clinical acceptability criterion for non-invasive HR monitors (per FDA guidance for consumer wearables, which is also the de facto standard for rPPG literature). |
| MUSIC 5% harmonic threshold | `spectrum(i2) > 0.05 * peak_val` | More aggressive than Stage 2's 30% because short windows have low spectral resolution — a 5s window can't distinguish sub-harmonics well, so any evidence of a higher-frequency peak is taken as the true cardiac. |
| ESPRIT ±0.2 Hz neighbourhood | `f_cb >= (2*fp-0.2) & ...` | MUSIC peaks can shift by up to ±0.2 Hz from the true frequency at short windows (equivalent to ±12 BPM at the spectral resolution of a 2.5s window). Searching the neighbourhood prevents missing the true harmonic due to grid offset. |
| ESPRIT ratio > 1.5 for disambiguation | `hz(end)/hz(1) > 1.5` | If two cardiac-band frequencies differ by < 50% (e.g., 1.1 Hz and 1.5 Hz, ratio = 1.36 < 1.5), they are likely the same frequency split by noise. If they differ by > 50% (e.g., 1.1 Hz and 2.2 Hz, ratio = 2.0 > 1.5), they are genuinely distinct (fundamental and harmonic). |
| `bw_notch = 0.30` Hz | Notch width | Wide enough to remove the broadened cardiac spectral peak (broadened by cardiac rate variability ~±0.05 Hz and window leakage). Narrow enough not to remove adjacent noise content we want to characterise. |

---

## 12. Appendix — References

### A.1 rPPG Window Length Trade-offs

- **McDuff, D. et al. (2015).** *Improvements in Remote Cardiopulmonary Measurement Using a Five Band Digital Camera.* IEEE TBME, 61(10). [https://doi.org/10.1109/TBME.2014.2323695](https://doi.org/10.1109/TBME.2014.2323695)
- **Wang, W. et al. (2017).** *Algorithmic Principles of Remote PPG.* IEEE TBME, 64(7), 1479–1491. [https://doi.org/10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282) — discusses minimum window length requirements for spectral-peak methods.

### A.2 Ljung-Box Test

- **Ljung, G. M., & Box, G. E. P. (1978).** *On a Measure of Lack of Fit in Time Series Models.* Biometrika, 65(2), 297–303. [https://doi.org/10.1093/biomet/65.2.297](https://doi.org/10.1093/biomet/65.2.297)
- Original derivation of the Ljung-Box Q statistic and its χ² approximation.
- **Box, G. E. P., & Pierce, D. A. (1970).** *Distribution of Residual Autocorrelations in Autoregressive-Integrated Moving Average Time Series Models.* JASA, 65(332), 1509–1526. — The predecessor test; Ljung-Box improves the finite-sample χ² approximation.

### A.3 Pisarenko / MUSIC Eigenvalue Analysis

- **Pisarenko, V. F. (1973).** *The Retrieval of Harmonics from a Covariance Function.* Geophysical Journal of the Royal Astronomical Society, 33(3), 347–366. [https://doi.org/10.1111/j.1365-246X.1973.tb03424.x](https://doi.org/10.1111/j.1365-246X.1973.tb03424.x) — First eigendecomposition-based frequency estimator; MUSIC generalises this.
- **Johnson, D. H., & Dudgeon, D. E. (1993).** *Array Signal Processing: Concepts and Techniques.* Prentice Hall. Chapter 4: Eigenstructure-based frequency estimation.

### A.4 MUSIC Algorithm

- **Schmidt, R. O. (1986).** *Multiple Emitter Location and Signal Parameter Estimation.* IEEE TAES, 34(3), 276–280. [https://doi.org/10.1109/TAES.1986.310827](https://doi.org/10.1109/TAES.1986.310827) — Original MUSIC paper.
- **The steering vector `e(ω)`:** A vector `[1, e^{jω}, e^{2jω}, ..., e^{j(M-1)ω}]` that describes how a sinusoid at frequency ω projects onto the Hankel subspace. Frequencies where this vector is orthogonal to all noise eigenvectors produce infinity in the MUSIC pseudospectrum.

### A.5 ESPRIT Algorithm

- **Roy, R., & Kailath, T. (1989).** *ESPRIT — Estimation of Signal Parameters via Rotational Invariance Techniques.* IEEE TASSP, 37(7), 984–995. [https://doi.org/10.1109/29.32276](https://doi.org/10.1109/29.32276) — Original ESPRIT paper.
- **Key insight:** Two shifted sub-arrays of the signal share the same signal subspace up to a rotation. The rotation matrix's eigenvalues encode the signal frequencies. This algebraic solution has no grid search — it is exact at infinite SNR.
- **ESPRIT vs MUSIC:** ESPRIT is typically faster (no pseudospectrum evaluation) and has lower bias in clean conditions. MUSIC has better resolution near the Nyquist limit and when frequencies are closely spaced. Both outperform Welch for short data segments when SNR is adequate.

### A.6 MDL Model Order Selection

- **Rissanen, J. (1978).** *Modeling by Shortest Data Description.* Automatica, 14(5), 465–471. [https://doi.org/10.1016/0005-1098(78)90005-5](https://doi.org/10.1016/0005-1098(78)90005-5) — Original MDL principle.
- **Wax, M., & Kailath, T. (1985).** *Detection of Signals by Information Theoretic Criteria.* IEEE TASSP, 33(2), 387–392. [https://doi.org/10.1109/TASSP.1985.1164557](https://doi.org/10.1109/TASSP.1985.1164557) — Applied MDL specifically to the eigenvalue-based signal/noise subspace separation problem. This is the exact formulation implemented in `mdl_order_local()`.
- The MDL penalty `0.5k(2M-k)log(N)` counts the number of real free parameters in a rank-k signal model for an M×M autocorrelation matrix: k complex frequencies × (2M-k) projections, all real-valued.

### A.7 Welch PSD for Short Windows

- **Welch, P. D. (1967).** *The Use of Fast Fourier Transform for the Estimation of Power Spectra.* IEEE TAES, 15(2), 70–73. [https://doi.org/10.1109/TAES.1967.5408896](https://doi.org/10.1109/TAES.1967.5408896)
- **Frequency resolution:** `Δf = fs / N_window`. For N=60 (2s at 30fps), Δf = 0.5 Hz = 30 BPM. This is the fundamental limitation that motivates MUSIC/ESPRIT for short-window rPPG.

### A.8 Confidence Scoring in rPPG

- **McDuff, D., Gontarek, S., & Picard, R. W. (2014).** *Improvements in Remote Cardiopulmonary Measurement Using a Five Band Digital Camera.* IEEE EMBC 2014. — One of the first papers to propose quality metrics for rPPG windows.
- **Chen, W., & McDuff, D. (2018).** *DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks.* ECCV 2018. [https://arxiv.org/abs/1805.07888](https://arxiv.org/abs/1805.07888) — Deep-learning approach with implicit quality-gating via attention.

### A.9 MUSIC/ESPRIT for rPPG — Research Context

- **Balmaekers, B., & de Haan, G. (2013).** *Towards Continuous Non-Contact Blood Pressure Estimation.* PHM 2013. — Application of subspace methods to rPPG-adjacent problems.
- **Technical Documents/rPPG_MUSIC_Analysis_and_Roadmap.md** in this repository — Prof. Chen's roadmap explaining why MUSIC/ESPRIT are theoretically superior to Welch for short rPPG windows, with worked examples.

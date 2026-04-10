# Frequency Domain Analysis — Run 1
**Date:** 2026-04-05  
**Script:** `bpm_controls_FDA.m`  
**Filter:** Butterworth order=4, band=[0.67, 3.0] Hz  
**Windows compared:** 5 s, 10 s, 21 s (full signal)  
**Methods:** FFT, Welch, STFT

---

## Console Output

```
Pipeline ready. T=630 frames, fs=29.9976 Hz

Window (s)    FFT BPM     Welch BPM   FFT err     Welch err
------------------------------------------------------------
5             73.8        62.4        +18.9       +7.5
10            54.0        54.9        -0.9         0.0
21            54.9        54.9         0.0         0.0
Full          54.9        54.9         0.0         0.0    (reference)
```

**Reference BPM: 54.9** (full-signal Welch with nfft=4096 zero-padding).  
Note: previous filter design runs reported 56.2 BPM. The difference is because the filter design script used `nfft=[]` (default, coarser grid) while this script uses `nfft=4096` (finer interpolated grid). The true peak sits between bins when using a coarser grid — zero-padding to 4096 reveals the interpolated peak at 54.9 BPM more accurately.

---

## Core Concept Before Reading Any Figure

### What "bin resolution" means — and why it is the fundamental limit

The FFT divides the frequency axis into discrete bins of width:

```
bin width = fs / N   Hz
```

where `N` is the number of samples in the window and `fs = 29.9976 Hz`.

| Window | N (samples) | Bin width (Hz) | Bin width (BPM) |
|---|---|---|---|
| 5 s | 150 | 0.200 Hz | **12.0 BPM/bin** |
| 10 s | 300 | 0.100 Hz | **6.0 BPM/bin** |
| 21 s | 630 | 0.048 Hz | **2.9 BPM/bin** |

**This is a hard physical limit — not a software choice.** You cannot know frequency more precisely than 1/T Hz from T seconds of data. This is the Fourier uncertainty principle: time and frequency cannot both be measured with arbitrary precision simultaneously.

At 5s: The bins near the true cardiac peak (0.915 Hz = 54.9 BPM) are:
- 0.8 Hz (48 BPM)
- 1.0 Hz (60 BPM)
- 1.2 Hz (72 BPM)

The true peak is between the 0.8 Hz and 1.0 Hz bins. Whether the FFT reports 48, 60, or even 72 BPM depends on which bin has the most energy — which in turn depends on noise and spectral leakage. The answer is essentially unreliable.

### Zero-padding ≠ higher resolution
The code uses `nfft=4096` with a 5s window (150 samples). This fills the remaining 3946 points with zeros before computing the FFT. The resulting frequency grid has 4096/2 = 2048 points — the curve looks smooth. **But the frequency resolution is still 0.200 Hz.** Zero-padding only interpolates between the existing bins. It does not let you see features narrower than one true bin.

---

## Figure 1 — FFT at Different Window Lengths

### What the axes mean
- **X-axis:** Frequency in Hz, zoomed to 0.5–3.5 Hz (the cardiac band and just outside)
- **Y-axis:** Magnitude in dB, normalized to 0 dB at the peak of the entire spectrum. 0 dB = the strongest frequency component in this window. −10 dB = 3× weaker. −20 dB = 10× weaker.
- **Dashed vertical lines:** Left = 0.67 Hz (lower cutoff), right = 3.0 Hz (upper cutoff)
- **Red vertical line:** The detected peak (highest point in the cardiac band)

---

### Subplot 1 — FFT, window=5s (bin=0.200 Hz=12 BPM/bin)

**What you see:**  
The spectrum has NO clearly dominant single peak. Instead there is a broad, irregular hump spanning from about 0.7 Hz to 2.5 Hz. The detected "peak" at 73.8 BPM (1.23 Hz) is only marginally higher than the surrounding spectrum — it is not a true isolated cardiac peak, it is simply where a broad noisy hump happens to be tallest.

There is a pronounced **dip near 0.9 Hz** — close to where the true cardiac signal actually lives. This is paradoxical but real: spectral leakage and noise at adjacent frequencies have constructed a hump that overwhelms the true cardiac component.

**Why this happens — two combined problems:**

**Problem 1 — Resolution:** At 12 BPM/bin, the true cardiac signal at 0.915 Hz spreads its energy across 2–3 adjacent bins (this is called spectral leakage — even with a Hann window, energy leaks into neighbouring bins). Noise at nearby frequencies also smears into the same bins. The result is a broad, flat region where no single bin stands out as the clear cardiac peak.

**Problem 2 — The signal itself was different in the first 5 seconds:** This is revealed by the STFT (Figure 3). The first 5 seconds of the recording captured a period when the subject's heart rate was genuinely higher (around 1.1–1.3 Hz = 66–78 BPM). The cardiac signal was NOT at 54.9 BPM during those 5 seconds — it settled to that frequency later. So the FFT at 5s is reporting a partially correct answer for the wrong reason: the first 5 seconds of signal genuinely contained more energy at higher frequencies.

**Conclusion:** At 5s, FFT gives 73.8 BPM — off by +18.9 BPM. Completely outside the ±5 BPM threshold. Unusable.

---

### Subplot 2 — FFT, window=10s (bin=0.100 Hz=6 BPM/bin)

**What you see:**  
A clearer shape. The dominant feature is a broad peak at the left side of the band, peaking around 0.9 Hz, with the detected peak at 54.0 BPM. The peak is visible as the highest point but is still broad — the bins at 0.9 Hz and 1.0 Hz are close in height.

Secondary features are visible at approximately 1.5, 2.0, and 2.3 Hz — these are noise components that the Butterworth filter has not completely suppressed (they are inside the passband).

**Why this works:** With 6 BPM/bin, the cardiac peak at 0.915 Hz now sits in a bin that is meaningfully taller than its neighbours. The −0.9 BPM error means the nearest bin to the true 0.915 Hz peak is the 0.9 Hz bin (54.0 BPM), which is correct to within one bin.

**Conclusion:** At 10s, FFT gives 54.0 BPM — error of −0.9 BPM. Within the ±5 BPM threshold. Reliable.

---

### Subplot 3 — FFT, window=21s (bin=0.048 Hz=2.9 BPM/bin)

**What you see:**  
A highly detailed spectrum with many narrow peaks and notches. The dominant peak at 54.9 BPM (0.915 Hz) is clearly the tallest feature in the cardiac band. Many smaller secondary peaks are visible throughout — at 1.0, 1.1, 1.4, 1.5, 1.8, 2.0, 2.3 Hz and elsewhere. Deep notches (drops to −30 to −40 dB) are scattered between peaks.

**Why so many features?** With 2.9 BPM/bin, the spectrum is fine enough to resolve individual components. The secondary peaks are:
1. **Harmonics of the cardiac signal:** The heartbeat waveform is not a perfect sine wave. It has harmonics at integer multiples of the fundamental. 0.915 Hz × 2 = 1.83 Hz, × 3 = 2.75 Hz — look for peaks near those frequencies.
2. **Noise components that survived the filter:** The Butterworth passband is [0.67, 3.0] Hz. Any noise in that band is passed through.
3. **Respiration:** Breathing frequency is typically 0.2–0.4 Hz, but its harmonics can appear inside the cardiac band.

The notches (deep dips) are from destructive interference between the signal and noise at those specific frequencies.

**Conclusion:** At 21s, FFT gives 54.9 BPM — exact match to reference. Zero error. Best possible result from Fourier analysis on this data.

---

## Figure 2 — Welch at Different Window Lengths

### What Welch does differently from FFT

Welch splits the signal into overlapping segments, computes the FFT of each segment, and **averages the power spectra**. This averaging does two things:
- Reduces the variance (random fluctuations) in the spectral estimate — the curve becomes smoother
- But: the frequency resolution of each averaged spectrum is still limited by the segment length, not the total signal length

In other words: Welch trades frequency resolution for statistical smoothness.

### Subplot 1 — Welch, nperseg=5s (bin=0.200 Hz=12 BPM/bin)

**What you see compared to FFT at 5s:**  
The Welch curve is SMOOTHER — the random jaggedness is gone, replaced by broader, rounder humps. The peak is detected at 62.4 BPM (1.04 Hz) — closer to truth than FFT's 73.8 BPM.

**Why Welch is better than FFT at 5s:** Using a 5s segment over a 21s signal gives approximately 4 non-overlapping segments to average. Averaging 4 spectra suppresses random noise peaks. The result: the noise hump that fooled the FFT into detecting 73.8 BPM is smoothed out, and the dominant feature moves closer to the true 0.915 Hz peak.

**Why Welch still fails at 5s:** Averaging suppresses random noise but cannot create frequency resolution that the segment length doesn't allow. The peak at 62.4 BPM (1.04 Hz) is still wrong by +7.5 BPM. The true 0.915 Hz peak and the 1.04 Hz region are within one bin of each other — with 0.2 Hz bins, they cannot be separated.

**Conclusion:** Welch at 5s gives 62.4 BPM — off by +7.5 BPM. Still outside ±5 BPM threshold. Unusable, but better than FFT's 18.9 BPM error.

---

### Subplot 2 — Welch, nperseg=10s (bin=0.100 Hz=6 BPM/bin)

**What you see:**  
A clearly dominant peak at the left side of the cardiac band, detected at 54.9 BPM. The curve is smoother than FFT at 10s. Two secondary bumps are visible: one near 1.0 Hz and another around 1.5 Hz. The dominant peak stands clearly above the rest.

**Why Welch is more reliable than FFT here:** At 10s, you get 2 overlapping segments. The averaging smooths out the competition between the 0.9 Hz and 1.0 Hz adjacent bins — the 0.9 Hz bin consistently wins across segments, confirming it as the true dominant frequency.

**Conclusion:** Welch at 10s gives 54.9 BPM — zero error. Perfect match. **10 seconds is the minimum window for reliable Welch-based BPM estimation on this signal.**

---

### Subplot 3 — Welch, nperseg=21s (bin=0.048 Hz=2.9 BPM/bin)

**What you see:**  
Similar detail level to FFT at 21s, but with some differences. The curve shows multiple narrow peaks and very deep notches (dropping to −80 dB). The dominant peak at 54.9 BPM is clear and stands about 15–20 dB above the next features.

**Why some notches are deeper in Welch than FFT at the same length:** With only 1.5 segments at this nperseg, the averaging is minimal. In some frequency regions, the small number of segments allows destructive interference to create deeper notches than the single FFT would show.

**Conclusion:** Welch at 21s gives 54.9 BPM — exact match. The finest resolution available from this signal.

---

## Figure 3 — STFT Time-Frequency Heatmap

This is the most informative figure in the entire analysis. It reveals something that FFT and Welch completely miss.

### What the axes mean
- **X-axis:** Time in seconds (4–16 s). The STFT starts at t=4s and ends at t=16s because the 8s sliding window needs 8s of signal to form the first estimate (centered at t=4s with 90% overlap). This means the STFT cannot see the first or last 4 seconds of the recording.
- **Y-axis:** Frequency in Hz (0.67–3.0 Hz — the cardiac band only)
- **Color:** Power in dB/Hz at each (time, frequency) point. **Red = high power, Blue/Dark = low power.** The color scale runs approximately from −45 dB/Hz (red) to −65 dB/Hz (dark blue).
- **White dashed line:** The dominant frequency at each time step — the instantaneous BPM estimate produced by the STFT

### What you see

**The red region (high power) is concentrated in the lower half of the band, roughly 0.8–1.4 Hz.** Above 1.5 Hz, the colors shift to blue/dark — these frequencies have much less power.

**The white dashed line tells the full story:**
- t = 4–5 s: White line is at ~1.2 Hz (72 BPM) — **high heart rate early in the recording**
- t = 5–8 s: White line drops to ~0.9 Hz (54 BPM) — heart rate settling
- t = 8–10 s: White line at ~1.0 Hz (60 BPM) — brief elevation
- t = 10–15 s: White line hovers around 0.9–1.0 Hz (54–60 BPM) — most stable period
- t = 15–16 s: White line rises again to ~1.3–1.4 Hz (78–84 BPM) — elevated at the end

### The most important finding from the STFT

**The heart rate was NOT constant during the 21-second recording.**

It started high (~72 BPM), settled to ~54 BPM in the middle, then rose again at the end. This is real physiological behaviour — the subject was likely adjusting their posture, breathing changed, or movement artifact caused an early spike.

**This directly explains the 5s FFT error of +18.9 BPM.** The code takes `S_filt(1:N)` for the 5s window = the first 5 seconds. The STFT shows that the first 5 seconds had the cardiac signal genuinely at ~1.1–1.3 Hz, not at 0.915 Hz. The FFT was not simply confused by noise — it was analyzing a portion of the signal where the heart rate was actually different. The 73.8 BPM result is therefore partially physically correct for those 5 seconds.

**This reveals the second fundamental problem with static FFT/Welch:**

FFT and Welch produce a SINGLE estimate for the entire window — they assume the signal is **stationary** (constant frequency) throughout. But the heart rate drifts. If you use a 10s or 21s window, you get the time-averaged BPM, which loses the moment-to-moment variation. If you use a short window to capture the variation, your frequency resolution is too coarse.

This is the Fourier trade-off: **time resolution vs frequency resolution.** You cannot have both.

STFT is a partial solution — it uses a sliding window (8s here) to produce a time-varying estimate. But the 8s window has bin resolution of 0.125 Hz = 7.5 BPM, which is still fairly coarse. And 8s is still a long window for real-time tracking.

### STFT parameters and their effects
| Parameter | Value used | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `stft_win_sec` | 8 s | Better freq resolution, worse time resolution | Worse freq resolution, better time resolution |
| `stft_ovlp_frac` | 90% | Smoother time track (more time points) | Fewer time points, more discontinuous track |

### Key observation: the horizontal dark bands
At certain frequencies (approximately 1.5, 1.8, 2.1, 2.4 Hz), there are horizontal dark bands — lower power at those frequencies across all time. These are NOT related to heart rate. They are likely:
1. Noise sources that are consistently weak in those specific frequency ranges
2. The harmonic structure of the signal creating notches between harmonics

---

## Figure 4 — BPM Accuracy vs Window Length

This is the summary figure and the core motivation for MUSIC.

### What the axes mean
- **X-axis:** Window length in seconds (5, 10, 21)
- **Y-axis:** Estimated BPM by each method
- **Black dashed line:** Reference BPM = 54.9 (full-signal Welch, most accurate available)
- **Black dotted lines:** ±5 BPM error threshold — outside this band, the estimate is clinically unreliable

### What you see

**At window = 5s:**
- FFT (blue circle): 73.8 BPM — massively above the +5 BPM threshold line. Error = +18.9 BPM.
- Welch (red square): 62.4 BPM — also above the +5 BPM threshold. Error = +7.5 BPM.
- Both methods fail. Neither is within the acceptable zone.

**At window = 10s:**
- FFT: 54.0 BPM — just below the reference, within the ±5 BPM band. Error = −0.9 BPM.
- Welch: 54.9 BPM — exactly on the reference. Error = 0.0 BPM.
- Both methods pass at 10s.

**At window = 21s:**
- Both: 54.9 BPM — exact match. Error = 0.0 BPM.

### The cliff edge between 5s and 10s

The transition from 10s → 5s is not gradual — it is a cliff. FFT drops from −0.9 BPM error to +18.9 BPM error. Welch drops from 0.0 BPM error to +7.5 BPM error. This cliff is caused by the combination of:
1. Coarser bins at 5s: 12 BPM/bin vs 6 BPM/bin at 10s — the bin containing the true peak changes
2. Time non-stationarity: the first 5s of signal genuinely had different heart rate (STFT revealed this)

---

## Key Findings

### Finding 1: Minimum window for reliable BPM is 10 seconds for Welch, 10 seconds for FFT
Both fail at 5s. Both succeed at 10s. The threshold is between 5 and 10 seconds.

### Finding 2: Welch is more robust than FFT at short windows
At 5s, Welch error is 7.5 BPM vs FFT error of 18.9 BPM. Averaging helps. But neither is within ±5 BPM.

### Finding 3: Heart rate was non-stationary during the recording
The STFT revealed that the first 5 seconds had genuinely higher heart rate (~72 BPM). This is normal physiological behaviour. It means:
- A static 21s FFT/Welch gives the time-averaged BPM — it misses the real-time variation
- A 5s window gets the instantaneous estimate but with terrible frequency resolution
- There is NO window length that gives both fine frequency resolution AND fine time resolution simultaneously using Fourier methods

### Finding 4: The 10s window requirement is impractical for real-time use
In the ROS2 DocBot live stream, you want BPM updates every ~2 seconds. A 10s minimum window means 8 seconds of latency (you must collect 10s before you can estimate). This is unacceptable for closed-loop robot repositioning.

### Finding 5: MUSIC is motivated by exactly this gap
MUSIC is a superresolution method. Its frequency resolution is NOT limited by 1/T. By exploiting the structure of the signal (one cardiac sinusoid = K=1), MUSIC can in principle detect the cardiac frequency from a 2–3 second window with accuracy comparable to what Welch needs 10+ seconds to achieve. This is what Prof. Chen's Section 3 is about.

---

## Summary Table

| Method | Window | BPM | Error | Resolution | Verdict |
|---|---|---|---|---|---|
| FFT | 5 s | 73.8 | +18.9 | 12 BPM/bin | FAIL |
| Welch | 5 s | 62.4 | +7.5 | 12 BPM/bin | FAIL |
| FFT | 10 s | 54.0 | −0.9 | 6 BPM/bin | PASS |
| Welch | 10 s | 54.9 | 0.0 | 6 BPM/bin | PASS |
| FFT | 21 s | 54.9 | 0.0 | 2.9 BPM/bin | PASS |
| Welch | 21 s | 54.9 | 0.0 | 2.9 BPM/bin | PASS |
| STFT | 8 s sliding | Time-varying | — | 7.5 BPM/bin | Informational only |

---

## Next Step: MUSIC (Section 3)

The frequency domain analysis section is complete. The gap it identified:

> FFT and Welch need ≥10 seconds for <5 BPM error. The target for a real-time system is 2–3 second windows. MUSIC must bridge this gap.

MUSIC implementation plan:
1. Build covariance matrix R from M-sample Hankel embedding of S_filt
2. Eigendecompose R → signal subspace (K=1 eigenvector) + noise subspace (M-K eigenvectors)
3. Compute MUSIC pseudo-spectrum: P(ω) = 1 / ||Qn^H · a(ω)||²
4. Find peak of P(ω) in cardiac band → BPM
5. Compare against FFT/Welch on same short window to show superresolution

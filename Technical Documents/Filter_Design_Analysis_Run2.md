# IIR Filter Design — Analysis Run 2
**Date:** 2026-04-05  
**Script:** `bpm_control_new.m`  
**Parameters:** order=4, Rp=0.5 dB, Rs=40 dB, band=[0.67, 3.0] Hz  
**Change from Run 1:** Magnitude response plot fixed — BEFORE and AFTER comparison now shown

---

## Console Output

```
Filter outputs: std bw=0.003858  c1=0.003895  c2=0.002796  el=0.003839

Filter           Peak Hz     BPM
--------------------------------
Butterworth       0.9374    56.2
Chebyshev I       0.9374    56.2
Chebyshev II      1.1132    66.8
Elliptic          0.9374    56.2
```

Identical to Run 1 — expected, since the fix was visualisation only, not signal processing.

---

## Figure 1a — Magnitude BEFORE Fix

![Magnitude BEFORE](/Users/devanshbajwala/Desktop/Screenshot 2026-04-05 BEFORE.png)

### What you see
- Y-axis stretched from +100 dB down to −800 dB
- The passband (near 0 dB) is compressed into a thin band at the very top of the plot
- Filter zeros appear as sharp narrow dips to −100 to −110 dB in the stopband
- The passband behaviour of any filter is unreadable at this scale

### Why it still looks "less broken" than Run 1 described
In Run 1 the plot may have hit true −Inf values from the default freqz frequency grid landing exactly on filter zeros. Here, `f_full = linspace(0, fs/2, 4096)` spreads 4096 points across 0–15 Hz, so the grid rarely hits a zero exactly — the dips reach "only" −100 dB instead of −∞. But the y-axis is still auto-scaled to −800 dB due to the extreme values present, so the passband is still unreadable.

**This plot is intentionally kept to show the problem. The AFTER plot (Fig 1b) is the usable one.**

---

## Figure 1b — Magnitude AFTER Fix

![Magnitude AFTER](/Users/devanshbajwala/Desktop/Screenshot 2026-04-05 AFTER.png)

### Fix applied
1. `freqz` computed on `f_zoom = linspace(0, 6, 8192)` — zoomed 0–6 Hz only
2. All dB values clipped to `max(20*log10(abs(H)), -80)` — no −Inf values in data
3. `ylim([-80 5])` and `xlim([0 6])` set **after** all `xline` calls

### What you now see clearly

| Filter | Passband (0.67–3.0 Hz) | Transition | Stopband |
|---|---|---|---|
| Butterworth (blue) | Flat at 0 dB, gradual rolloff near edges | Widest transition band of all four | Smooth monotonic rolloff, reaches −25 dB by 6 Hz |
| Chebyshev I (red) | Flat at 0 dB | Sharper than Butterworth | Monotonic rolloff, slightly steeper than Butterworth |
| Chebyshev II (green) | Passband starts lower — rolls off before reaching 0.67 Hz from below (stopband specification mismatch) | Very sharp | Flat stopband floor at exactly −40 dB (= Rs=40 dB) with deep notches dipping well below −80 dB |
| Elliptic (magenta) | Flat at 0 dB in passband | Sharpest transition of all four | Has deep notches just outside both cutoffs, then floor around −40 dB |

### Key observation: Chebyshev II passband starts below 0.67 Hz
This directly explains the 66.8 BPM vs 56.2 BPM discrepancy. The green line in the passband region (between dashed vertical lines) is noticeably lower on the left side — the filter is already attenuating at 0.67 Hz because for Chebyshev II, 0.67 Hz is the **stopband** edge, not the passband edge. Power around 0.94 Hz is partially attenuated, shifting the detected peak to 1.11 Hz.

### Key observation: Elliptic has the sharpest cutoffs
The magenta line drops almost vertically at both 0.67 Hz and 3.0 Hz. This is the defining characteristic of Elliptic filters — they trade off ripple in both passband and stopband for the steepest possible transition.

### Key observation: Butterworth has the most gradual rolloff
The blue line rolls off most slowly after 3.0 Hz. This means more high-frequency noise leaks through vs the other filters, but it also means the most predictable group delay for real-time use.

---

## Figure 2 — Phase Response

![Phase Response](/Users/devanshbajwala/Desktop/Screenshot 2026-04-05 Phase.png)

### What you see
- **Butterworth (blue):** Smooth, continuously declining from 0° to about −650° across 0–6 Hz. No jumps.
- **Chebyshev I (red):** Almost identical to Butterworth, slightly more total phase shift (declining to about −680°). They track together through the passband.
- **Chebyshev II (green):** Has sudden vertical jumps at approximately 0.67 Hz and 3.0 Hz. Between the jumps the curve is smooth. These jumps are unwrap artifacts — they occur exactly at filter zeros where phase is undefined.
- **Elliptic (magenta):** Large swings in the 0–0.5 Hz region (below the passband), then settling, with a jump around 3.0 Hz. The early swings reflect the elliptic filter's poles and zeros clustered near the lower band edge.

### Why jumps appear in Chebyshev II and Elliptic
At a filter zero, the signal magnitude = 0 exactly. Phase at a zero is mathematically undefined. `unwrap()` tries to remove 2π discontinuities but fails here because the phase flip is not a wrapping artifact — it is a true discontinuity at the zero location. Not a code bug. Not a filter malfunction.

### Practical implication
With `filtfilt`, all phase distortion is cancelled for all four filters regardless of this plot. Phase response only matters if switching to real-time (`filter` instead of `filtfilt`).

---

## Figure 3 — Group Delay

*(Not re-shared in Run 2 — identical to Run 1 since filter design parameters unchanged)*

See Run 1 documentation for group delay analysis. Summary:
- Butterworth: smooth, gradual — best for real-time
- Chebyshev I: similar to Butterworth
- Chebyshev II: spikes at ~2 Hz and 3 Hz (~1750 ms)
- Elliptic: sharpest spikes at band edges (>3000 ms)

---

## Figure 4 — Filtered BVP Signals

![Filtered Signals](/Users/devanshbajwala/Desktop/Screenshot 2026-04-05 Signals.png)

### What you see
- **Gray (S_det):** Raw detrended CHROM signal, amplitude ±0.05, very noisy
- **Blue + Red (Butterworth + Chebyshev I):** Perfectly overlapping smooth oscillation
- **Magenta (Elliptic):** Nearly identical to blue/red — the three are indistinguishable in practice
- **Green (Chebyshev II):** Visibly different — different amplitude envelope and oscillation period

### Reading the period
Blue/red/magenta peaks repeat approximately every 1.07 s (= 60/56.2 BPM). Look at 5–15 s where the signal is most stable — you can count roughly 9–10 complete cycles.

Green peaks have a slightly shorter spacing (60/66.8 ≈ 0.9 s), consistent with its higher detected BPM.

---

## Figure 5 — PSD of Filtered Signals (Cardiac Band)

![PSD](/Users/devanshbajwala/Desktop/Screenshot 2026-04-05 PSD.png)

### What you see
- **Blue (Butterworth):** Broad hump, noise floor around −55 to −60 dB/Hz. Peak near 0.75–0.9 Hz.
- **Red (Chebyshev I):** Nearly identical to Butterworth
- **Magenta (Elliptic):** Slightly different shape but same peak location. The elliptic filter's steeper rolloff causes the power to cut more sharply near 0.67 Hz and 3.0 Hz — the hump is slightly narrower than Butterworth.
- **Green (Chebyshev II):** Dramatically different. Power is suppressed to −83 dB near 0.67 Hz, rises through 1–2 Hz, then drops sharply after 2.5 Hz to −110 to −125 dB. Peak at 1.11 Hz = 66.8 BPM.

### What the Elliptic shape tells us
The magenta line is slightly narrower than the blue/red hump. This is the elliptic filter's sharpness working as intended — it rejects more noise right at the band edges. But since all three give 56.2 BPM, the peak detection is unaffected.

---

## Run 2 vs Run 1: What Changed

| Item | Run 1 | Run 2 |
|---|---|---|
| BPM results | Same | Same (expected) |
| Filtered signals | Same | Same (expected) |
| PSD | Same | Same (expected) |
| Fig 1a (BEFORE) | Single broken plot — passband invisible | Now labelled as BEFORE, kept for comparison |
| Fig 1b (AFTER) | Did not exist | New — all four filters readable in [−80, 5] dB window |
| Figure numbering | Figs 1–5 | Figs 1a, 1b, 2, 3, 4, 5 (total 6 figures) |

---

## Confirmed Conclusions After Run 2

1. **Magnitude fix works.** All four filters are now clearly readable in Fig 1b.
2. **Butterworth has the slowest rolloff** — widest transition band, most gradual stopband.
3. **Elliptic has the sharpest cutoffs** — steepest transition, but ripple in both bands.
4. **Chebyshev I is intermediate** — sharper than Butterworth, cleaner stopband than Elliptic.
5. **Chebyshev II is NOT directly comparable** — its stopband edge semantics make the effective passband narrower. Not useful for direct comparison without recalculating band edges.
6. **For offline (filtfilt):** Butterworth, Chebyshev I, and Elliptic are equivalent. All give 56.2 BPM.
7. **For real-time:** Butterworth recommended. See Run 1 group delay analysis.

---

## Next Steps

### Immediate: Parameter exploration
Change one parameter at a time, re-run, observe the change in Fig 1b:

```matlab
% Try these in order:
order = 2;   % then 6, then 8 — observe transition band width
Rp    = 1.0; % then 3.0 — observe passband ripple in Chebyshev I / Elliptic
Rs    = 60;  % then 80  — observe stopband floor depth in Chebyshev II / Elliptic
```

### After filter selection: Section 2 — Frequency Domain Analysis
FFT vs Welch vs STFT at different window lengths.

### After frequency analysis: Section 3 — MUSIC
Subspace-based peak detection, K=1, minimum window study.

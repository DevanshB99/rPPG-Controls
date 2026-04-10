# IIR Filter Design — Analysis Run 3
**Date:** 2026-04-05  
**Script:** `bpm_control_new.m`  
**Change from Run 2:** `order` reduced from **4 → 2**  
**Other parameters unchanged:** Rp=0.5 dB, Rs=40 dB, band=[0.67, 3.0] Hz

---

## Console Output

```
Filter outputs: std bw=0.003809  c1=0.004533  c2=0.001121  el=0.004523

Filter           Peak Hz     BPM
--------------------------------
Butterworth       0.9374    56.2
Chebyshev I       0.8788    52.7
Chebyshev II      1.3475    80.9
Elliptic          0.8788    52.7
```

---

## BPM Comparison: order=4 vs order=2

| Filter | order=4 BPM | order=2 BPM | Change |
|---|---|---|---|
| Butterworth | 56.2 | 56.2 | No change |
| Chebyshev I | 56.2 | **52.7** | −3.5 BPM |
| Chebyshev II | 66.8 | **80.9** | +14.1 BPM |
| Elliptic | 56.2 | **52.7** | −3.5 BPM |

**At order=4:** Butterworth, Chebyshev I, Elliptic agreed at 56.2 BPM.  
**At order=2:** Only Butterworth stays at 56.2. Chebyshev I and Elliptic shift to 52.7. Chebyshev II is completely off at 80.9.

**Conclusion: order=2 is too low for reliable BPM estimation on this signal.**

---

## Std of Filtered Signals Comparison

| Filter | order=4 std | order=2 std | Interpretation |
|---|---|---|---|
| Butterworth | 0.003858 | 0.003809 | Slightly lower — less signal energy passed |
| Chebyshev I | 0.003895 | 0.004533 | Higher — wider effective passband at low order |
| Chebyshev II | 0.002796 | **0.001121** | Much lower — filter is killing the signal |
| Elliptic | 0.003839 | 0.004523 | Higher — more signal through (sharper cutoff holds better) |

---

## Figure 1b — Magnitude Response (order=2)

![Magnitude order=2](/Users/devanshbajwala/Desktop/Screenshot 2026-04-05 Run3 Magnitude.png)

### What changed vs order=4

The most important change is how wide the transition bands become at order=2.

| Filter | What you see at order=2 | What this means |
|---|---|---|
| Butterworth (blue) | Very gentle S-curve. Still at −5 dB at 0.3 Hz (well below the 0.67 Hz cutoff). Rolls off to only −22 dB at 6 Hz. | Extremely gradual rolloff. Significant noise from 3–6 Hz leaks into the output. |
| Chebyshev I (red) | Almost perfectly overlaps Butterworth at order=2. Slightly steeper near 3.0 Hz upper cutoff but barely distinguishable. | At this low order, Chebyshev I and Butterworth are nearly equivalent. |
| Elliptic (magenta) | Sharpest of the four — drops steeply at both 0.67 Hz and 3.0 Hz. Still maintains its characteristic steep transition even at order=2. But stopband is only ~−12 to −15 dB at 6 Hz. | Elliptic still has the sharpest cutoff, but the stopband attenuation is weak at order=2. |
| Chebyshev II (green) | Starts at −40 dB, rises steeply to near 0 dB only around 0.8–1.0 Hz, peaks there, then rolls off again. Deep notch at ~0.4 Hz and ~3.7 Hz. | At order=2, the passband is extremely narrow — only a small window around 0.8–1.4 Hz sees near-0 dB gain. Everything else is attenuated. |

### Critical observation: Butterworth and Chebyshev I transition bands
At order=4, the Butterworth filter was still declining at 6 Hz (~−25 dB). At order=2, it reaches only ~−22 dB at 6 Hz but with a much wider transition — meaning the filter barely starts rejecting noise until it is well past 3.0 Hz. The passband is also only approximately flat; it starts rolling off noticeably before reaching 3.0 Hz.

---

## Figure 3 — Group Delay (order=2)

![Group Delay order=2](/Users/devanshbajwala/Desktop/Screenshot 2026-04-05 Run3 GroupDelay.png)

| Filter | Peak delay (ms) | Location |
|---|---|---|
| Butterworth (blue) | ~530 ms | 0.5 Hz |
| Chebyshev I (red) | ~530 ms | 0.5 Hz — nearly identical to Butterworth |
| Elliptic (magenta) | ~660 ms | 0.4 Hz |
| Chebyshev II (green) | **~1750 ms** | 1.3 Hz — enormous spike |

Compared to order=4:
- Butterworth and Chebyshev I: group delay peaks are **lower** at order=2 (~530 ms vs the Run 1/2 values). This is expected — fewer filter poles means less phase accumulation.
- Chebyshev II: still has a massive group delay spike (~1750 ms), now centered at 1.3 Hz instead of 2 Hz. The spike shifted because the zero locations changed with the lower order design.
- Elliptic: peak is lower than order=4 (660 ms vs 3000+ ms at order=4). At order=2, the elliptic filter has fewer poles/zeros, so the phase concentration is less extreme.

For offline use (`filtfilt`): still irrelevant — all group delay is cancelled.  
For real-time use: Butterworth/Chebyshev I at order=2 would actually be acceptable (~530 ms peak). But the BPM accuracy is unacceptable.

---

## Figure 4 — Filtered BVP Signals (order=2)

![Filtered Signals order=2](/Users/devanshbajwala/Desktop/Screenshot 2026-04-05 Run3 Signals.png)

This figure shows the biggest visual impact of lowering the order:

- **Blue (Butterworth):** Signal is present but lower amplitude (~0.006 peak). Waveform shape is smoother than order=4 version. Some noise still visible — the weak transition band lets high-frequency noise bleed through slightly.
- **Magenta (Elliptic):** Highest amplitude of the four (~0.016 peak). Even at order=2, Elliptic's sharp cutoff preserves more of the cardiac signal energy in its output. This explains the higher std (0.004523).
- **Red (Chebyshev I):** Very similar to Elliptic, amplitude close to 0.014 peak. Both Chebyshev I and Elliptic now differ clearly from Butterworth.
- **Green (Chebyshev II):** Nearly flat — amplitude barely exceeds 0.002. The filter is suppressing almost all signal. The std of 0.001121 (vs 0.003809 for Butterworth) confirms this — Chebyshev II at order=2 is passing less than 30% of the signal energy that Butterworth passes.

### Why does Elliptic have MORE signal at order=2 than Butterworth?
At order=2, Butterworth's passband is not truly flat — it starts rolling off before the upper cutoff, losing some cardiac signal energy. Elliptic maintains a flatter passband response right up to the cutoff, then drops sharply. So Elliptic actually passes more of the cardiac frequency components faithfully.

### Why is Chebyshev II essentially dead?
At order=2, the Chebyshev II passband is so narrow that very little of the 0.67–3.0 Hz band actually gets through. The filter is designed for stopband behaviour, not passband behaviour, and at low order there aren't enough poles to keep the passband open while achieving the desired stopband rejection.

---

## Figure 5 — PSD (order=2)

![PSD order=2](/Users/devanshbajwala/Desktop/Screenshot 2026-04-05 Run3 PSD.png)

- **Blue (Butterworth) + Red (Chebyshev I):** Nearly overlapping, flat noise floor around −50 dB/Hz across the entire band. **No dominant peak visible** — the signal is there but drowned in noise because the weak transition band let high-frequency noise in from above 3.0 Hz. The noise floor rises to the same level as the cardiac signal. Butterworth still detects 56.2 BPM because it peaks at 0.9374 Hz even if weakly. Chebyshev I peaks at 0.8788 Hz — a 1-bin shift from noise affecting the peak.

- **Magenta (Elliptic):** Slightly higher floor (~−45 dB/Hz) and a marginally clearer shape. The elliptic's sharper cutoff preserves signal amplitude better. Peak at 0.8788 Hz → 52.7 BPM.

- **Green (Chebyshev II):** Very different curve — starts at −110 dB/Hz at 0.67 Hz, rises to a peak around 1.3–1.4 Hz at −54 dB/Hz, then drops steeply. Peak at 1.35 Hz → 80.9 BPM. The filter is only passing the signal around 1.3 Hz effectively, so that frequency dominates the PSD.

### Why did Chebyshev I and Elliptic shift from 56.2 → 52.7 BPM?
The order=2 transition band is so wide that noise from the stopband leaks into the passband. This raises the noise floor near the band edges, which can shift the apparent PSD peak by one or two frequency bins. At Welch resolution with this signal length, one bin is approximately 0.05 Hz = 3 BPM. The shift from 0.9374 → 0.8788 Hz is 3.5 BPM — exactly one bin shift caused by noise contaminating the low-frequency edge of the band.

---

## Key Findings from Run 3

### order=2 destroys filter consensus
At order=4, three filters agreed on 56.2 BPM. At order=2, only Butterworth stays at 56.2 BPM, Chebyshev I and Elliptic shift to 52.7, and Chebyshev II goes to 80.9.

### order=2 has insufficient stopband rejection
The transition bands are so wide that stopband noise contaminates the passband PSD. The cardiac peak is no longer cleanly isolated.

### Chebyshev II at order=2 is non-functional for rPPG
The signal is suppressed to 30% of Butterworth's output. Not usable.

### Lower order = lower group delay (but it doesn't matter for offline use)
At order=2, Butterworth/Chebyshev I group delay peaks around 530 ms vs higher values at order=4. For real-time this would be better latency, but the accuracy penalty is unacceptable.

### Elliptic is the most robust at low order
Even at order=2, Elliptic maintains the sharpest cutoff of all four filters. It preserves more cardiac signal amplitude than Butterworth or Chebyshev I at the same order. However its BPM estimate is still degraded (52.7 vs 56.2) due to noise contamination.

---

## Summary Table: order=4 vs order=2

| Metric | order=4 | order=2 |
|---|---|---|
| Filter consensus | 3/4 agree at 56.2 BPM | 1/4 at 56.2, 2/4 at 52.7, 1/4 at 80.9 |
| Chebyshev II std | 0.002796 | 0.001121 |
| Elliptic std | 0.003839 | 0.004523 |
| Noise leakage | Low | High (wide transition bands) |
| Group delay (BW peak) | ~500 ms | ~530 ms |
| Recommendation | Viable for offline | Too noisy for reliable BPM |

---

## Next Steps

### Try order=6
Going higher from the baseline (order=4) to see how sharpness improves further. Expected: all 4 filters converge more tightly, Chebyshev II gets closer to agreeing with the others, group delays increase.

```matlab
order = 6;   % change from 4
% Keep: Rp=0.5, Rs=40, f_low=0.67, f_high=3.0
```

After order exploration, try `Rs=60` to see effect of deeper stopband rejection on Chebyshev II and Elliptic behaviour.

# IIR Filter Design — Analysis Run 1
**Date:** 2026-04-05  
**Script:** `bpm_control_new.m`  
**Parameters:** order=4, Rp=0.5 dB, Rs=40 dB, band=[0.67, 3.0] Hz  
**Video:** Movie on 4-5-26 at 10.28.mov (~21 s)

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

---

## Figure 1 — Magnitude Response

![Magnitude Response](/User/devanshbajwala/Desktop/Screenshot%202026-04-05%20at%2011.51.07.png)

### What the axes mean
- **X-axis:** Frequency in Hz. The two dashed vertical lines mark the filter cutoffs: 0.67 Hz (lower) and 3.0 Hz (upper). Everything between these lines is the **passband** — what you want to keep. Everything outside is the **stopband** — what you want to block.
- **Y-axis:** Gain in decibels (dB). 0 dB = signal passes through unchanged. Negative dB = signal is attenuated. The more negative, the stronger the rejection.

### What to look for
| Region | Ideal behaviour |
|---|---|
| Passband (0.67–3.0 Hz) | Flat, as close to 0 dB as possible. Any variation here = the filter is modifying signal it should preserve. |
| Transition band (just outside cutoffs) | Drop as steeply as possible. A steeper drop means less leakage from noise just outside the band. |
| Stopband (far from cutoffs) | As deep (negative) as possible. −40 dB means noise is attenuated by a factor of 100×. |

### What this plot is actually showing (problem)
**The y-axis stretched to −800 dB, making the passband invisible.** This is a known MATLAB bug: Chebyshev II and Elliptic filters have exact mathematical zeros in the stopband — the magnitude literally hits 0, and `20×log10(0) = −∞`. MATLAB ignores the `ylim([-80 5])` setting when the data contains −Inf. The `xline` calls also reset the x-axis limits.

**This plot is not readable in its current form. It is being fixed in Run 2.**

### What we can still read
Even though the plot is broken, the passband region (between the dashed lines, near 0 dB) shows the filters' relative behaviour. Chebyshev II and Elliptic show visible ripple notches (the dips going far below the others). Butterworth and Chebyshev I appear as the smoothest lines near 0 dB.

---

## Figure 2 — Group Delay

![Group Delay](/Users/devanshbajwala/Desktop/Screenshot 2026-04-05 at 11.50.59.png)

### What the axes mean
- **X-axis:** Frequency in Hz
- **Y-axis:** How many milliseconds a signal at that frequency is delayed by the filter

A flat group delay = all frequencies delayed equally = the pulse shape is preserved.  
A varying group delay = different frequencies arrive at different times = the BVP waveform is smeared/distorted.

### What you see
| Filter | Group delay behaviour |
|---|---|
| Butterworth (blue) | Smooth peaks at band edges (~500 ms), flat elsewhere |
| Chebyshev I (red) | Similar to Butterworth, slightly more pronounced peaks |
| Chebyshev II (green) | Additional large spikes at ~2 Hz and ~3 Hz (~1750 ms peak) |
| Elliptic (magenta) | Sharpest spikes at band edges, exceeding 3000 ms |

### Why this matters
For **offline processing** (`filtfilt`): `filtfilt` runs the filter forward then backward, which perfectly cancels all phase delay. Group delay becomes exactly zero. So for our current script, this graph is informational only.

For **real-time / live-stream use** (future ROS2 implementation): `filtfilt` cannot be used. You would use `filter` instead, and the group delay directly determines how late the BPM reading arrives. Elliptic's 3000 ms spike at the band edge = 3 seconds of lag at 0.67 Hz. That is unacceptable for real-time.

**Conclusion for real-time:** Butterworth is the safest choice. Its group delay variation is the most gradual and predictable.

---

## Figure 3 — Phase Response

![Phase Response](/Users/devanshbajwala/Desktop/Screenshot%202026-04-05%20at%2011.50.23.png)

### What the axes mean
- **X-axis:** Frequency Hz
- **Y-axis:** Phase shift in degrees introduced by the filter at each frequency
- A perfectly linear phase response = straight diagonal line = no waveform distortion

### What you see
- **Butterworth (blue)** and **Chebyshev I (red):** Smooth, continuously declining curves. Similar to each other.
- **Chebyshev II (green)** and **Elliptic (magenta):** Have sudden vertical jumps between +300° and −400°.

### Why Chebyshev II and Elliptic have jumps
These jumps are **not real filter discontinuities**. They are phase-wrapping artifacts from the `unwrap()` function. At the filter's exact zeros (where magnitude = 0), the phase is mathematically undefined and flips by 180°. The `unwrap` function tries to correct this but fails at the zeros. The true phase is continuous — this is a visualisation artifact only.

### What this tells us
Since we use `filtfilt`, all phase distortion is cancelled for all four filters. This graph mainly confirms that Butterworth and Chebyshev I have simpler, more predictable phase behaviour.

---

## Figure 4 — Filtered BVP Signals

![Filtered Signals](/Users/devanshbajwala/Desktop/Screenshot%202026-04-05%20at%2011.50.38.png)

### What the axes mean
- **X-axis:** Time in seconds (the ~21 second video)
- **Y-axis:** Amplitude of the BVP signal (dimensionless — the CHROM output after detrending)
- **Gray:** S_det — the raw detrended CHROM signal, very noisy, the input to all four filters
- **Coloured lines:** Each filter's output

### What you see
- **Blue (Butterworth)**, **Red (Chebyshev I)**, **Magenta (Elliptic):** Nearly perfectly overlapping. You cannot distinguish them. All three extract the same underlying oscillation.
- **Green (Chebyshev II):** Clearly different — larger amplitude swings, slower oscillation period. Its peaks are spaced further apart, corresponding to ~66.8 BPM vs the others' 56.2 BPM.

### Reading the rhythm
Look at the approximate spacing between peaks in the blue/red/magenta lines. At 56 BPM, one heartbeat = 60/56 ≈ 1.07 seconds. You should see repeating peaks roughly every 1 second. This is visible in the signal between t=5 and t=15 s where it is relatively stable.

### Why Chebyshev II is the outlier
Explained in detail in Figure 5 below.

---

## Figure 5 — PSD of Filtered Signals (Cardiac Band)

![PSD](/Users/devanshbajwala/Desktop/Screenshot%202026-04-05%20at%2011.50.50.png)

### What the axes mean
- **X-axis:** Frequency Hz — zoomed into the cardiac band only (0.67–3.0 Hz)
- **Y-axis:** Power Spectral Density in dB/Hz — how much power the signal has at each frequency
- **A peak** in this plot = a dominant oscillation at that frequency = the heart rate candidate

### What you see
- **Blue, Red, Magenta (Butterworth, Chebyshev I, Elliptic):** Essentially identical curves, sitting on top of each other. All show a broad hump peaking around **0.94 Hz → 56.2 BPM**. The noise floor is relatively flat around −55 to −60 dB/Hz across the band.
- **Green (Chebyshev II):** Very different. Power is heavily suppressed in the lower part of the band (0.67–1.0 Hz drops to −83 dB), rises through the middle, then falls off sharply after 2.3 Hz (dropping to −110 dB by 3.0 Hz). Its peak sits at **1.11 Hz → 66.8 BPM**.

### Why Chebyshev II gives a different BPM — the specification mismatch

This is a critical filter design concept:

**For `butter`, `cheby1`, `ellip`:** The cutoff frequencies you specify (0.67 and 3.0 Hz) define the **passband edge** — where the filter's gain is at most 3 dB below 0 dB (Butterworth) or at most Rp dB below (Cheby1, Elliptic). The filter passes everything in [0.67, 3.0] Hz.

**For `cheby2`:** The cutoff frequencies define the **stopband edge** — where the filter is attenuated by Rs dB. The actual passband is **narrower** than [0.67, 3.0] Hz. The filter begins rolling off before reaching 0.67 Hz from below, meaning it suppresses the low end of the cardiac band. The signal around 0.94 Hz is partially attenuated, so the peak detection shifts to 1.11 Hz instead.

This is not a malfunction — it is how Chebyshev Type II is mathematically defined. To make it comparable, you would need to find the passband edge frequencies that give a stopband edge at 0.67/3.0 Hz, which requires a separate calculation.

---

## Key Findings from Run 1

### Filter agreement
Three of four filters (Butterworth, Chebyshev I, Elliptic) agree exactly: **56.2 BPM, peak at 0.9374 Hz**.

### Chebyshev II is not comparable as specified
Its cutoff specification has different semantics. Cannot directly compare its BPM result to the other three without recalculating the band edges.

### For offline processing (filtfilt): all three agreeing filters are equivalent
The filtered signals and PSDs are practically identical. The choice between Butterworth, Chebyshev I, and Elliptic makes no difference to the final BPM estimate at order=4.

### For real-time processing: Butterworth is recommended
Lowest and most gradual group delay variation. Most predictable behaviour at band edges.

---

## What to Change and What to Expect

### Parameter: `order`
Try values: 2, 4, 6, 8

| Effect | What you will see |
|---|---|
| Higher order | Steeper transition band (sharper cutoff), more group delay variation, more edge ringing |
| Lower order | Gentler rolloff, less ringing, but more noise leakage near the band edges |

### Parameter: `Rp` (passband ripple — Chebyshev I, Elliptic only)
Try: 0.1, 0.5, 1.0, 3.0 dB

| Effect | What you will see |
|---|---|
| Lower Rp | Flatter passband, but the filter needs a higher order to achieve the same transition band sharpness |
| Higher Rp | More ripple in the passband, but steeper transition. At 3 dB, the cutoff point is at the −3 dB point by definition |

### Parameter: `Rs` (stopband attenuation — Chebyshev II, Elliptic only)
Try: 20, 40, 60, 80 dB

| Effect | What you will see |
|---|---|
| Higher Rs | Deeper stopband rejection, but more group delay variation and ringing |
| Lower Rs | Less attenuation of out-of-band noise |

---

## Fix Applied in Run 2

**Problem:** Magnitude plot y-axis stretched to −800 dB due to −Inf values from filter zeros. `ylim` overridden by MATLAB. `xline` calls reset x-axis limits.

**Fix:**
1. Clip all dB values to −80 dB floor before plotting: `max(20*log10(abs(H)), -80)`
2. Compute `freqz` on a zoomed frequency vector `[0, 6]` Hz instead of full Nyquist range
3. Set `xlim` and `ylim` **after** all `xline` calls
4. Show original broken plot (Figure 1a) alongside fixed plot (Figure 1b) so the difference is visible

---

## Next Steps

1. Run 2: Fix magnitude plot, verify all 4 filters are now readable
2. Tweak parameters (order, Rp, Rs) and observe changes
3. Decide on final filter for frequency domain analysis
4. Frequency domain analysis: FFT vs Welch vs STFT (Prof. Chen Section 2)
5. MUSIC implementation (Prof. Chen Section 3) — after studying the paper

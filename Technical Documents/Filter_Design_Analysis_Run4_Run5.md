# IIR Filter Design — Analysis Run 4 (order=6) and Run 5 (order=8)
**Date:** 2026-04-05  
**Script:** `bpm_control_new.m`  
**Parameters:** Rp=0.5 dB, Rs=40 dB, band=[0.67, 3.0] Hz  
**Run 4:** order=6 | **Run 5:** order=8  
*(No figures shared — console output is the primary data)*

---

## Console Output

### Run 4 — order=6
```
Filter outputs: std bw=0.003846  c1=0.003774  c2=0.003427  el=0.003693

Filter           Peak Hz     BPM
--------------------------------
Butterworth       0.9374    56.2
Chebyshev I       0.8788    52.7   ← still shifted
Chebyshev II      0.9374    56.2   ← rejoined consensus
Elliptic          0.9374    56.2
```

### Run 5 — order=8
```
Filter outputs: std bw=0.003848  c1=0.003701  c2=0.003632  el=0.003579

Filter           Peak Hz     BPM
--------------------------------
Butterworth       0.9374    56.2
Chebyshev I       0.9374    56.2   ← back in consensus
Chebyshev II      0.9374    56.2
Elliptic          0.9374    56.2
```

**At order=8: all four filters agree at exactly 56.2 BPM. Full consensus.**

---

## Complete Cross-Run BPM Table

| Filter | order=2 | order=4 | order=6 | order=8 |
|---|---|---|---|---|
| Butterworth | 56.2 ✓ | 56.2 ✓ | 56.2 ✓ | 56.2 ✓ |
| Chebyshev I | 52.7 ✗ | 56.2 ✓ | 52.7 ✗ | 56.2 ✓ |
| Chebyshev II | 80.9 ✗✗ | 66.8 ✗ | 56.2 ✓ | 56.2 ✓ |
| Elliptic | 52.7 ✗ | 56.2 ✓ | 56.2 ✓ | 56.2 ✓ |
| Filters agreeing | 1/4 | 3/4 | 3/4 | **4/4** |

---

## Complete Std Progression

| Filter | order=2 | order=4 | order=6 | order=8 |
|---|---|---|---|---|
| Butterworth | 0.003809 | 0.003858 | 0.003846 | 0.003848 |
| Chebyshev I | 0.004533 | 0.003895 | 0.003774 | 0.003701 |
| Chebyshev II | 0.001121 | 0.002796 | 0.003427 | 0.003632 |
| Elliptic | 0.004523 | 0.003839 | 0.003693 | 0.003579 |

---

## What Each Progression Tells Us

### Butterworth std is stable across all orders (0.003809–0.003858)
Butterworth's maximally flat design means the passband gain is consistent regardless of order — the passband shape doesn't change much, only the transition band steepness. So the amount of cardiac signal that gets through stays nearly constant.

### Chebyshev II std: 0.001121 → 0.003632 (rising toward Butterworth's level)
As order increases, Chebyshev II can keep its passband more open while still achieving the Rs=40 dB stopband attenuation. The effective passband progressively widens toward [0.67, 3.0 Hz] as the order increases. At order=8 it's at 0.003632 — still slightly below Butterworth's 0.003848, meaning there is still some passband suppression from the specification mismatch, but it's small enough not to affect BPM peak detection.

### Chebyshev I and Elliptic std: both decline as order increases
Higher order = more selective filtering = narrower passband shape = slightly less total signal energy in the output. Both converge toward ~0.003600 by order=8.

---

## Why Chebyshev I Was the Outlier at order=6

At order=6, Chebyshev I still gave 52.7 BPM while Butterworth, Chebyshev II, and Elliptic all gave 56.2. This seems counterintuitive — why would Chebyshev I be the odd one out?

**Explanation — passband ripple oscillation landing on the cardiac peak:**

Chebyshev Type I has an equiripple passband — the gain oscillates in the passband between 0 dB and −Rp dB (−0.5 dB here). At order=6, this equiripple pattern creates 6 oscillations across the passband. Depending on where those oscillations land, a small dip can fall right on the cardiac peak at 0.9374 Hz, making the neighbouring 0.8788 Hz point appear slightly higher in the PSD. At order=8, the ripple oscillations are distributed more finely (8 oscillations) and no single dip is deep enough to shift the peak by one bin.

This is a subtle but real effect: **Chebyshev I passband ripple can perturb the BPM estimate at intermediate orders.** It resolves at higher orders where the ripple is finer-grained.

---

## Why Chebyshev II Rejoined at order=6

At order=4, Chebyshev II's stopband specification at 0.67/3.0 Hz made the effective passband so narrow that the 0.9374 Hz peak was partially attenuated, shifting the detected peak. By order=6, the filter has enough poles to:
1. Place tight zeros that enforce the 40 dB stopband outside [0.67, 3.0 Hz]
2. Keep the passband gain near 0 dB over a wide enough region inside the band

The transition from 66.8 BPM (order=4) → 56.2 BPM (order=6) confirms that the effective passband has expanded enough to include the 0.9374 Hz peak at unit gain.

---

## Filter Selection Decision

Based on all 5 runs (order=2, 4, 6, 8):

### Selected: **order=4, Butterworth**

| Criterion | Butterworth order=4 |
|---|---|
| BPM accuracy | 56.2 BPM at all orders tested — most robust |
| Consensus | Part of 3/4 agreement from order=4 upward |
| Passband | Maximally flat — no ripple in passband |
| Group delay | Most gradual, most predictable |
| Numerical stability | order=4 is well within the safe range for `[b,a]` coefficient form |
| Real-time readiness | Best choice when `filtfilt` is replaced by `filter` in ROS2 |
| Simplicity | No Rp, Rs tuning needed — just order and cutoff |

### Why not order=8?
- Full consensus is achieved at order=8, but Butterworth at order=4 already gives the correct answer.
- Higher order = larger `[b,a]` coefficient vectors = higher numerical sensitivity. At order>6, the `[b,a]` form in MATLAB can accumulate floating-point errors. The correct approach at high order is second-order sections (SOS), which requires a different MATLAB function (`sosfilt` or `designfilt`). For this project's needs, order=4 is sufficient.
- Real-time latency increases with order. For ROS2 live-stream with `filter` (not `filtfilt`), lower order is always preferred when accuracy is not compromised.

### Why not Elliptic?
Elliptic gives the same BPM at order=4, and is theoretically "optimal" (most stopband rejection for a given order). However:
- Phase behaviour is more complex (large spikes in group delay at band edges)
- Passband and stopband both have ripple — two parameters to tune instead of zero
- Offers no practical accuracy advantage over Butterworth at order=4 for this signal

---

## Final Filter Parameters for the Pipeline

```matlab
f_low  = 0.67;   % Hz
f_high = 3.0;    % Hz
order  = 4;

Wn = [f_low f_high] / (fs/2);
[b, a] = butter(order, Wn, 'bandpass');
S_filt = filtfilt(b, a, S_det);
```

---

## Next Steps

Section 1 (Filter Design) is complete. Proceeding to:

**Section 2 — Frequency Domain Analysis**  
Compare FFT vs Welch vs STFT at different window lengths. Key question: how short can the window be while keeping BPM error < 5 BPM? This directly motivates why MUSIC (Section 3) is needed.

Parameters to vary in the frequency analysis:
- Window length: 5 s, 10 s, 15 s (full signal)
- FFT resolution: 1/window_length Hz per bin
- Welch: overlap and averaging effect
- STFT: sliding window, time-varying BPM tracking

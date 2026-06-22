# Filter Design: Mathematical Foundation
### rPPG Bandpass Filter for DocBot Heart Rate Extraction
**Reference:** Chen & Tomizuka, *Selective Model Inversion and Adaptive Disturbance Observer for Rejection of Time-Varying Vibrations on an Active Suspension*, EJC 2013.

---

## 1. The Problem and Why It Connects to Prof. Chen's Paper

The rPPG BVP signal contains:
- **DC + respiratory drift** (0–0.4 Hz) — skin illumination changes, slow motion
- **Cardiac signal** (0.7–3.5 Hz) — the signal we want
- **Motion artifact and noise** (> 4.5 Hz)

Prof. Chen's Fig. 4 in the EJC paper shows the core principle: a well-designed frequency-selective filter dramatically attenuates the target band (55 Hz vibration in his case) while leaving all other frequencies untouched. The closed-loop spectrum shows a sharp, deep notch exactly at 55 Hz and a flat floor everywhere else. Our filter does the inverse — we *pass* the cardiac band and *reject* everything outside it. The design philosophy is identical: **selectivity requires sharpness at the transition band edges**, and the choice of filter type determines how efficiently (in terms of filter order) that sharpness is achieved.

Prof. Chen's paper also motivates the IIR vs FIR comparison directly. His disturbance observer uses an IIR Q-filter (Section IV of the paper) because IIR achieves narrow-band selectivity with minimum parameters. He explicitly notes that FIR-based adaptive algorithms exist but require more parameters for the same frequency resolution — exactly the tradeoff we quantify here.

---

## 2. Filter Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Lower passband edge `f_p1` | 0.7 Hz | 42 BPM minimum physiological heart rate |
| Upper passband edge `f_p2` | 3.5 Hz | 210 BPM maximum |
| Lower stopband edge `f_s1` | 0.4 Hz | 0.3 Hz transition to suppress respiratory drift (~0.2–0.3 Hz) |
| Upper stopband edge `f_s2` | 4.5 Hz | 1.0 Hz transition above cardiac band |
| Passband ripple `Rp` | 1 dB | Max ±0.5 dB gain variation inside 0.7–3.5 Hz |
| Stopband attenuation `Rs` | 40 dB | Signal outside band reduced to 1% of its amplitude |

The **lower transition band (0.3 Hz wide)** is the critical constraint — it is the narrowest and drives the required filter order for both IIR and FIR designs.

Normalised edge frequencies used in MATLAB:
```
Wp = [f_p1, f_p2] / (fs/2)    % passband, normalised to [0,1]
Ws = [f_s1, f_s2] / (fs/2)    % stopband, normalised to [0,1]
```

---

## 3. IIR Filter Design

An IIR filter has feedback. Its transfer function is a rational fraction in z:

```
H(z) = B(z)/A(z) = (b0 + b1·z⁻¹ + … + bM·z⁻ᴹ) / (1 + a1·z⁻¹ + … + aN·z⁻ᴺ)
```

Because A(z) ≠ 1, the filter has poles. A pole at location p inside the unit circle |p| < 1 creates a resonance — the filter's impulse response grows and then decays, never reaching exactly zero in finite time. This gives the "infinite" in IIR.

**Key advantage:** low filter order for sharp transitions.
**Key tradeoff:** nonlinear phase (different frequencies experience different group delays). For our offline pipeline, `filtfilt` (forward + backward pass) cancels all phase. For real-time deployment, this matters.

MATLAB's `*ord()` functions compute the **minimum order** to meet the {Rp, Rs, Wp, Ws} specification:
```matlab
[N, Wn] = buttord(Wp, Ws, Rp, Rs)
```

### 3.1 Butterworth Filter

**Magnitude squared response:**
```
|H(jΩ)|² = 1 / (1 + (Ω/Ωc)^(2N))
```
All derivatives of |H|² at Ω = 0 are zero — this is called "maximally flat." The response is monotonically decreasing with no ripple anywhere. Rolloff is −20·N dB/decade beyond the cutoff.

Consequence: to achieve Rs = 40 dB attenuation at a stopband edge that is close to the passband edge, the Butterworth filter needs the highest order of all four IIR types. It "wastes" tolerance budget by being flat everywhere instead of concentrating effort at the transition.

**Proof of maximally flat property:** Expanding around Ω = 0:
```
|H(jΩ)|² = 1 - (Ω/Ωc)^(2N) + O(Ω^(4N))
```
The first 2N−1 derivatives at Ω = 0 are all zero, leaving only the 2N-th derivative non-zero.

### 3.2 Chebyshev Type I Filter

**Magnitude squared response:**
```
|H(jΩ)|² = 1 / (1 + ε²·Tₙ²(Ω/Ωₚ))
```
where Tₙ is the Nth-order Chebyshev polynomial and ε² = 10^(Rp/10) − 1.

The Chebyshev polynomial is defined by:
```
T_N(cos θ) = cos(Nθ)
```
which creates equiripple oscillations between ±1 in the passband (|Ω| ≤ Ωₚ). Outside the passband, T_N grows monotonically. This means:
- Passband: |H|² oscillates between 1/(1+ε²) and 1 → equiripple within ±Rp dB
- Stopband: monotonically decreasing, steeper rolloff than Butterworth at the same order

By "using up" the allowed passband ripple budget in an equiripple fashion rather than keeping the response flat, Cheby I achieves a steeper transition for the same N.

### 3.3 Chebyshev Type II Filter

Type II is the "inverse Chebyshev" — obtained by applying a frequency transformation to Type I:
```
|H(jΩ)|² = 1 / (1 + 1/(ε²·Tₙ²(Ωₛ/Ω)))
```

This results in:
- **Flat passband** (monotonic, no ripple)
- **Equiripple stopband** — the response oscillates between −Rs dB and −∞ dB

The equiripple in the stopband occurs because Type II places **transmission zeros directly on the unit circle** (in the z-domain). These zeros create exact nulls at specific stopband frequencies, guaranteeing that the stopband floor never exceeds −Rs dB. This is the only IIR filter in this set with guaranteed hard stopband notches.

### 3.4 Elliptic (Cauer) Filter

The elliptic filter uses Jacobi elliptic rational functions to achieve equiripple in **both** the passband and the stopband simultaneously:

```
|H(jΩ)|² = 1 / (1 + ε²·Rₙ²(ξ, Ω/Ωₚ))
```

where Rₙ is the Nth-degree Chebyshev rational function of the first kind with selectivity factor ξ = Ωₚ/Ωₛ.

This is the **minimax-optimal IIR filter**: for any given {Rp, Rs, transition bandwidth}, the elliptic filter achieves the specification with the lowest possible filter order. No other IIR design can do better. The proof follows from the equiripple theorem — any approximation that achieves the minimax error must alternate between its extrema the maximum possible number of times, which the elliptic function does by construction.

**In our case:** the elliptic filter typically requires 2–3× fewer poles than the Butterworth filter for the same {Rp, Rs, Wp, Ws} specification.

### 3.5 Zero-Phase Filtering with filtfilt

`filtfilt` applies the filter twice: once forward, once backward. If the forward pass introduces phase φ(ω) at each frequency, the backward pass adds another φ(ω). When reversed, the net phase is φ − φ = 0. The magnitude response is squared: |H(ω)|² rather than |H(ω)|. This doubles the effective filter order but eliminates all phase distortion.

---

## 4. FIR Filter Design

An FIR filter has no feedback. Its transfer function is a finite polynomial:
```
H(z) = b0 + b1·z⁻¹ + b2·z⁻² + … + bN·z⁻ᴺ
```

Because H(z) has no denominator (all poles sit at z = 0 inside the unit circle), the filter is **unconditionally stable**.

**Key advantage:** symmetric coefficients (b_n = b_{N−n}) guarantee exactly linear phase:
```
∠H(ω) = −(N/2)·ω
```
This is a straight line — every frequency is delayed by the same constant N/(2·fs) seconds, so the waveform shape is perfectly preserved. This property is structurally guaranteed by the filter's symmetry and cannot be obtained from IIR filters.

**The cost:** FIR filters require much higher order than IIR for the same transition sharpness.

### 4.1 Kaiser Order Estimate

The Kaiser formula estimates the minimum FIR order to achieve Rs dB of stopband attenuation over a transition band of width Δf Hz:

```
N ≈ (Rs − 7.95) / (2.285 · Δω_min)
```

where `Δω_min = 2π · Δf / fs` is the tightest transition width in rad/sample.

For our spec at fs ≈ 30 fps:
```
Δf = 0.3 Hz  (lower transition)
Δω = 2π · 0.3/30 = 0.0628 rad/sample
N  = (40 − 7.95) / (2.285 · 0.0628) ≈ 223 taps
```

Compare to the elliptic IIR order of ~6–8. The FIR latency is N/(2·fs) ≈ 3.7 seconds — significant for real-time applications but irrelevant for offline processing.

The reason for the high order is that `Δf/fs = 0.01` — only 1% of the Nyquist bandwidth. The FIR filter must implement a very narrow transition in normalised frequency, which demands a long impulse response.

### 4.2 Windowed-Sinc with Hamming Window

The ideal bandpass impulse response is the inverse DTFT of a rectangular frequency window:

```
h_ideal[n] = (2·f_p2/fs)·sinc(2·f_p2·(n−D)/fs) − (2·f_p1/fs)·sinc(2·f_p1·(n−D)/fs)
```

where D = N/2 is the centre delay and `sinc(x) = sin(πx)/(πx)`.

Since h_ideal is infinite in length, truncating it to N+1 samples causes **Gibbs phenomenon** — a 9% overshoot at each band edge that does not diminish as N → ∞. The solution is to multiply by a smooth window function w[n] before truncating:

```
h[n] = h_ideal[n] · w[n]
```

The **Hamming window** is:
```
w_Hamming[n] = 0.54 − 0.46·cos(2π·n/N),   0 ≤ n ≤ N
```

It is designed so that the two sidelobes adjacent to the main lobe nearly cancel. This reduces the peak sidelobe to −43 dB and the resulting stopband attenuation to ~41 dB — just barely meeting our Rs = 40 dB spec at high enough N.

**Transition width relationship:** `Δf ≈ 8·fs/N`, so:
- N = 51:  Δf ≈ 4.7 Hz — very wide, filter leaks badly near the cardiac band edges
- N = 151: Δf ≈ 1.6 Hz — acceptable for upper transition, still leaky at lower edge
- N = N_est: Δf ≈ 0.3 Hz — meets spec

### 4.3 Kaiser Window FIR

The Kaiser window is near-optimal for the sidelobe suppression vs. main-lobe width tradeoff:

```
w_Kaiser[n] = I₀(β · √(1 − (2n/N − 1)²)) / I₀(β),   0 ≤ n ≤ N
```

where I₀ is the zeroth-order modified Bessel function of the first kind.

The shape parameter β is computed from Rs:
```
β = 0.1102·(Rs − 8.7)                          for Rs > 50 dB
β = 0.5842·(Rs − 21)^0.4 + 0.07886·(Rs − 21)  for 21 ≤ Rs ≤ 50 dB
β = 0                                           for Rs < 21 dB
```

For Rs = 40 dB: β ≈ 3.395.

`kaiserord()` solves the inverse problem — given {Rp, Rs, band edges, fs}, it computes both the required N and β simultaneously. The filter is then built with `fir1(..., kaiser(N+1, beta))`.

### 4.4 Parks-McClellan Optimal Equiripple (firpm)

The **Remez exchange algorithm** solves the following optimisation problem:

```
Minimise over b:   max_ω | W(ω) · (H(ω) − D(ω)) |
```

where D(ω) is the desired response (1 in passband, 0 in stopbands) and W(ω) is a frequency-domain weighting function. This is the **Chebyshev (minimax) approximation** problem.

The solution has the property that the approximation error `W(ω)·(H(ω) − D(ω))` is **equiripple** — it alternates between its maximum and minimum values at least N+2 times. By the equiripple theorem (Chebyshev's theorem on best polynomial approximations), this is the unique optimal solution: **for a fixed order N, no other linear-phase FIR filter achieves a smaller worst-case weighted error**.

Comparison to Kaiser: Kaiser window FIR is near-optimal but not exactly optimal. The Parks-McClellan filter typically achieves the same attenuation with 10–15% fewer taps, or better attenuation with the same number of taps.

**Weighting in our design:**
```
w_ratio = 10^((Rs − Rp)/20)
weights_pm = [w_ratio, 1, w_ratio]   % [stopband, passband, stopband]
```

The stopband is weighted `w_ratio` times more than the passband. This tells the Remez algorithm to concentrate the error budget on achieving tight stopband attenuation rather than passband flatness — appropriate since respiratory drift (stopband) is the dominant artifact.

---

## 5. IIR vs FIR: Summary Comparison

| Property | IIR (Butterworth/Cheby/Elliptic) | FIR (Hamming/Kaiser/PM) |
|----------|-----------------------------------|--------------------------|
| Filter order for spec | Low (N ≈ 6–12) | High (N ≈ 200+) |
| Phase response | Nonlinear | Exactly linear |
| Stability | Conditional (poles must be inside unit circle) | Unconditional |
| Phase with filtfilt (offline) | Cancelled → zero-phase | Already linear, filtfilt still valid |
| Real-time single-pass | Phase distorts waveform shape | Constant delay, no distortion |
| Computational cost | Low | High (N multiplications per sample) |
| Best choice for HRV (offline) | Elliptic (minimum order) | Parks-McClellan (optimal for N) |
| Best choice for real-time DocBot | Elliptic + phase correction | Parks-McClellan |

**Recommendation for BPM estimation (offline):** The elliptic IIR filter is the practical choice — it meets the {Rp, Rs} spec with the fewest poles, and `filtfilt` removes its phase nonlinearity completely. The Parks-McClellan FIR is theoretically cleaner but offers no practical advantage for this application.

**When FIR becomes essential:** If future DocBot implementations switch to real-time single-pass filtering (to reduce latency for live heart rate display), the Parks-McClellan FIR is preferred because its linear phase guarantees no waveform distortion across the cardiac band. This is particularly important for HRV (heart rate variability) analysis where the exact shape of the BVP pulse — not just its dominant frequency — carries clinical information.

---

## 6. Connection to Prof. Chen's EJC 2013 Paper

Prof. Chen's paper explicitly compares FIR vs IIR adaptive Q-filters for narrow-band disturbance rejection (Section I and II). His conclusion — that IIR structures require the minimum number of adaptation parameters for n unknown sinusoidal components — directly parallels our finding: the elliptic IIR filter meets the cardiac band specification with the fewest poles.

The shape of his Q-filter (Fig. 3 in the paper: narrow bandpass peaks at exactly 60 Hz and 90 Hz, magnitude ≈ 0 everywhere else) is the frequency-domain template for what selective filtering should look like. In our rPPG application, we are designing the complementary structure — a filter whose passband spans 0.7–3.5 Hz and whose stopband approaches that same near-zero level outside.

The critical metric from his benchmark results (Fig. 4) is that the closed-loop attenuation at 55 Hz is ~25–30 dB while **the spectrum outside the filter band is left essentially unchanged**. This is the design criterion: a good filter concentrates its effect precisely in the target frequency range and has minimal impact elsewhere.

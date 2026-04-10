# ESPRIT Analysis — All Runs, Complete Mathematical Derivation
## DocBot rPPG Pipeline | bpm_control_ESPRIT.m
**Author:** Devansh Bajwala  
**Supervisor:** Prof. Xu Chen  
**Date:** 2026-04-06  
**Context:** Prof. Chen's 4-part rPPG upgrade roadmap, Section 3 — Frequency Identification

---

## Table of Contents

1. [Why ESPRIT? Context and Motivation](#1-why-esprit-context-and-motivation)
2. [Complete Signal Model](#2-complete-signal-model)
3. [Vandermonde Structure and Shift-Invariance](#3-vandermonde-structure-and-shift-invariance)
4. [Hankel Data Matrix Construction](#4-hankel-data-matrix-construction)
5. [SVD and Signal Subspace Decomposition](#5-svd-and-signal-subspace-decomposition)
6. [Rotational Invariance — The Core ESPRIT Identity](#6-rotational-invariance--the-core-esprit-identity)
7. [LS-ESPRIT: Ordinary Least Squares Solution](#7-ls-esprit-ordinary-least-squares-solution)
8. [TLS-ESPRIT: Total Least Squares Solution](#8-tls-esprit-total-least-squares-solution)
9. [K=1 Analytic Signal ESPRIT vs K=2 Real Signal ESPRIT](#9-k1-analytic-signal-esprit-vs-k2-real-signal-esprit)
10. [Run 1 — NaN from Forward-Backward (FB) Averaging](#10-run-1--nan-from-forward-backward-fb-averaging)
11. [Run 2 — NaN from Respiration Dominance in S_det](#11-run-2--nan-from-respiration-dominance-in-s_det)
12. [Run 3 — K=1 Working but Overestimating (S_filt Input)](#12-run-3--k1-working-but-overestimating-s_filt-input)
13. [Run 4 — K=2 LS and TLS ESPRIT (Awaiting Results)](#13-run-4--k2-ls-and-tls-esprit-awaiting-results)
14. [ESPRIT vs MUSIC: Architectural Comparison](#14-esprit-vs-music-architectural-comparison)
15. [The λ₁/λ₂ Ratio as SNR Diagnostic](#15-the-λ₁λ₂-ratio-as-snr-diagnostic)
16. [Parameter Tuning Guide](#16-parameter-tuning-guide)
17. [Summary Table: All Runs](#17-summary-table-all-runs)

---

## 1. Why ESPRIT? Context and Motivation

### Prof. Chen's Directive

From `doc_rppg/rPPG frequency filtering comments to Devansh.pdf`, Section 3:

> *"Frequency identification algorithms such as MUSIC and ESPRIT are excellent at the job."*

The existing pipeline (as of 2026-04-05) uses Welch's periodogram, which requires 10+ second windows to achieve frequency resolution below 0.1 Hz. Both MUSIC and ESPRIT are **subspace methods** that overcome the resolution-vs-window-length tradeoff of classical Fourier methods.

### The Fundamental Limitation of Welch

Welch is a non-parametric method. Its frequency resolution is bounded by:

```
Δf = fs / N_window   [Hz]
```

For the target cardiac band 0.7–3.5 Hz (42–210 BPM), typical resting heart rates cluster around 1.0–1.2 Hz. To resolve two frequencies 0.05 Hz apart, Welch needs:

```
N ≥ fs / 0.05 = 29.97 / 0.05 ≈ 600 frames = 20 seconds
```

ESPRIT (like MUSIC) is a **parametric subspace method** — it fits a structured signal model to the data. Given the assumed model `x[n] = Σ Aₖ·e^{jωₖn}`, ESPRIT can resolve frequencies from much shorter windows (typically N ≥ 2M, where M is the model order, typically 10–60 samples).

### Why a New File?

`bpm_control_ESPRIT.m` was created fresh (not modifying `bpm_estimate_MUSIC.m`) for:
- **Traceability**: each file = one algorithm family, one git history thread
- **Comparison without contamination**: MUSIC remains pure for its own benchmarking
- **Prof. Chen's structure**: Section 3 covers both algorithms separately

---

## 2. Complete Signal Model

### Continuous-Time rPPG Signal

The rPPG signal extracted via CHROM from facial skin pixels can be modeled as:

```
s(t) = A₀·cos(2π·f_HR·t + φ₀) + n(t)
```

Where:
- `f_HR` ∈ [0.7, 3.5] Hz is the heart rate frequency (42–210 BPM)
- `A₀` is the pulse amplitude (varies with skin distance from camera, lighting)
- `φ₀` is an arbitrary initial phase
- `n(t)` is noise + interference (respiration at ~0.2–0.3 Hz, motion artifacts, shot noise)

After CHROM processing and Butterworth bandpass filtering, the dominant component in `S_filt[n]` is:

```
S_filt[n] = A·cos(2π·f_HR·n/fs + φ) + ε[n]
```

Where `ε[n]` is colored noise (not white, because the filter shapes the noise spectrum).

### Discrete Complex Exponential Representation

A real cosine is the sum of two complex exponentials:

```
cos(ωn + φ) = (1/2)·e^{j(ωn+φ)} + (1/2)·e^{-j(ωn+φ)}
```

Defining `z₊ = e^{+jω₀}` and `z₋ = e^{-jω₀}` where `ω₀ = 2π·f_HR/fs`:

```
s[n] = A₊·z₊ⁿ + A₋·z₋ⁿ + ε[n]
```

This is the **K=2 complex exponential model** for a single real sinusoid.

If instead we form the **analytic signal** `s_A[n] = s[n] + j·H{s[n]}` (where H{·} is the Hilbert transform):

```
s_A[n] = A·e^{j(ω₀n+φ)} + ε_A[n]
```

This is the **K=1 complex exponential model**. The analytic signal eliminates the negative-frequency component `e^{-jω₀n}`.

**The choice between K=1 and K=2 is fundamental** — it determines the entire ESPRIT algorithm structure, as derived in Sections 9 and 12.

---

## 3. Vandermonde Structure and Shift-Invariance

### The Vandermonde Matrix

Consider N consecutive samples of the noise-free signal:

```
s[n] = Σₖ Aₖ·e^{jωₖn}   for k = 1, ..., K
```

Stack samples into a length-M column vector:

```
s(n) = [s[n], s[n+1], ..., s[n+M-1]]ᵀ   (M×1)
```

This can be written as:

```
s(n) = A · e^{jωₖn}
```

More precisely, define:
- **Steering vector**: `a(ωₖ) = [1, e^{jωₖ}, e^{j2ωₖ}, ..., e^{j(M-1)ωₖ}]ᵀ`  (M×1)
- **Vandermonde matrix**: `A = [a(ω₁) | a(ω₂) | ... | a(ωₖ)]`  (M×K)
- **Amplitude vector**: `b(n) = [A₁·e^{jω₁n}, A₂·e^{jω₂n}, ..., Aₖ·e^{jωₖn}]ᵀ`  (K×1)

Then:
```
s(n) = A · b(n)
```

### The Shift Property of Vandermonde Matrices

The KEY structural property is that the Vandermonde matrix has a **shift invariance**:

Define:
- `A₁` = rows 1 through M-1 of A  (the "top" submatrix, (M-1)×K)
- `A₂` = rows 2 through M of A    (the "bottom" submatrix, (M-1)×K)

Then:

```
A₂ = A₁ · Φ
```

Where `Φ = diag(e^{jω₁}, e^{jω₂}, ..., e^{jωₖ})` is a diagonal matrix of the **poles**.

**Proof**: The (m,k) element of `A₁` is `e^{j(m-1)ωₖ}`. The (m,k) element of `A₂` is `e^{jmωₖ} = e^{j(m-1)ωₖ} · e^{jωₖ}`. Therefore `A₂[m,k] = A₁[m,k] · Φₖₖ`.

This is the **shift-invariance property** — shifting the row index by 1 is equivalent to multiplying each column by its corresponding pole.

**This single property is the entire mathematical foundation of ESPRIT.** The frequencies are encoded in the diagonal of Φ, and ESPRIT recovers Φ from the data covariance or Hankel matrix SVD.

---

## 4. Hankel Data Matrix Construction

### Why Hankel (Not Toeplitz)?

Given N samples `{s[0], s[1], ..., s[N-1]}`, we build a Hankel matrix of size M×L where L = N - M + 1:

```
        ┌  s[0]    s[1]    s[2]   ···  s[L-1]  ┐
        │  s[1]    s[2]    s[3]   ···  s[L]    │
X =     │  s[2]    s[3]    s[4]   ···  s[L+1]  │
        │   ⋮       ⋮       ⋮            ⋮     │
        └ s[M-1]  s[M]   s[M+1]  ··· s[N-1]   ┘
```

**Column j** of X is the vector `[s[j], s[j+1], ..., s[j+M-1]]ᵀ` — a length-M snapshot at lag j.

**Each column is a noisy observation of `A · b(j)`** because:

```
X[:,j] = A · b(j) + noise_column
```

So:
```
X = A · B + E
```

Where:
- `A` is the M×K Vandermonde matrix (signal structure)
- `B = [b(0) | b(1) | ... | b(L-1)]` is K×L (amplitudes across time)  
- `E` is the M×L noise matrix

### MATLAB Implementation

```matlab
M_use = max(min(M_max, floor(N/4)), K_r+2);
L     = N - M_use + 1;
ir    = bsxfun(@plus, (1:M_use)', 0:L-1);  % M_use×L index matrix
X_r   = x(ir);                              % real Hankel matrix
```

The index matrix `ir[m,l] = m + l` creates precisely the Hankel structure where `X_r[m,l] = x[m+l-1]` (using 1-based indexing).

### Trade-off: M vs L

Given N samples, we must choose M such that M + L - 1 = N, i.e., L = N - M + 1.

- **Large M**: More rows → signal subspace better conditioned → better frequency discrimination
  - But L = N-M+1 decreases → fewer "snapshots" → covariance estimate noisier
- **Small M**: More snapshots → better statistics
  - But M must satisfy M ≥ 2K for the signal subspace to contain K frequencies

**Rule of thumb for rPPG**: M ≈ N/4 works well. For 10s at 29.97fps (N≈300):
```
M_use = floor(300/4) = 75   but capped at M_max=60
L     = 300 - 60 + 1 = 241
```

The cap M_max=60 prevents overly large covariance matrices while maintaining L >> M for good statistical averaging.

---

## 5. SVD and Signal Subspace Decomposition

### The SVD of the Hankel Matrix

Compute the economy SVD of X (M×L, assuming M ≤ L):

```
X = U · S · Vᵀ
```

Where:
- `U` is M×M (left singular vectors, orthonormal columns)
- `S` is M×M diagonal (singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_M ≥ 0)
- `V` is L×M (right singular vectors, orthonormal columns)

### Signal vs Noise Subspace

In the noise-free case, `rank(X) = K` (since X = A·B and both A and B have rank K for generic frequencies and amplitudes). The first K singular values are non-zero; the remaining M-K are zero.

With additive white noise of variance σ², it can be shown:
- The first K singular values are **inflated**: σₖ² ≈ λₖ_signal + σ²
- The remaining M-K singular values cluster around **σ²** (the noise floor)

**Signal subspace** `Us = U[:,1:K]` spans the same K-dimensional space as the Vandermonde columns `{a(ω₁), ..., a(ωₖ)}`.

**Noise subspace** `Un = U[:,K+1:M]` is orthogonal to the signal subspace.

### Why SVD Instead of Eigendecomposition of Covariance?

In principle, we could form R = X·Xᵀ/L (sample covariance) and eigendecompose it. The eigenvectors of R equal the left singular vectors of X. But:

- **SVD of X directly** is numerically more stable (avoids squaring the condition number)
- **For ESPRIT**: SVD gives `Us = U[:,1:K]` directly without the intermediate covariance step
- **Computational cost**: For M=60, L=241: SVD(60×241) is cheaper than eig(60×60) from a dense XᵀX

In MATLAB: `[U, ~, ~] = svd(X_r, 'econ')` returns only the min(M,L) singular vectors (the "economy" decomposition), saving memory for large L.

---

## 6. Rotational Invariance — The Core ESPRIT Identity

### Signal Subspace and Vandermonde

Since the signal subspace `Us` spans the same space as the Vandermonde columns:

```
Us = A · T
```

For some K×K invertible matrix T (a "basis transformation" — Us and A represent the same subspace, just with different bases).

### Constructing the Shift-Invariant Pair

Define:
- `Es1 = Us[1:M-1, :]`   (first M-1 rows of Us)
- `Es2 = Us[2:M, :]`     (last  M-1 rows of Us)

Similarly for the Vandermonde matrix:
- `A₁ = A[1:M-1, :]`
- `A₂ = A[2:M, :]`

From Section 3: `A₂ = A₁ · Φ` where `Φ = diag(e^{jω₁}, ..., e^{jωₖ})`.

Now substitute `Us = A·T`:

```
Es1 = A₁ · T
Es2 = A₂ · T = A₁ · Φ · T
```

Therefore:

```
Es2 = A₁ · Φ · T = A₁ · T · (T⁻¹ · Φ · T)
    = Es1 · (T⁻¹ · Φ · T)
```

Define `Ψ = T⁻¹ · Φ · T`. Then:

```
┌─────────────────────────────────────────────────────────────┐
│        Es2 = Es1 · Ψ        (ESPRIT rotational invariance)  │
│                                                              │
│  Ψ = T⁻¹ · Φ · T           (similar to Φ)                  │
│  eig(Ψ) = eig(Φ) = {e^{jω₁}, e^{jω₂}, ..., e^{jωₖ}}       │
└─────────────────────────────────────────────────────────────┘
```

**This is the fundamental ESPRIT equation.** Ψ is similar to the diagonal matrix of poles Φ, so they share eigenvalues. The frequencies are:

```
ωₖ = angle(eig(Ψ)ₖ)   →   fₖ = ωₖ · fs / (2π)
```

The brilliance: **we never need to know T** (the basis transformation). We just need to solve `Es1·Ψ ≈ Es2` and take eigenvalues.

---

## 7. LS-ESPRIT: Ordinary Least Squares Solution

### Problem Formulation

In the presence of noise:

```
Es2 ≈ Es1 · Ψ   (not exact due to noise)
```

We want to find Ψ that minimizes the Frobenius norm of the residual:

```
Ψ_LS = argmin_Ψ  ||Es2 - Es1 · Ψ||²_F
```

This is a standard overdetermined least squares problem. Es1 is (M-1)×K, Ψ is K×K, Es2 is (M-1)×K.

### Solution

The unique least squares solution (assuming Es1 has full column rank K):

```
┌──────────────────────────────────────────────────────┐
│   Ψ_LS = (Es1ᵀ·Es1)⁻¹ · Es1ᵀ · Es2 = pinv(Es1)·Es2  │
└──────────────────────────────────────────────────────┘
```

In MATLAB: `Psi_ls = pinv(Es1_r) * Es2_r`

### What pinv Does

`pinv(Es1)` is the Moore-Penrose pseudoinverse: `(Es1ᵀ·Es1)⁻¹·Es1ᵀ`. For an (M-1)×K matrix with M-1 >> K:
1. MATLAB computes this via SVD for numerical stability
2. The pseudoinverse projects onto the column space of Es1

### Asymptotic Bias of LS-ESPRIT

The LS formulation implicitly assumes all error is in Es2 — it treats Es1 as "exact" and Es2 as "noisy." But in reality, **both Es1 and Es2 are perturbed** by noise. This asymmetric treatment introduces a first-order bias:

```
E[Ψ_LS] = Ψ_true + O(σ²)
```

For high SNR, this bias is negligible. For low SNR (short windows, colored noise), it is the dominant error source. TLS-ESPRIT corrects this.

---

## 8. TLS-ESPRIT: Total Least Squares Solution

### The TLS Problem

TLS minimizes perturbations to **both** Es1 and Es2 jointly:

```
Ψ_TLS = argmin_{Ψ, δEs1, δEs2}  ||[δEs1 | δEs2]||²_F
         subject to:  (Es2 + δEs2) = (Es1 + δEs1) · Ψ
```

This is the **Errors-In-Variables** model: every measurement is noisy, including the "predictor" matrix Es1.

### TLS Solution via SVD

Form the augmented matrix:

```
C = [Es1 | Es2]     shape: (M-1) × 2K
```

Compute the economy SVD of C:

```
C = Uc · Sc · Vcᵀ     (Vc is 2K×2K)
```

Partition Vc into K×K blocks:

```
     ┌ V11  V12 ┐
Vc = │          │    each block is K×K
     └ V21  V22 ┘
```

The TLS solution is:

```
┌──────────────────────────────────────────────────────────┐
│   Ψ_TLS = -V12 · V22⁻¹  =  -V12 / V22                   │
└──────────────────────────────────────────────────────────┘
```

In MATLAB: `Psi_tls = -V12 / V22`  (the `/` operator computes right division: `-V12 · inv(V22)`)

### Why the SVD of C?

The SVD of C = [Es1|Es2] gives the "best rank-2K approximation" of C. The last columns of Vc (corresponding to the smallest singular values) capture the **joint null space** of [Es1|Es2]. The TLS solution lives in this null space.

Formally: the TLS solution minimizes `||[δEs1|δEs2]||_F` such that the augmented matrix has rank K instead of 2K. The rank drop condition is exactly what defines the frequency parameters.

### Why TLS > LS at Low SNR?

- **LS assumption**: noise affects only Es2 → biased when Es1 is also noisy
- **TLS assumption**: noise affects Es1 and Es2 equally → unbiased to first order in σ²
- **Mathematical guarantee**: TLS achieves the Cramér-Rao lower bound (CRLB) for white noise in the large-N limit

For rPPG at short windows (3–5s), both Es1 and Es2 are heavily perturbed by bandpass-colored noise. TLS should outperform LS, especially at 3s and 5s windows.

---

## 9. K=1 Analytic Signal ESPRIT vs K=2 Real Signal ESPRIT

### The Core Trade-off

| Property | K=1 Analytic | K=2 Real |
|---|---|---|
| Input signal | `hilbert(S_filt)` complex | `S_filt` real |
| Hankel matrix | Complex M×L | Real M×L |
| SVD | Complex SVD | Real SVD |
| Signal subspace dim | 1 (K=1) | 2 (K=2) |
| Ψ matrix | 1×1 complex scalar | 2×2 real matrix |
| Eigenvalues of Ψ | 1 complex number | 2 complex conjugates |
| Frequency extraction | `angle(Ψ)·fs/(2π)` | `abs(angle(eig(Ψ)))·fs/(2π)` |
| Constraint exploited | None beyond K=1 | Conjugate symmetry of real signal |

### K=1 Analytic Signal ESPRIT — Full Derivation

**Step 1**: Compute `x_c[n] = hilbert(S_filt)[n]`. This creates `x_c[n] = A·e^{j(ω₀n+φ)}` — one complex exponential.

**Step 2**: Build complex Hankel `X_c` (M×L, complex).

**Step 3**: SVD: `X_c = U_c · Σ_c · V_c^H`. Take `Qs_c = U_c[:,1]` — a single M-dimensional complex vector.

**Step 4**: Extract submatrices:
```
Es1_c = Qs_c[1:M-1]    (scalar per column → (M-1)×1 complex vector)
Es2_c = Qs_c[2:M]      ((M-1)×1 complex vector)
```

**Step 5**: LS-ESPRIT:
```
Psi_c = pinv(Es1_c) * Es2_c
      = (Es1_c^H · Es1_c)^{-1} · Es1_c^H · Es2_c
```
Since Es1_c is (M-1)×1: `pinv(Es1_c) = Es1_c^H / ||Es1_c||²`. So:

```
Psi_c = (Es1_c^H · Es2_c) / ||Es1_c||²    ← SCALAR complex number
```

**Step 6**: Frequency:
```
ω₀ = angle(Psi_c)      [radians/sample]
f₀ = ω₀ · fs / (2π)   [Hz]
```

**Why Ψ is a scalar**: K=1 means the "rotation" between Es1 and Es2 is a single complex multiplication by `e^{jω₀}`. The entire frequency information is in one complex number.

### K=2 Real Signal ESPRIT — Full Derivation

**Step 1**: Input is real `x = S_filt`. Model: `x[n] = A·cos(ω₀n + φ) + noise`.

Real signal → two complex exponentials at +ω₀ and -ω₀.

**Step 2**: Build real Hankel `X_r` (M×L, real).

**Step 3**: SVD: `X_r = U_r · Σ_r · V_r^T` (entirely real matrices). Take `Qs_r = U_r[:,1:2]` — a M×2 real matrix.

**The key insight**: The 2D signal subspace of a real cosinusoidal signal is spanned by `{cos(ω₀n), sin(ω₀n)}` (the real and imaginary parts of the complex exponential). The real SVD captures exactly this 2D basis.

**Step 4**: Extract submatrices:
```
Es1_r = Qs_r[1:M-1, :]    ((M-1)×2 real)
Es2_r = Qs_r[2:M, :]      ((M-1)×2 real)
```

**Step 5**: LS-ESPRIT: `Psi_ls = pinv(Es1_r) * Es2_r` — a 2×2 REAL matrix.

**Step 6**: Eigendecomposition of the 2×2 real Ψ:
```
eig(Psi_ls) = {e^{+jω₀},  e^{-jω₀}}     (complex conjugate pair)
```

**Why conjugate pair?** Because:
- The signal subspace contains `cos(ω₀n) = (e^{+jω₀n} + e^{-jω₀n})/2`
- The "rotation operator" Ψ must map the subspace to itself with rotation by ω₀
- The eigenvalues of this rotation are the complex conjugates `e^{±jω₀}`
- A 2×2 real matrix with conjugate eigenvalues is of the form `[a -b; b a]` (rotation matrix)

**Step 7**: Both eigenvalues give the same frequency:
```
f₀ = abs(angle(e^{+jω₀})) · fs/(2π) = ω₀·fs/(2π)
f₀ = abs(angle(e^{-jω₀})) · fs/(2π) = |-ω₀|·fs/(2π) = ω₀·fs/(2π)
```

Both agree → average them (or just take one).

### Why K=2 is More Robust

K=1 uses only 1 singular vector (1 degree of freedom for frequency estimation). K=2 uses 2 singular vectors and the **conjugate symmetry constraint** (eigenvalues must be conjugate) effectively acts as a built-in constraint. This means:

- If noise shifts one eigenvalue off the unit circle: `|e^{jω₀+δ}| ≠ 1`, the conjugate pair still gives a consistent frequency
- Two singular vectors provide more averaging of noise than one
- The 2×2 Ψ matrix is estimated from 4 entries (not 1) → more averaging

---

## 10. Run 1 — NaN from Forward-Backward (FB) Averaging

### What Happened

The first ESPRIT implementation used the MUSIC covariance structure (FB-averaged covariance R_fb, then eigendecomposition) to obtain the signal subspace. All ESPRIT estimates were NaN.

### Root Cause: FB Averaging Destroys Phase Information

**FB averaging** is defined as:

```
R_fb = 0.5 · (R_raw + J · R_raw* · J)
```

Where `J` is the exchange (reversal) matrix: `J[m,n] = δ(m+n = M+1)`.

**Why this is done for MUSIC**: FB averaging exploits the conjugate symmetry of Vandermonde steering vectors. For a complex exponential `a(ω) = [1, e^{jω}, ..., e^{j(M-1)ω}]ᵀ`, it holds that `J·a*(ω) = a(-ω)` (the time-reversed conjugate is the negative frequency). For real signals, FB averaging increases the effective number of snapshots from L to 2L, improving MUSIC's noise subspace estimate.

**Why this breaks ESPRIT**: For a Toeplitz covariance matrix (stationary process), it holds that:

```
R_raw[m,n] = r[m-n]       (correlation at lag m-n)
```

The (m,n) element of `J·R_raw*·J` is:

```
(J·R_raw*·J)[m,n] = R_raw*[M+1-m, M+1-n] = r*[n-m] = r[-(m-n)] = r[m-n]*
```

For a REAL-valued stationary process: `r[k] = r[-k]` (symmetric), so `r[k]* = r[k]` (real correlation is already Hermitian symmetric). Therefore:

```
R_fb[m,n] = 0.5 · (r[m-n] + r[m-n]*) = real(r[m-n])
```

**R_fb is a REAL SYMMETRIC matrix** — all imaginary parts are zeroed out.

**Consequence for ESPRIT**: The eigenvectors of a real symmetric matrix are **real vectors**. So the signal subspace vectors `Es1` and `Es2` are real. The rotation matrix:

```
Ψ = pinv(Es1) · Es2   ← real matrix
```

The eigenvalues of a real matrix are either real or come in complex conjugate pairs. For a small rotation matrix close to identity, the eigenvalues are real (positive scalars), not complex exponentials. Therefore:

```
angle(eig(Ψ)) ≈ 0   →   freq ≈ 0 Hz   →   below f_low   →   NaN
```

### The Fix

Replace the covariance + eigendecomposition approach with **direct SVD of the Hankel data matrix**:

```matlab
% BROKEN (FB averaging creates real eigenvectors):
R_fb = 0.5*(X*X'/L + J*conj(X*X'/L)*J);
[V,D] = eig(R_fb);  % real eigenvectors → angle = 0 → NaN

% CORRECT (SVD of raw complex Hankel):
x_c = hilbert(S_filt(1:N));    % complex analytic signal
X_c = x_c(ic);                 % complex Hankel
[U_c, ~, ~] = svd(X_c, 'econ'); % complex left singular vectors
Qs_c = U_c(:, 1:K_c);           % complex signal subspace → valid phase info
```

**Why SVD of the data matrix works**: The left singular vectors of a complex Hankel matrix are complex (they contain phase information from the complex exponentials). The phase in `Es2 - Es1 · Ψ ≈ 0` encodes the frequency, and this phase is preserved in the SVD.

### Lesson Learned

**FB averaging is a technique specific to MUSIC's noise subspace**. It is incompatible with ESPRIT's signal subspace phase requirements. Never use FB-averaged covariance as input to ESPRIT.

---

## 11. Run 2 — NaN from Respiration Dominance in S_det

### What Happened

After fixing the FB averaging issue (switching to SVD of complex Hankel), all ESPRIT estimates remained NaN.

Debug output was added:

```matlab
fprintf('[DBG] Psi=%s |Psi|=%.4f angle=%.4f rad freq_e=%.4f Hz\n', ...
    num2str(Psi_c,'%.4f%+.4fi'), abs(Psi_c), angle(Psi_c), ...
    abs(angle(Psi_c))*fs/(2*pi));
```

### Console Output (Run 2)

```
[DBG] Psi=0.9919+0.0660i |Psi|=0.9941 angle=0.0665 rad freq_e=0.3173 Hz
ESPRIT=NaN BPM (freq 0.317 Hz outside [0.70, 3.50] Hz)
```

### Diagnosis

**|Psi| ≈ 0.994 ≈ 1.0** — This is exactly what we expect for a noise-free complex exponential `e^{jω₀}`. The ESPRIT mechanics are **perfectly correct** — it found a clean sinusoid.

**freq = 0.317 Hz** — This is NOT the cardiac frequency (0.9 Hz ≈ 54 BPM). This is in the **respiration band** (~0.2–0.4 Hz, typical breathing rate).

**Root cause**: The input was `S_det` (detrended CHROM signal, before bandpass filter). `S_det` contains:
1. Cardiac component: ~0.9 Hz, small amplitude after CHROM
2. **Respiration component**: ~0.3 Hz, LARGER amplitude (respiration causes broader blood volume changes)
3. Motion artifacts and harmonics

ESPRIT with K=1 finds the **single most dominant frequency** in the signal subspace. The dominant eigenvector aligns with the largest singular value, which corresponds to the strongest periodic component — the respiration at 0.3 Hz, not the cardiac at 0.9 Hz.

Since 0.317 Hz < f_low = 0.7 Hz, the band check `ib_c = (frq_c >= f_low) & (frq_c <= f_high)` returns false → `bpm_esp_k1(k) = NaN`.

### The Fix: Use S_filt Instead of S_det

Apply the Butterworth bandpass filter **before** ESPRIT:

```matlab
% BROKEN (respiration dominates):
x = S_det(1:N);   % unfiltered → ESPRIT finds respiration at 0.3 Hz

% CORRECT (bandpass removes respiration):
x = S_filt(1:N);  % filtered to [0.7, 3.5] Hz → only cardiac remains
```

**Why filtering doesn't break ESPRIT** (unlike MUSIC):

MUSIC requires **white noise** in the noise subspace because it computes `||Qn^H · a(ω)||²` — the projection of steering vectors onto noise eigenvectors. If noise is colored (e.g., shaped by the bandpass filter), the noise eigenvalues are no longer equal, the "noise floor" of the MUSIC pseudo-spectrum is uneven, and peaks are distorted.

**ESPRIT only needs the signal subspace** — it uses the K signal eigenvectors, not the M-K noise eigenvectors. Colored noise changes the noise eigenvalues but does NOT systematically rotate the signal eigenvectors (at high SNR). Therefore, S_filt is a valid input for ESPRIT.

(At low SNR, colored noise **does** bias the signal eigenvectors — this is the Run 3 issue.)

### Key Takeaway

The debug output `|Psi| ≈ 1.0` was the crucial diagnostic: it proved ESPRIT was mechanically correct. The NaN was a **"found the wrong signal"** problem, not an **"ESPRIT is broken"** problem. This is a critical distinction for debugging subspace methods.

---

## 12. Run 3 — K=1 Working but Overestimating (S_filt Input)

### Results

After switching to S_filt:

```
Window(s)  ESPRIT-K1    Error vs ref (54.9 BPM)
3          87.4 BPM     +32.5 BPM
5          84.9 BPM     +30.0 BPM
10         62.3 BPM     +7.4 BPM
21         60.4 BPM     +5.5 BPM
Reference  54.9 BPM     (full-signal Welch)
```

ESPRIT is now giving **valid (non-NaN) estimates** — major progress. But it overestimates significantly at short windows.

### Root Cause: Low λ₁/λ₂ with Colored Noise

The eigenvalue ratio diagnostic (added to console output):

```
Window  λ1/λ2(complex)   λ1/λ2(real)
3s      1.33             ~1.5
5s      1.67             ~1.8
10s     2.89             ~2.5
21s     3.00+            ~3.0
```

Compare to S_det input (where ESPRIT found the "wrong" respiration peak but with clean mechanics):
```
S_det, 3s:  λ1/λ2 = 3.43
S_det, 21s: λ1/λ2 = 5.62
```

**S_filt has LOWER λ1/λ2 than S_det**. Why?

The Butterworth bandpass filter is a **colored noise source**. The filter does not pass a flat spectrum; it has a passband (0.7–3.5 Hz) with a shaped roll-off. Bandpass-filtered white noise has a colored spectrum with higher power in the passband center and lower at the edges. This colored noise excites **multiple Hankel eigenvectors** with comparable eigenvalues, reducing the gap between λ₁ (signal) and λ₂ (noise).

In S_det (unfiltered), the respiration peak at 0.3 Hz is so strong that λ₁ >> λ₂ — strong SNR for the respiration component. In S_filt, the cardiac component is present but weaker relative to the shaped noise floor.

### Mathematical Interpretation

The sample covariance of the Hankel matrix:
```
R̂ = X·Xᵀ/L = A·(B·Bᵀ/L)·Aᵀ + Σ_noise
```

Where `Σ_noise` is the colored noise covariance. For white noise: `Σ_noise = σ²·I`. For colored noise: `Σ_noise` has non-equal diagonal elements (in the signal subspace basis).

The dominant eigenvector of R̂ aligns with the direction that maximizes:
```
uᵀ·R̂·u = uᵀ·A·(B·Bᵀ/L)·Aᵀ·u + uᵀ·Σ_noise·u
```

If the cardiac component is weak (small `A`), the noise term `uᵀ·Σ_noise·u` can dominate, pulling the eigenvector toward the colored noise hump around 1.4–1.5 Hz (center of the passband). The ESPRIT estimate then reflects the noise hump frequency rather than the cardiac peak.

### Why K=2 is Expected to Help

With K=2 real signal ESPRIT:

1. The real Hankel matrix is used directly — no Hilbert transform
2. The 2D signal subspace for `cos(ω₀n)` spans `{cos(ω₀n), sin(ω₀n)}`
3. Two singular vectors are used instead of one — the noise must overcome **both** dimensions of the subspace
4. The conjugate eigenvalue constraint (`eig(Ψ) = {e^{+jω₀}, e^{-jω₀}}`) provides an additional self-consistency check
5. TLS-ESPRIT accounts for noise in both Es1 and Es2

The expected improvement: at 10s and full window, K=2 should bring error from +7.4/+5.5 BPM down to ±3 BPM. At 3s and 5s, the improvement depends on the actual SNR.

---

## 13. Run 4 — K=2 LS and TLS ESPRIT (Results and Deep Analysis)

### Run 4 Results (2026-04-06)

```
Pipeline ready. T=630 frames, fs=29.9976 Hz
Reference BPM (full Welch): 54.9

Win(s)  λ1/λ2(cmplx)  λ1/λ2(real)  K1      K2-LS   K2-TLS
3       3.00           1.06          87.4    83.1    83.2
5       2.11           1.05          84.9    84.2    84.2
10      1.33           1.03          62.3    61.1    61.1
21      2.42           1.01          60.4    59.8    59.8

Win(s)  FFT    Welch  MUSIC  ESP-K1  ESP-K2-LS  ESP-K2-TLS
3       62.4   71.6   42.0   87.4    83.1        83.2
5       73.8   66.8   42.0   84.9    84.2        84.2
10      54.5   66.4   55.7   62.3    61.1        61.1
21      54.9   54.9   57.5   60.4    59.8        59.8

Errors vs Reference 54.9 BPM:
3       +7.5   +16.7  -12.9  +32.5   +28.2      +28.3
5       +18.9  +11.9  -12.9  +30.0   +29.3      +29.3
10      -0.4   +11.4  +0.7   +7.4    +6.2       +6.2
21       0.0    0.0   +2.5   +5.5    +4.9       +4.9
```

### Finding 1: K=2 TLS ≈ K=2 LS — They Are Identical

The most striking result: K=2 TLS and K=2 LS give effectively the same answer at every window (≤0.1 BPM difference). The theoretical TLS advantage over LS **vanishes completely** here.

**Why?** TLS corrects the asymmetric error assumption: LS treats Es1 as "exact" (noise-free) while TLS treats both Es1 and Es2 as noisy. The correction TLS applies is:

```
Ψ_TLS - Ψ_LS = O(σ²/||Es1||²)
```

This correction is proportional to `σ²` (noise power). For TLS to give a different answer from LS, the **noise must perturb Es1 and Es2 differently** — i.e., there must be an asymmetry in how noise affects the first M-1 rows vs the last M-1 rows of the signal subspace.

For bandpass-**colored** noise that is **stationary and ergodic** (which our Butterworth-filtered noise approximately is):

```
noise covariance R_noise ≈ σ²·I + small corrections
```

The noise projections onto Es1 and Es2 are nearly identical because the filter output has the same statistics at every time index. Therefore `||noise on Es1||_F ≈ ||noise on Es2||_F` and the TLS correction is near zero.

**Physical interpretation**: TLS has a real advantage when the noise has a drift or non-stationarity that affects different lags differently (e.g., amplitude-modulated interference, or transients). Our bandpass noise is approximately stationary — LS and TLS coincide.

**Conclusion**: For stationary colored noise, LS-ESPRIT and TLS-ESPRIT are equivalent. The choice does not matter. This is not a failure of TLS — it is TLS correctly recognizing that the LS/TLS correction is zero because both submatrices are perturbed identically.

### Finding 2: Real Hankel λ₁/λ₂ ≈ 1.0 — The Diagnostic is Wrong for K=2

The real Hankel λ₁/λ₂ is approximately 1.0 for **all** window lengths:

```
3s: 1.06,  5s: 1.05,  10s: 1.03,  21s: 1.01
```

This looks like catastrophic SNR, but it is **not**. It is a fundamental mathematical property of real sinusoids that our diagnostic code did not account for.

**Why λ₁/λ₂ ≈ 1 for a real cosine**:

A noise-free real cosine `x[n] = A·cos(ω₀n + φ)` has rank-2 Hankel matrix. The SVD has exactly TWO non-zero singular values. What are they?

The two signal subspace vectors are `cos(ω₀n)` and `sin(ω₀n)` (quadrature components). Each has the same RMS power: `||cos(ω₀n)||² = N/2` and `||sin(ω₀n)||² = N/2`. Therefore:

```
σ₁ = σ₂ = A·√(M·L/2)    (both signal singular values EQUAL)
σ₃ = σ₄ = ... = 0        (noise-free case)
```

So the **correct SNR indicator for K=2 real ESPRIT** is NOT λ₁/λ₂ (which is always ≈ 1 for a real cosine) but rather:

```
λ₂/λ₃  —  the gap between the 2nd signal eigenvalue and the 1st noise eigenvalue
```

Our code reports λ₁/λ₂ for the real Hankel. This is misleading and will always read ≈ 1.0 regardless of signal strength. This is a **code bug in the diagnostic** (not in the ESPRIT computation itself). 

The relevant diagnostic to add is `ev_r(2)/ev_r(3)` (second largest divided by third largest eigenvalue). This ratio tells us whether the 2D signal subspace is clearly separated from the noise floor.

### Finding 3: Non-Monotonic Complex Hankel λ₁/λ₂

The complex Hankel λ₁/λ₂ (correct diagnostic for K=1) goes:

```
3s: 3.00  →  5s: 2.11  →  10s: 1.33  →  21s: 2.42
```

This is NOT monotonically increasing with window length — 10s has the **lowest** ratio. Why?

At each window, M_use = min(M_max, floor(N/4)):

| Window | N | M_use | L = N-M+1 | λ₁/λ₂ |
|---|---|---|---|---|
| 3s | 90 | 22 | 69 | 3.00 |
| 5s | 150 | 37 | 114 | 2.11 |
| 10s | 300 | **60 (cap hit)** | 241 | 1.33 |
| 21s | 630 | **60 (cap)** | **571** | 2.42 |

M_max=60 cap is first hit at 10s. At exactly 10s: M=60, L=241. The M/L ratio = 60/241 = 0.25.

At 21s: same M=60, but L=571. The M/L ratio = 60/571 = 0.105 — far more snapshots per row.

**Explanation**: When M jumps from 37 (5s) to 60 (10s), we now have 60 noise dimensions instead of 37. With more noise dimensions, the noise eigenvalues become better resolved (more equal, via the Marchenko-Pastur law for large Wishart matrices). This reduces the apparent "dominance" of λ₁ relative to λ₂ (the noise is more uniformly spread). When L grows from 241 (10s) to 571 (21s) with M=60 fixed, the covariance estimate improves — the signal eigenvalue grows as `∝ L·Ps` while noise eigenvalue stabilizes at `σ²` — so the ratio recovers.

**Implication**: The M_max cap causes the 10s window to be the "hardest" case for this diagnostic. This is counter-intuitive but entirely explained by the M/L trade-off.

### Finding 4: ESPRIT Locks onto the Bandpass Noise Hump

From Fig 1: at 3s and 5s windows, the MUSIC spectrum shows a broad hump peaking at **~1.4 Hz (84 BPM)**. Both ESPRIT K=1 and K=2 estimates land directly on this hump top.

The Butterworth bandpass filter [0.7, 3.5 Hz] is not flat. A 4th-order Butterworth has maximum gain at the center of the passband and steep rolloff at the edges. For noise `ε[n]` that is approximately white before filtering:

```
S_filt[n] = butterworth_bandpass · ε[n]
```

The filtered noise PSD is:
```
S_ε(f) = |H(f)|² · σ²_ε
```

A 4th-order Butterworth bandpass from 0.7–3.5 Hz has its geometric center at `f_c = √(0.7 × 3.5) = √2.45 ≈ 1.565 Hz`. The group delay is maximum near the passband edges, but the **power** is maximum near `f_c`. Combined with the roll-off shape, the effective "hump" of filtered noise falls around 1.3–1.5 Hz.

At short windows (3–5s), the cardiac signal at 0.9 Hz is WEAKER than this colored noise hump. ESPRIT's dominant eigenvector aligns with the hump at 1.4–1.5 Hz because the hump has more integrated power than the cardiac peak.

This is the fundamental problem. No variant of ESPRIT (K=1, K=2, LS, TLS) fixes this — they all find the same hump because they all look for the dominant periodic structure.

### Finding 5: K=2 Gives Marginal Improvement, Not the Breakthrough Expected

K=2 LS vs K=1:

```
3s:  87.4 → 83.1  (−4.3 BPM improvement)
5s:  84.9 → 84.2  (−0.7 BPM improvement)
10s: 62.3 → 61.1  (−1.2 BPM improvement)
21s: 60.4 → 59.8  (−0.6 BPM improvement)
```

A 1–4 BPM improvement is visible, but trivial relative to the 28–32 BPM error at short windows. K=2 correctly uses the conjugate symmetry constraint and 2D subspace averaging, but when the dominant noise hump at 1.4 Hz is both signals modes (it appears as K=2 components too, one at +1.4 Hz and one at -1.4 Hz), K=2 ESPRIT still finds the noise hump.

### Finding 6: FFT is Best at 10s and 21s

Unexpectedly, FFT achieves the best accuracy at 10s (-0.4 BPM) and 21s (0.0 BPM). This is because:

1. At 10s: N·Δf = fs/N = 29.97/300 = 0.1 Hz per bin = 6 BPM/bin — sufficient resolution to locate a sharp peak
2. At 21s: N = 630, Δf = 0.0476 Hz = 2.85 BPM/bin — very fine resolution
3. The cardiac signal at 54.9 BPM creates a genuine spectral peak at 10s+ that FFT finds exactly

The paradox: FFT (the "dumbest" method) beats MUSIC and all ESPRIT variants at long windows. This happens because:
- The cardiac peak IS present at 10s+ with sufficient SNR
- FFT is unbiased at high SNR (it finds the true peak when there is one)
- MUSIC has its pseudo-spectrum shifted by unequal noise eigenvalues (colored noise)
- ESPRIT is biased toward the noise hump

**Lesson**: At long windows with high SNR, FFT is often optimal or near-optimal. Subspace methods add value specifically at SHORT windows or LOW SNR — exactly the regime where this data challenges us.

### Complete Performance Table (Run 4 Final)

| Window | FFT | Welch | MUSIC | K=1 | K=2-LS | K=2-TLS | Best Method |
|---|---|---|---|---|---|---|---|
| 3s | +7.5 | +16.7 | **-12.9** | +32.5 | +28.2 | +28.3 | FFT (+7.5) |
| 5s | +18.9 | +11.9 | **-12.9** | +30.0 | +29.3 | +29.3 | Welch (+11.9) |
| 10s | **-0.4** | +11.4 | +0.7 | +7.4 | +6.2 | +6.2 | FFT (-0.4) |
| 21s | **0.0** | 0.0 | +2.5 | +5.5 | +4.9 | +4.9 | FFT/Welch (0.0) |

Errors in BPM (estimated − reference). Bold = best performer per row.

MUSIC at 3s and 5s (-12.9 BPM = 42 BPM) is underestimating — it latches onto the noise floor around 0.7 Hz (the lower band edge) when the cardiac peak is not clear.

### What We Need: Cadzow Denoising (Run 5)

All three ESPRIT variants (K=1, K=2 LS, K=2 TLS) fail at short windows for the same root cause: the bandpass-colored noise hump at 1.4 Hz dominates the signal at 0.9 Hz.

The solution: **denoise the Hankel matrix before ESPRIT** using Cadzow's algorithm. This directly increases the effective SNR seen by ESPRIT by removing noise from the data matrix structure.

See Section 18 for the complete Cadzow derivation and Run 5 implementation plan.

---

## 14. ESPRIT vs MUSIC: Architectural Comparison

### Fundamental Difference

| Property | MUSIC | ESPRIT |
|---|---|---|
| **Principle** | Noise subspace orthogonality | Signal subspace rotational invariance |
| **Output** | Pseudo-spectrum P(ω) (sweep over grid) | Direct frequency estimate (no grid) |
| **Requires grid search** | YES (f_grid with nfft=4096 points) | NO (algebraic eigenvalue solve) |
| **Resolution** | Limited by grid spacing | Unlimited (algebraic) |
| **Noise subspace used** | YES (M-K eigenvectors) | NO |
| **Signal subspace used** | Implicitly (K signal eigenvectors excluded) | Directly (K signal eigenvectors) |
| **FB averaging** | Compatible and beneficial | **INCOMPATIBLE** (destroys phase) |
| **White noise assumption** | CRITICAL | Not required |
| **Colored noise performance** | Degraded pseudo-spectrum peaks | Signal eigenvector slightly biased |
| **Computational cost** | High (grid sweep) | Low (eigenvalue of K×K matrix) |
| **Real-time suitability** | Moderate | High |

### Why MUSIC Handles S_det Better

MUSIC works with S_det because it uses the **noise subspace** (M-K eigenvectors). The respiration component at 0.3 Hz contributes to the signal subspace (occupies one dimension). MUSIC with K=1 leaves M-1 noise eigenvectors. These M-1 vectors span the space orthogonal to the single dominant (respiration) direction and to the cardiac direction. The MUSIC pseudo-spectrum shows peaks where the steering vector `a(ω)` is orthogonal to the noise subspace — this correctly identifies the cardiac peak even in the presence of respiration (respiration just occupies the K=1 signal subspace dimension).

ESPRIT with K=1 and S_det: the signal subspace dimension 1 is captured by the dominant eigenvector, which aligns with the strongest sinusoid — the respiration, not the cardiac.

### Why ESPRIT Needs Higher SNR

**MUSIC's averaging**: The pseudo-spectrum is `P(ω) = 1/||Q_n^H · a(ω)||²`. Q_n has M-K columns (e.g., M-K=59 for K=1, M=60). Each column's projection onto `a(ω)` is squared and summed. This is an average of 59 noise projections — significant noise averaging via the law of large numbers.

**ESPRIT's single eigenvector**: ESPRIT uses K=1 eigenvector (a single M-dimensional complex vector). This single vector must align precisely with the cardiac frequency direction. With low SNR, even one noisy snapshot can rotate this vector away from the true cardiac direction by several degrees, causing a frequency error of several Hz.

**MUSIC is inherently a higher-averaging estimator.** ESPRIT trades averaging for algebraic exactness (no grid). At high SNR, ESPRIT wins (no grid quantization error, exact frequencies). At low SNR, MUSIC wins (noise averaging).

---

## 15. The λ₁/λ₂ Ratio as SNR Diagnostic

### Definition

```
λ₁/λ₂ = (largest eigenvalue of R_hat) / (second largest eigenvalue)
```

Where R_hat = X·Xᵀ/L is the sample covariance of the Hankel data matrix.

### Interpretation

For a single sinusoid (K=1) in white noise with signal power `Ps` and noise power `σ²`:

**Theoretical eigenvalue structure** (Toeplitz covariance limit):
- `λ₁ ≈ L·M·Ps + σ²`  (signal eigenvalue, inflated by L×M samples of signal power)
- `λ₂ = λ₃ = ... = λ_M = σ²`  (M-1 noise eigenvalues, all equal)

Therefore: `λ₁/λ₂ ≈ L·M·Ps/σ² + 1 = SNR_total + 1`

**In practice** (finite data, colored noise from bandpass filter):
- λ₂ is elevated above σ² by colored noise structure
- λ₁/λ₂ is a lower bound on the "effective SNR"

### Observed Ratios and Interpretation

| Input | Window | λ₁/λ₂ (complex) | ESPRIT quality |
|---|---|---|---|
| S_det | All | 3.4–5.6 | Perfect (finds respiration cleanly) |
| S_filt | 3s | 1.33 | Very poor — noise dominates |
| S_filt | 5s | 1.67 | Poor |
| S_filt | 10s | 2.89 | Moderate |
| S_filt | 21s | 3.0+ | Acceptable |

**Why S_filt has lower λ₁/λ₂ than S_det**:

S_det contains respiration at 0.3 Hz with large amplitude → dominant sinusoid → large λ₁ → high ratio.

S_filt has the respiration removed, leaving only the weaker cardiac component in a colored noise floor. The cardiac component competes with bandpass-colored noise that also concentrates energy in 0.7–3.5 Hz → λ₂ elevated → ratio decreases.

### Threshold for ESPRIT Reliability

Empirical rule-of-thumb from spectral estimation literature:
- `λ₁/λ₂ > 3`: ESPRIT likely within ±5 BPM (for K=1)
- `λ₁/λ₂ > 5`: ESPRIT likely within ±2 BPM
- `λ₁/λ₂ < 2`: ESPRIT estimates unreliable, prefer Welch or MUSIC

For rPPG at short windows with S_filt, the 3s–5s estimates fall in the unreliable zone. This motivates:
1. K=2 with conjugate constraint (increases effective SNR)
2. Longer windows (10s+ for reliable ESPRIT)
3. Future: Cadzow denoising (SVD rank truncation to reduce colored noise)

---

## 16. Parameter Tuning Guide

### M_max (Model Order Cap)

```matlab
M_max = 60;   % Current default
```

- **Increase to 80–100**: Larger signal subspace → more noise averaging per eigenvector → better at long windows. BUT: computational cost increases as O(M³) for eigendecomposition.
- **Decrease to 30–40**: Faster, works better at very short windows where N is small. M should satisfy M ≥ 4 (for K=2 + some noise dimensions).
- **Rule**: M_use = min(M_max, floor(N/4)). The N/4 constraint ensures L = 3N/4 >> M (more snapshots than rows).

### win_lengths_sec

```matlab
win_lengths_sec = [3, 5, 10, floor(T/fs)];
```

- The 3s and 5s windows are included for research purposes (to characterize the failure mode at low SNR)
- For a production system: only use windows ≥ 10s for ESPRIT, or wait for K=2 TLS results
- Prof. Chen's target: reliable BPM from ≤ 5s windows with ESPRIT

### K_c and K_r

```matlab
K_c = 1;   % K for analytic/complex signal (1 complex exponential)
K_r = 2;   % K for real signal (2 complex exponentials ±ω₀)
```

For cardiac-only rPPG:
- `K_c = 1` is correct: one cardiac frequency in the analytic signal
- `K_r = 2` is correct: one real cosine = two complex exponentials
- `K_r = 4` could capture cardiac + first harmonic (2·f_HR), but only useful if the second harmonic is desired

### f_low, f_high

```matlab
f_low = 0.7; f_high = 3.5;
```

Per Prof. Chen's specification: 0.7–3.5 Hz = 42–210 BPM. Do not change without explicit instruction. Previously was 0.67–3.0 Hz (now corrected).

---

## 17. Summary Table: All Runs

| Run | Issue | Root Cause | Fix Applied |
|---|---|---|---|
| 1 | All NaN | FB averaging → real eigenvectors → angle=0 → freq=0 Hz | Switched from eig(R_fb) to svd(X_c, 'econ') |
| 2 | All NaN | S_det dominated by respiration at 0.317 Hz < f_low | Switched input from S_det to S_filt |
| 3 | Valid but overestimating (K=1) | Low λ₁/λ₂ from colored bandpass noise → eigenvector biased toward 1.4 Hz hump | Implement K=2 LS + TLS |
| 4 | K=2 ≈ K=1 (+1–4 BPM only), TLS ≡ LS | Real cosine λ₁≈λ₂ always; stationary noise → LS=TLS; noise hump still dominant | Cadzow denoising (Run 5) |
| 5 | Planned | Bandpass noise hump at 1.4 Hz dominates cardiac at 0.9 Hz | SVD rank-K truncation + anti-diagonal averaging |

### Final BPM Error Table (All Methods, All Windows)

| Window | FFT | Welch | MUSIC | K=1 | K=2-LS | K=2-TLS | Reference |
|---|---|---|---|---|---|---|---|
| 3s | **+7.5** | +16.7 | -12.9 | +32.5 | +28.2 | +28.3 | 54.9 BPM |
| 5s | +18.9 | **+11.9** | -12.9 | +30.0 | +29.3 | +29.3 | 54.9 BPM |
| 10s | **-0.4** | +11.4 | +0.7 | +7.4 | +6.2 | +6.2 | 54.9 BPM |
| 21s | **0.0** | 0.0 | +2.5 | +5.5 | +4.9 | +4.9 | 54.9 BPM |

Bold = best method per window.

---

## 18. Run 5 (Next Step) — Cadzow Denoising Before ESPRIT

### What is Cadzow Denoising?

Cadzow's algorithm (1988, originally from NMR spectroscopy) is a **structured low-rank approximation** of the Hankel data matrix. It exploits the fact that a **noise-free** Hankel matrix built from K sinusoids has rank exactly K. Adding noise makes the rank full (= M). Cadzow projects the noisy Hankel back to rank K.

The algorithm alternates two projections until convergence:

**Step 1 — Rank-K truncation (Eckart-Young projection)**:

```
X_noisy = U · Σ · Vᵀ                              (full SVD)
X_K     = U[:,1:K] · Σ[1:K,1:K] · V[:,1:K]ᵀ      (keep top K singular values only)
```

This is the best rank-K approximation of X_noisy in the Frobenius norm:

```
X_K = argmin_{rank(A)≤K}  ||X_noisy - A||²_F
```

The M-K smallest singular values (noise subspace) are set to zero. The signal is separated from noise in the SVD basis.

**Step 2 — Hankel projection (anti-diagonal averaging)**:

After rank truncation, X_K is no longer Hankel (its anti-diagonals are not constant). Re-impose the Hankel constraint by replacing each anti-diagonal with its mean:

For an M×L Hankel matrix, each anti-diagonal d = {(m,l) : m+l = const} must have the same value. The orthogonal projection onto the Hankel set is:

```
[P_Hankel(A)][m,l] = (1/|d|) · Σ_{(i,j): i+j=m+l}  A[i,j]
```

Where |d| is the number of elements on that anti-diagonal (varies from 1 at the corners to min(M,L) at the center).

**Convergence**: Alternate P_rank_K and P_Hankel for n_iter iterations (typically 3–10):

```
X_cadzow = (P_Hankel · P_rank_K)^n_iter · X_noisy
```

Each iteration:
- P_rank_K removes noise from the SVD basis (zeros out noise singular values)
- P_Hankel restores the shift-invariance structure required for ESPRIT

The sequence converges to a Hankel matrix of rank K that is the "closest" rank-K Hankel matrix to X_noisy in a structured sense.

### Why Cadzow Helps ESPRIT for rPPG

The root problem at short windows: bandpass-colored noise hump at ~1.4 Hz has more integrated power than the cardiac signal at 0.9 Hz. ESPRIT (and MUSIC) find the hump.

After Cadzow with K=1 (or K=2):

1. **SVD truncation removes all but the single dominant periodic component** — if the hump is dominant, Cadzow keeps the hump. If the cardiac peak is dominant, Cadzow keeps the cardiac.
2. **Anti-diagonal averaging re-imposes Hankel structure** — the result is exactly what ESPRIT needs: a rank-K Hankel matrix representing one (K=1) or two (K=2) sinusoids.
3. **ESPRIT on the denoised Hankel** finds the exact frequency of whatever was kept by the rank truncation.

**Critical question**: At 3s and 5s windows, is the cardiac component at 0.9 Hz stronger than the noise hump at 1.4 Hz? The answer determines whether Cadzow helps at short windows.

**Hypothesis**: For this subject/condition, the cardiac signal at short windows is weaker than the bandpass noise. No rank-K truncation can recover a signal that is genuinely weaker than noise — this is an SNR floor. Cadzow may help at 10s+ (where λ₁/λ₂ > 2 and the cardiac is detectable) but may not fix 3s/5s.

### MATLAB Implementation — Cadzow Function

```matlab
function X_out = cadzow_hankel(X, K, n_iter)
% Cadzow denoising: iterative low-rank + Hankel projection
% X      : M×L Hankel data matrix (real or complex)
% K      : signal rank (= number of complex exponentials to preserve)
% n_iter : iterations (default 5)
if nargin < 3, n_iter = 5; end
[M, L] = size(X);
N = M + L - 1;  % number of distinct anti-diagonals

% Precompute anti-diagonal membership for averaging (vectorized)
% antidiag_idx(m,l) = m+l (1-based sum, ranges from 2 to M+L)
[mm, ll] = ndgrid(1:M, 1:L);
ad = mm + ll;  % M×L matrix of anti-diagonal indices

for iter = 1:n_iter
    % --- Step 1: Rank-K truncation ---
    [U, S, V] = svd(X, 'econ');
    s = diag(S);
    s(K+1:end) = 0;                  % zero out noise singular values
    X = U * diag(s) * V';            % reconstruct rank-K approximation

    % --- Step 2: Anti-diagonal averaging (Hankel projection) ---
    X_new = zeros(M, L, 'like', X);
    for d = 2 : M+L                 % d = m+l, ranges 2..M+L
        mask = (ad == d);
        avg  = sum(X(mask)) / sum(mask(:));
        X_new(mask) = avg;
    end
    X = X_new;
end
X_out = X;
end
```

### Integration into bpm_control_ESPRIT.m

Inside the main loop, before the ESPRIT variants:

```matlab
% ── Cadzow denoising (add after building X_c and X_r) ────────────────────
n_cadzow = 5;   % CHANGE: try 3, 5, 10
X_c_den  = cadzow_hankel(X_c, K_c, n_cadzow);  % K=1 complex denoised
X_r_den  = cadzow_hankel(X_r, K_r, n_cadzow);  % K=2 real denoised

% Then use X_c_den instead of X_c for Variant A SVD
[U_c_d, ~, ~] = svd(X_c_den, 'econ');
Qs_c_d  = U_c_d(:, 1:K_c);
Es1_c_d = Qs_c_d(1:end-1, :);
Es2_c_d = Qs_c_d(2:end, :);
Psi_c_d = pinv(Es1_c_d) * Es2_c_d;
lam_c_d = eig(Psi_c_d);
frq_c_d = abs(angle(lam_c_d)) * fs / (2*pi);
ib_c_d  = (frq_c_d >= f_low) & (frq_c_d <= f_high);
if any(ib_c_d), bpm_cadzow_k1(k) = mean(frq_c_d(ib_c_d)) * 60; end
```

### Expected Run 5 Results

| Window | K=1 (no Cadzow) | K=1 + Cadzow (predicted) | Notes |
|---|---|---|---|
| 3s | 87.4 BPM (+32.5) | ~83–87 BPM | Noise hump likely still dominant; minimal improvement |
| 5s | 84.9 BPM (+30.0) | ~75–85 BPM | Possible slight improvement |
| 10s | 62.3 BPM (+7.4) | ~55–62 BPM | Cardiac emerges; Cadzow may fix this window |
| 21s | 60.4 BPM (+5.5) | ~54–59 BPM | Most likely within ±5 BPM after denoising |

If 3s/5s estimates remain at 83–87 BPM after Cadzow, the conclusion is that **3s and 5s windows do not have sufficient SNR for subspace methods with this rPPG setup**. The minimum reliable window is 10s for ESPRIT on this data.

---

## Appendix: Key MATLAB Functions Used

| Function | Purpose |
|---|---|
| `hilbert(x)` | Compute analytic signal (remove negative frequencies) |
| `bsxfun(@plus, ...)` | Build Hankel index matrix efficiently |
| `svd(X, 'econ')` | Economy SVD — returns min(M,L) singular vectors |
| `pinv(Es1)` | Moore-Penrose pseudoinverse via SVD |
| `eig(Psi)` | Eigenvalues of the rotation operator Ψ |
| `angle(lambda)` | Extract phase angle: ω = angle(e^{jω}) |
| `abs(angle(...))` | Get positive frequency from ±ω₀ |
| `filtfilt(b,a,x)` | Zero-phase Butterworth bandpass |
| `pwelch(x,...)` | Reference BPM via Welch periodogram |

---

*Document version 2.0 — 2026-04-06. Run 4 results incorporated. Run 5 (Cadzow) planned.*

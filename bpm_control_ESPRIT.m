clear; close all; clc;

IMG_PATH = '/Users/devanshbajwala/Documents/VS Code WS/PyQt/DocBOT/rPPG Controls/Photo on 4-5-26 at 10.31.jpg';
VID_PATH = '/Users/devanshbajwala/Documents/VS Code WS/PyQt/DocBOT/rPPG Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/recording_2026-04-07T22-04-35Z.mov';

img = imread(IMG_PATH);
vid = VideoReader(VID_PATH);
fs  = vid.FrameRate;
[H_img, W_img, ~] = size(img);

% ── Face detection ────────────────────────────────────────────────────────
vid_tmp     = VideoReader(VID_PATH);
first_frame = readFrame(vid_tmp);
clear vid_tmp;

try
    detector  = vision.CascadeObjectDetector();
    all_boxes = step(detector, first_frame);
catch
    all_boxes = [];
end

if isempty(all_boxes)
    x1 = floor(W_img*0.25); y1 = floor(H_img*0.05);
    x2 = floor(W_img*0.75); y2 = floor(H_img*0.75);
else
    [~, bbox_idx] = max(all_boxes(:,3) .* all_boxes(:,4));
    bbox = all_boxes(bbox_idx,:);
    x1 = max(bbox(1), 1);
    y1 = max(bbox(2), 1);
    x2 = min(bbox(1)+bbox(3)-1, W_img);
    y2 = min(bbox(2)+bbox(4)-1, H_img);
end

% ── RGB extraction ────────────────────────────────────────────────────────
vid2 = VideoReader(VID_PATH);
R_t = []; G_t = []; B_t = [];

while hasFrame(vid2)
    frame = readFrame(vid2);
    fc    = frame(y1:y2, x1:x2, :);
    fcd   = double(fc);

    Yf  =  0.299*fcd(:,:,1)    + 0.587*fcd(:,:,2)    + 0.114*fcd(:,:,3);
    Cbf = -0.168736*fcd(:,:,1) - 0.331264*fcd(:,:,2) + 0.5*fcd(:,:,3)      + 128;
    Crf =  0.5*fcd(:,:,1)      - 0.418688*fcd(:,:,2) - 0.081312*fcd(:,:,3) + 128;
    Mf  = (Cbf>=77)&(Cbf<=127)&(Crf>=133)&(Crf<=173)&(Yf>40);

    if sum(Mf(:)) < 50; continue; end

    pix = reshape(fcd, [], 3);
    msk = Mf(:);
    R_t(end+1) = mean(pix(msk,1)); %#ok<SAGROW>
    G_t(end+1) = mean(pix(msk,2)); %#ok<SAGROW>
    B_t(end+1) = mean(pix(msk,3)); %#ok<SAGROW>
end

T      = length(R_t);
t_axis = (0:T-1) / fs;

% ── CHROM ─────────────────────────────────────────────────────────────────
R_n = R_t / mean(R_t);
G_n = G_t / mean(G_t);
B_n = B_t / mean(B_t);

Xs    = 3*R_n - 2*G_n;
Ys    = 1.5*R_n + G_n - 1.5*B_n;
alpha = std(Xs) / std(Ys);
S     = Xs - alpha * Ys;

% ── Detrend ───────────────────────────────────────────────────────────────
t_vec  = (1:T)';
coeffs = [t_vec, ones(T,1)] \ S(:);
S_det  = S(:) - (coeffs(1)*t_vec + coeffs(2));

fprintf('Pipeline ready. T=%d frames, fs=%.4f Hz\n', T, fs);

% ── Butterworth bandpass ───────────────────────────────────────────────────
f_low = 0.7; f_high = 3.5;
Wn = [f_low f_high] / (fs/2);
[b, a] = butter(4, Wn, 'bandpass');
S_filt = filtfilt(b, a, S_det);

% =========================================================================
%  ESPRIT — THREE VARIANTS compared against MUSIC, FFT, Welch
%
%  Variant A: K=1, Analytic signal (hilbert), LS-ESPRIT via SVD(X_complex)
%    Signal model: x_c[n] = A*e^{j(w0*n + phi)} + noise  (K=1 complex exp.)
%    Input: S_filt → hilbert → complex Hankel X_c
%    Subspace: U_svd(:,1:1) from SVD(X_c)
%    Issue at SNR < 10 dB: dominant singular vector biased toward
%    the colored-noise hump rather than the cardiac peak.
%
%  Variant B: K=2, Real signal, LS-ESPRIT via SVD(X_real)
%    Signal model: x[n] = A*cos(w0*n+phi) + noise
%                        = (A/2)*e^{+jw0*n} + (A/2)*e^{-jw0*n} + noise
%    So K=2 complex exponentials in the real signal.
%    Real Hankel X_r: real M×L matrix. SVD gives REAL singular vectors.
%    The K=2 signal subspace spanned by {cos(w0*n), sin(w0*n)} vectors.
%    Psi is a 2×2 REAL matrix. eig(Psi) = {e^{+jw0}, e^{-jw0}}.
%    Both eigenvalues give abs(angle) = w0 → frequency estimate.
%    Advantage: conjugate symmetry constraint improves robustness at low SNR.
%
%  Variant C: K=2, Real signal, TLS-ESPRIT (Total Least Squares)
%    LS-ESPRIT minimises ||Es2 - Es1*Psi||_F (errors only in Es2).
%    TLS-ESPRIT minimises perturbations to BOTH Es1 and Es2 jointly.
%    More robust when both submatrices are noisy (which is always true
%    at low SNR). Uses SVD of the augmented matrix C = [Es1 | Es2].
%
%  CHANGE: M_max — try 40, 60, 80
%  CHANGE: win_lengths_sec — window lengths to evaluate
% =========================================================================
M_max = 60;
K_c   = 1;   % K for complex/analytic signal (Variant A)
K_r   = 2;   % K for real signal (Variants B and C)
win_lengths_sec = [3, 5, 10, floor(T/fs)];

nfft   = 4096;
f_grid = linspace(f_low, f_high, nfft);

% ── Reference BPM: full signal Welch ──────────────────────────────────────
[P_ref, f_ref] = pwelch(S_filt, hann(T), floor(T/2), nfft, fs);
band_ref = (f_ref >= f_low) & (f_ref <= f_high);
[~, ri] = max(P_ref(band_ref));
bpm_ref = f_ref(band_ref); bpm_ref = bpm_ref(ri) * 60;
fprintf('Reference BPM (full Welch): %.1f\n\n', bpm_ref);

% ── MUSIC reference (from bpm_estimate_MUSIC.m, S_det input, FB-averaged R) ──
bpm_music = zeros(1, numel(win_lengths_sec));
for k = 1:numel(win_lengths_sec)
    N_m  = min(round(win_lengths_sec(k) * fs), T);
    xm   = hilbert(S_det(1:N_m));
    M_m  = max(min(M_max, floor(N_m/4)), K_c+2);
    L_m  = N_m - M_m + 1;
    im   = bsxfun(@plus, (1:M_m)', 0:L_m-1);
    Xm   = xm(im);
    Rm   = (Xm*Xm')/L_m;
    Jm   = fliplr(eye(M_m));
    Rm   = 0.5*(Rm + Jm*conj(Rm)*Jm);
    [Vm,Dm] = eig(Rm);
    [~,sm]  = sort(real(diag(Dm)),'descend');
    Qnm  = Vm(:, sm(K_c+1:end));
    Am   = exp(1j*(0:M_m-1)'*(2*pi*f_grid/fs));
    Pm   = 1./sum(abs(Qnm'*Am).^2, 1);
    [~,pm] = max(Pm);
    bpm_music(k) = f_grid(pm)*60;
end

% Storage
bpm_fft       = zeros(1, numel(win_lengths_sec));
bpm_welch     = zeros(1, numel(win_lengths_sec));
bpm_esp_k1    = nan(1,   numel(win_lengths_sec));   % Variant A
bpm_esp_k2_ls = nan(1,   numel(win_lengths_sec));   % Variant B
bpm_esp_k2_tls= nan(1,   numel(win_lengths_sec));   % Variant C
eig_cmplx     = cell(1,  numel(win_lengths_sec));   % eigenvalues of complex R
eig_real      = cell(1,  numel(win_lengths_sec));   % eigenvalues of real R

% ── Main loop ─────────────────────────────────────────────────────────────
fprintf('%-8s  %-10s  %-10s  %-12s  %-12s\n', ...
    'Win(s)', 'λ1/λ2(cmplx)', 'λ1/λ2(real)', 'ESP-K1', 'ESP-K2-LS', 'TLS');
fprintf('%s\n', repmat('-',1,70));

for k = 1:numel(win_lengths_sec)
    N = min(round(win_lengths_sec(k) * fs), T);
    x = S_filt(1:N);   % real bandpass signal

    % FFT
    Xf   = abs(fft(x.*hann(N), nfft));
    Xf   = Xf(1:nfft/2+1);
    ff   = (0:nfft/2)*fs/nfft;
    bf   = (ff>=f_low)&(ff<=f_high);
    [~,pf] = max(Xf(bf)); fbs = ff(bf);
    bpm_fft(k) = fbs(pf)*60;

    % Welch
    npw = max(floor(N/2), round(2*fs));
    [Pw,fw] = pwelch(x, hann(npw), floor(npw/2), nfft, fs);
    bw = (fw>=f_low)&(fw<=f_high);
    [~,pw] = max(Pw(bw)); fbw = fw(bw);
    bpm_welch(k) = fbw(pw)*60;

    % Shared M, L, index
    M_use = max(min(M_max, floor(N/4)), K_r+2);

    % ── Variant A: K=1 analytic (complex Hankel) ──────────────────────────
    % Build complex Hankel from hilbert(S_filt)
    x_c   = hilbert(x);
    L_c   = N - M_use + 1;
    ic    = bsxfun(@plus, (1:M_use)', 0:L_c-1);
    X_c   = x_c(ic);                             % M_use × L_c  complex
    R_c   = (X_c * X_c') / L_c;
    ev_c  = sort(real(eig(R_c)), 'descend');
    eig_cmplx{k} = ev_c(1:min(M_use,20));

    [U_c, ~, ~] = svd(X_c, 'econ');
    Qs_c  = U_c(:, 1:K_c);
    Es1_c = Qs_c(1:end-1, :);
    Es2_c = Qs_c(2:end,   :);
    Psi_c = pinv(Es1_c) * Es2_c;
    lam_c = eig(Psi_c);
    frq_c = abs(angle(lam_c)) * fs / (2*pi);
    ib_c  = (frq_c >= f_low) & (frq_c <= f_high);
    if any(ib_c), bpm_esp_k1(k) = mean(frq_c(ib_c)) * 60; end

    % ── Variant B: K=2 real LS-ESPRIT (real Hankel) ───────────────────────
    % No hilbert needed. Real signal has K=2 components: +w0 and -w0.
    % Real Hankel X_r → real SVD → real singular vectors Qs_r.
    % Psi_ls = pinv(Es1_r)*Es2_r is a 2×2 REAL matrix.
    % eig(Psi_ls) = conjugate pair {e^{+jw0}, e^{-jw0}}.
    L_r   = N - M_use + 1;
    ir    = bsxfun(@plus, (1:M_use)', 0:L_r-1);
    X_r   = x(ir);                               % M_use × L_r  REAL
    R_r   = (X_r * X_r') / L_r;
    ev_r  = sort(real(eig(R_r)), 'descend');
    eig_real{k} = ev_r(1:min(M_use,20));

    [U_r, ~, ~] = svd(X_r, 'econ');
    Qs_r  = U_r(:, 1:K_r);                       % M_use × 2  real
    Es1_r = Qs_r(1:end-1, :);                    % (M_use-1) × 2
    Es2_r = Qs_r(2:end,   :);                    % (M_use-1) × 2

    Psi_ls  = pinv(Es1_r) * Es2_r;               % 2×2 real
    lam_ls  = eig(Psi_ls);                       % {e^{+jw0}, e^{-jw0}}
    frq_ls  = abs(angle(lam_ls)) * fs / (2*pi);
    ib_ls   = (frq_ls >= f_low) & (frq_ls <= f_high);
    if any(ib_ls), bpm_esp_k2_ls(k) = mean(frq_ls(ib_ls)) * 60; end

    % ── Variant C: K=2 real TLS-ESPRIT ────────────────────────────────────
    % Form C = [Es1_r | Es2_r], shape (M_use-1) × (2*K_r) = (M_use-1) × 4
    % SVD: C = U*S*V^T  →  V is 4×4
    % Partition V into 2×2 blocks:
    %   V = [V11 V12]   rows 1:K_r correspond to Es1 columns
    %       [V21 V22]   rows K_r+1:2*K_r correspond to Es2 columns
    % TLS solution:  Psi_tls = -V12 * inv(V22)
    % Eig(Psi_tls) = {e^{+jw0}, e^{-jw0}}  (same as LS but more robust)
    C_tls = [Es1_r, Es2_r];                      % (M_use-1) × 4
    [~, ~, Vc] = svd(C_tls, 'econ');             % Vc is 4×4
    V12   = Vc(1:K_r,       K_r+1:2*K_r);        % 2×2
    V22   = Vc(K_r+1:2*K_r, K_r+1:2*K_r);        % 2×2
    Psi_tls  = -V12 / V22;                       % 2×2
    lam_tls  = eig(Psi_tls);
    frq_tls  = abs(angle(lam_tls)) * fs / (2*pi);
    ib_tls   = (frq_tls >= f_low) & (frq_tls <= f_high);
    if any(ib_tls), bpm_esp_k2_tls(k) = mean(frq_tls(ib_tls)) * 60; end

    % NOTE: For real Hankel, λ1/λ2 ≈ 1 always (cos(wn) has equal power in
    % cos and sin components). The correct SNR indicator is λ2/λ3 (signal
    % subspace edge vs noise floor). Report both for completeness.
    fprintf('%-8d  λcmplx=%.2f  λreal(1/2)=%.2f  λreal(2/3)=%.2f  K1=%.1f  K2-LS=%.1f  TLS=%.1f\n', ...
        win_lengths_sec(k), ev_c(1)/ev_c(2), ev_r(1)/ev_r(2), ev_r(2)/ev_r(3), ...
        bpm_esp_k1(k), bpm_esp_k2_ls(k), bpm_esp_k2_tls(k));
end

% ── Fig 1 — ESPRIT estimates overlaid on MUSIC spectrum context ────────────
figure('Name','Fig 1 — All ESPRIT Variants vs Window Length');
for k = 1:numel(win_lengths_sec)
    N   = min(round(win_lengths_sec(k) * fs), T);
    x_c = hilbert(S_filt(1:N));
    M_m = max(min(M_max, floor(N/4)), K_c+2);
    L_m = N - M_m + 1;
    im  = bsxfun(@plus, (1:M_m)', 0:L_m-1);
    Xm  = x_c(im);
    Rm  = (Xm*Xm')/L_m;
    Jm  = fliplr(eye(M_m));
    Rm  = 0.5*(Rm + Jm*conj(Rm)*Jm);
    [Vm,Dm] = eig(Rm);
    [~,sm]  = sort(real(diag(Dm)),'descend');
    Qnm = Vm(:, sm(K_c+1:end));
    Am  = exp(1j*(0:M_m-1)'*(2*pi*f_grid/fs));
    Pm  = 1./sum(abs(Qnm'*Am).^2,1);
    Pm_db = 10*log10(Pm/max(Pm));
    [~,pm_ctx] = max(Pm);

    subplot(numel(win_lengths_sec),1,k);
    plot(f_grid, Pm_db, 'g', 'LineWidth',1.0); hold on;
    xline(f_grid(pm_ctx), 'g:', sprintf('MUSIC %.1f', bpm_music(k)), ...
        'LabelVerticalAlignment','bottom','LineWidth',1);
    xline(bpm_ref/60, 'k--', sprintf('Ref %.1f', bpm_ref), 'LineWidth',1.5);
    if ~isnan(bpm_esp_k1(k))
        xline(bpm_esp_k1(k)/60, 'm-', sprintf('K1 %.1f', bpm_esp_k1(k)), ...
            'LabelVerticalAlignment','top','LineWidth',1.5);
    end
    if ~isnan(bpm_esp_k2_ls(k))
        xline(bpm_esp_k2_ls(k)/60, 'c-', sprintf('K2-LS %.1f', bpm_esp_k2_ls(k)), ...
            'LabelVerticalAlignment','top','LineWidth',2.0);
    end
    if ~isnan(bpm_esp_k2_tls(k))
        xline(bpm_esp_k2_tls(k)/60, 'y-', sprintf('K2-TLS %.1f', bpm_esp_k2_tls(k)), ...
            'LabelVerticalAlignment','top','LineWidth',2.0);
    end
    xlim([0.5 3.5]); ylim([-40 2]); ylabel('dB (norm)');
    ev_c = eig_cmplx{k}; ev_r = eig_real{k};
    % cmplx: λ1/λ2 is SNR indicator (K=1, one signal eigenvalue)
    % real:  λ2/λ3 is SNR indicator (K=2, two equal signal eigenvalues; λ1/λ2≈1 always)
    title(sprintf('win=%ds  M=%d  cmplx λ1/λ2=%.2f  real λ2/λ3=%.2f', ...
        win_lengths_sec(k), M_use, ev_c(1)/ev_c(2), ev_r(2)/ev_r(3)));
    grid on;
end
xlabel('Frequency (Hz)');

% ── Fig 2 — BPM accuracy: all methods ─────────────────────────────────────
figure('Name','Fig 2 — BPM Accuracy: All Methods');
plot(win_lengths_sec, bpm_fft,        'b-o',  'LineWidth',1.5,'MarkerSize',9); hold on;
plot(win_lengths_sec, bpm_welch,      'r-s',  'LineWidth',1.5,'MarkerSize',9);
plot(win_lengths_sec, bpm_music,      'g-^',  'LineWidth',1.5,'MarkerSize',9);
plot(win_lengths_sec, bpm_esp_k1,     'm-d',  'LineWidth',2.0,'MarkerSize',9);
plot(win_lengths_sec, bpm_esp_k2_ls,  'c-p',  'LineWidth',2.0,'MarkerSize',10);
plot(win_lengths_sec, bpm_esp_k2_tls, 'y-h',  'LineWidth',2.0,'MarkerSize',10);
yline(bpm_ref,   'k--', sprintf('Reference %.1f BPM',bpm_ref), 'LineWidth',1.5);
yline(bpm_ref+5, 'k:', '+5 BPM'); yline(bpm_ref-5,'k:','-5 BPM');
xlabel('Window Length (s)'); ylabel('Estimated BPM');
legend('FFT','Welch','MUSIC','ESPRIT K=1','ESPRIT K=2 LS','ESPRIT K=2 TLS','Location','best');
title(sprintf('BPM Accuracy vs Window Length  (M_max=%d)', M_max));
xticks(win_lengths_sec); ylim([bpm_ref-30, bpm_ref+30]); grid on;

% ── Fig 3 — Eigenvalue spectra: complex vs real Hankel ────────────────────
figure('Name','Fig 3 — Eigenvalue Spectra: Complex (K=1) vs Real (K=2)');
n_w = numel(win_lengths_sec);
for k = 1:n_w
    ec = eig_cmplx{k}; ec_n = ec/ec(1);
    er = eig_real{k};  er_n = er/er(1);

    subplot(2, n_w, k);
    bar(1:numel(ec_n), ec_n, 'm');
    title(sprintf('%ds complex  λ1/λ2=%.2f', win_lengths_sec(k), ec(1)/ec(2)));
    xlabel('Index'); ylabel('Norm.'); ylim([0 1.1]); grid on;

    subplot(2, n_w, k+n_w);
    bar(1:numel(er_n), er_n, 'c');
    % For real cosine, λ1≈λ2 always (equal signal eigenvalues).
    % λ2/λ3 is the true SNR indicator: signal subspace edge vs noise floor.
    title(sprintf('%ds real  λ2/λ3=%.2f', win_lengths_sec(k), er(2)/er(3)));
    xlabel('Index'); ylabel('Norm.'); ylim([0 1.1]); grid on;
end
sgtitle('Top row: complex Hankel (K=1 analytic)   Bottom row: real Hankel (K=2 real)');

% ── Fig 4 — Summary: all estimates on 5s and 10s windows side by side ─────
figure('Name','Fig 4 — BPM Error Summary');
methods = {'FFT','Welch','MUSIC','ESP-K1','ESP-K2-LS','ESP-K2-TLS'};
colors_bar = [0 0.4 0.8; 0.8 0.1 0.1; 0.1 0.7 0.1; 0.8 0 0.8; 0 0.8 0.8; 0.9 0.7 0];
for ki = 1:numel(win_lengths_sec)
    vals = [bpm_fft(ki), bpm_welch(ki), bpm_music(ki), ...
            bpm_esp_k1(ki), bpm_esp_k2_ls(ki), bpm_esp_k2_tls(ki)] - bpm_ref;
    subplot(1, numel(win_lengths_sec), ki);
    hb = bar(1:6, vals, 0.6);
    for bi = 1:6, hb.FaceColor = 'flat'; hb.CData(bi,:) = colors_bar(bi,:); end
    yline(0,'k--','LineWidth',1.5);
    yline(5,'k:'); yline(-5,'k:');
    set(gca,'XTick',1:6,'XTickLabel',methods,'XTickLabelRotation',30);
    ylabel('BPM error (est − ref)');
    title(sprintf('Window = %ds', win_lengths_sec(ki)));
    ylim([-20 40]); grid on;
end
sgtitle(sprintf('BPM Error relative to Reference %.1f BPM  (±5 BPM threshold shown)', bpm_ref));

% ── Console summary ───────────────────────────────────────────────────────
fprintf('\n%-8s  %-7s  %-7s  %-7s  %-9s  %-11s  %-11s\n', ...
    'Win(s)','FFT','Welch','MUSIC','ESP-K1','ESP-K2-LS','ESP-K2-TLS');
fprintf('%s\n', repmat('-',1,70));
for k = 1:numel(win_lengths_sec)
    fprintf('%-8d  %-7.1f  %-7.1f  %-7.1f  %-9.1f  %-11.1f  %-11.1f\n', ...
        win_lengths_sec(k), bpm_fft(k), bpm_welch(k), bpm_music(k), ...
        bpm_esp_k1(k), bpm_esp_k2_ls(k), bpm_esp_k2_tls(k));
end
fprintf('\n%-8s  %-7s  %-7s  %-7s  %-9s  %-11s  %-11s  (errors vs %.1f BPM)\n', ...
    'Win(s)','FFT_e','Welch_e','MUSIC_e','K1_e','K2-LS_e','K2-TLS_e', bpm_ref);
fprintf('%s\n', repmat('-',1,70));
for k = 1:numel(win_lengths_sec)
    fprintf('%-8d  %-7.1f  %-7.1f  %-7.1f  %-9.1f  %-11.1f  %-11.1f\n', ...
        win_lengths_sec(k), ...
        bpm_fft(k)-bpm_ref, bpm_welch(k)-bpm_ref, bpm_music(k)-bpm_ref, ...
        bpm_esp_k1(k)-bpm_ref, bpm_esp_k2_ls(k)-bpm_ref, bpm_esp_k2_tls(k)-bpm_ref);
end

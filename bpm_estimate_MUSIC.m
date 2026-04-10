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
    [~, idx] = max(all_boxes(:,3) .* all_boxes(:,4));
    bbox = all_boxes(idx,:);
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

% ── Butterworth bandpass (for FFT + Welch comparison) ─────────────────────
f_low = 0.7; f_high = 3.5;
Wn = [f_low f_high] / (fs/2);
[b, a] = butter(4, Wn, 'bandpass');
S_filt = filtfilt(b, a, S_det);

% =========================================================================
%  MUSIC ALGORITHM — Section 3: Prof. Chen roadmap
%
%  CHANGE: M_max          — max embedding dimension per window.
%                           Actual M is min(M_max, floor(N/4)) per window.
%                           Needs M >= 1 cardiac period in samples (~33 at 30fps).
%                           Try 40, 60, 80. Higher = sharper spectrum but needs more data.
%  CHANGE: K              — number of signal components. K=1 for rPPG.
%  CHANGE: win_lengths_sec — window lengths to compare across all methods.
%
%  MUSIC input: S_det (detrended only — white noise assumption).
%  Why not S_filt (bandpass)? → Creates colored noise → broad hump, wrong BPM.
% =========================================================================
M_max = 60;
K     = 1;
win_lengths_sec = [3, 5, 10, floor(T/fs)];

nfft   = 4096;
f_grid = linspace(f_low, f_high, nfft);   % fine frequency grid for MUSIC

% ── Reference BPM: full signal Welch ──────────────────────────────────────
[P_ref, f_ref] = pwelch(S_filt, hann(T), floor(T/2), nfft, fs);
band_ref = (f_ref >= f_low) & (f_ref <= f_high);
[~, ri] = max(P_ref(band_ref));
fr = f_ref(band_ref);
bpm_ref = fr(ri) * 60;
fprintf('Reference BPM (full Welch): %.1f\n\n', bpm_ref);

% Storage
bpm_fft   = zeros(1, numel(win_lengths_sec));
bpm_welch = zeros(1, numel(win_lengths_sec));
bpm_music = zeros(1, numel(win_lengths_sec));
P_music_all = zeros(numel(win_lengths_sec), nfft);
eig_store   = cell(1, numel(win_lengths_sec));

% ── Fig 1 — MUSIC pseudo-spectrum at each window length ───────────────────
figure('Name','Fig 1 — MUSIC Pseudo-Spectrum vs Window Length');

for k = 1:numel(win_lengths_sec)
    N = min(round(win_lengths_sec(k) * fs), T);
    x = S_filt(1:N);

    % ── FFT ───────────────────────────────────────────────────────────────
    X_fft  = abs(fft(x .* hann(N), nfft));
    X_fft  = X_fft(1:nfft/2+1);
    f_fft  = (0:nfft/2) * fs / nfft;
    band_f = (f_fft >= f_low) & (f_fft <= f_high);
    [~, pi_f] = max(X_fft(band_f));
    fb_f = f_fft(band_f);
    bpm_fft(k) = fb_f(pi_f) * 60;

    % ── Welch ─────────────────────────────────────────────────────────────
    nperseg_w = max(floor(N/2), round(2*fs));   % at least 2 s per segment
    [P_w, f_w] = pwelch(x, hann(nperseg_w), floor(nperseg_w/2), nfft, fs);
    band_w = (f_w >= f_low) & (f_w <= f_high);
    [~, pi_w] = max(P_w(band_w));
    fb_w = f_w(band_w);
    bpm_welch(k) = fb_w(pi_w) * 60;

    % ── MUSIC ─────────────────────────────────────────────────────────────
    % Input: S_det (not S_filt) — white noise assumption holds on unfiltered signal.
    % Analytic signal: K=1 valid only for complex signal (one complex exponential).
    % Adaptive M: must span ≥1 cardiac cycle (~33 samples) but M ≤ N/4 for
    %             sufficient snapshots (L = N-M+1 ≥ 3M).
    x_m   = S_det(1:N);           % MUSIC input: detrended (best SNR for eigenvalue separation)
    x_c   = hilbert(x_m);         % analytic signal → K=1 valid

    M_use = min(M_max, floor(N/4));   % adaptive: never exceed N/4
    M_use = max(M_use, K+2);          % safety: must be > K

    % Step 1: Hankel embedding
    L     = N - M_use + 1;
    idx   = bsxfun(@plus, (1:M_use)', 0:L-1);
    X_mat = x_c(idx);                          % M_use x L  complex

    % Step 2: Forward-backward covariance (doubles effective snapshots,
    %         improves Hermitian-Toeplitz structure for better eigenvalue separation)
    R_raw = (X_mat * X_mat') / L;
    J     = fliplr(eye(M_use));
    R     = 0.5 * (R_raw + J * conj(R_raw) * J);   % M_use x M_use  Hermitian

    % Step 3: Eigendecompose, sort descending
    [V, D]  = eig(R);
    [~, si] = sort(real(diag(D)), 'descend');
    V = V(:, si);

    % Eigenvalue diagnostic: λ1/λ2 is the key SNR indicator for MUSIC.
    % Need λ1/λ2 >> 1 (ideally >10) for a well-defined noise subspace.
    eig_sorted = sort(real(diag(D)), 'descend');
    fprintf('  window=%ds  M=%d  λ1/λ2=%.2f  (need >>1 for sharp MUSIC peak)\n', ...
        win_lengths_sec(k), M_use, eig_sorted(1)/eig_sorted(2));

    % Step 4: Noise subspace
    Qn = V(:, K+1:end);   % M_use x (M_use-K)

    % Step 5: Steering matrix for this M_use (recomputed since M_use changes per window)
    omega_v = 2*pi * f_grid / fs;
    A_use   = exp(1j * (0:M_use-1)' * omega_v);   % M_use x nfft

    % Step 6: MUSIC pseudo-spectrum  P(f) = 1 / ||Qn^H * a(f)||^2
    denom   = sum(abs(Qn' * A_use).^2, 1);         % 1 x nfft
    P_music = 1 ./ denom;

    P_music_db = 10*log10(P_music / max(P_music));
    P_music_all(k, :) = P_music_db;
    [~, pi_m] = max(P_music);
    bpm_music(k) = f_grid(pi_m) * 60;

    % Store eigenvalues for Fig 4
    eig_store{k} = eig_sorted(1:min(M_use, 20));   %#ok<AGROW>  top 20 eigenvalues

    % Plot pseudo-spectrum
    subplot(numel(win_lengths_sec), 1, k);
    plot(f_grid, P_music_db, 'g', 'LineWidth', 1.5); hold on;
    xline(f_grid(pi_m), 'r--', ...
        sprintf('MUSIC %.1f BPM  |  FFT %.1f  |  Welch %.1f', ...
        bpm_music(k), bpm_fft(k), bpm_welch(k)), 'LabelVerticalAlignment','bottom');
    xlim([0.5 3.5]); ylim([-40 2]);
    ylabel('dB (norm)');
    title(sprintf('MUSIC  window=%ds  M=%d  K=%d  λ1/λ2=%.2f', ...
        win_lengths_sec(k), M_use, K, eig_sorted(1)/eig_sorted(2)));
    grid on;
end
xlabel('Frequency (Hz)');

% ── Fig 2 — BPM accuracy: FFT vs Welch vs MUSIC ───────────────────────────
figure('Name','Fig 2 — BPM Accuracy: FFT vs Welch vs MUSIC');
plot(win_lengths_sec, bpm_fft,   'b-o', 'LineWidth', 1.5, 'MarkerSize', 9); hold on;
plot(win_lengths_sec, bpm_welch, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 9);
plot(win_lengths_sec, bpm_music, 'g-^', 'LineWidth', 2.0, 'MarkerSize', 9);
yline(bpm_ref,   'w--', sprintf('Reference %.1f BPM', bpm_ref), 'LineWidth', 1.5);
yline(bpm_ref+5, 'w:',  '+5 BPM threshold');
yline(bpm_ref-5, 'w:',  '-5 BPM threshold');
xlabel('Window Length (s)'); ylabel('Estimated BPM');
legend('FFT', 'Welch', 'MUSIC');
title(sprintf('BPM Accuracy vs Window Length  (M_max=%d, K=%d, input=S_det)', M_max, K));
xticks(win_lengths_sec);
ylim([bpm_ref-25, bpm_ref+25]);
grid on;

% ── Fig 4 — Eigenvalue spectrum: shows whether signal subspace is isolated ──
% Each bar is one eigenvalue. MUSIC works when bar #1 is far taller than the rest.
% All bars nearly equal → λ1/λ2 ≈ 1 → SNR too low → flat pseudo-spectrum.
figure('Name','Fig 4 — Eigenvalue Spectrum');
colors = {'b','r','g','m'};
for k = 1:numel(win_lengths_sec)
    ev = eig_store{k};
    ev_norm = ev / ev(1);   % normalise to largest = 1
    subplot(1, numel(win_lengths_sec), k);
    bar(1:numel(ev), ev_norm, colors{k});
    xlabel('Index'); ylabel('Norm. magnitude');
    title(sprintf('%ds  λ1/λ2=%.2f', win_lengths_sec(k), ev(1)/ev(2)));
    grid on; ylim([0 1.1]);
end
sgtitle('Eigenvalue spectrum — bar 1 = signal, rest = noise (need bar 1 >> bar 2)');

% ── Fig 3 — Direct overlay: all three spectra on the 5s window ─────────────
% This shows MUSIC's sharper pseudo-spectrum vs the broad FFT and Welch curves
win_compare = 5;
N5 = min(round(win_compare * fs), T);
x5 = S_filt(1:N5);

% FFT on 5s
X5f = abs(fft(x5 .* hann(N5), nfft));
X5f = X5f(1:nfft/2+1);
f5f = (0:nfft/2) * fs / nfft;
band5f = (f5f >= f_low) & (f5f <= f_high);
X5f_db = 20*log10(X5f(band5f) / max(X5f(band5f)));

% Welch on 5s
np5 = max(floor(N5/2), round(2*fs));
[P5w, f5w] = pwelch(x5, hann(np5), floor(np5/2), nfft, fs);
band5w = (f5w >= f_low) & (f5w <= f_high);
P5w_db = 10*log10(P5w(band5w) / max(P5w(band5w)));

% MUSIC on 5s
x5_m  = S_det(1:N5);
x5_c  = hilbert(x5_m);
M5    = min(M_max, floor(N5/4));
M5    = max(M5, K+2);
L5    = N5 - M5 + 1;
idx5  = bsxfun(@plus, (1:M5)', 0:L5-1);
X5mat = x5_c(idx5);
R5raw = (X5mat * X5mat') / L5;
J5    = fliplr(eye(M5));
R5    = 0.5 * (R5raw + J5 * conj(R5raw) * J5);
[V5, D5] = eig(R5);
[~, si5] = sort(real(diag(D5)), 'descend');
Qn5    = V5(:, si5(K+1:end));
A5     = exp(1j * (0:M5-1)' * (2*pi*f_grid/fs));
denom5 = sum(abs(Qn5' * A5).^2, 1);
P5m    = 1 ./ denom5;
P5m_db = 10*log10(P5m / max(P5m));
[~, pi5m] = max(P5m);
bpm5m  = f_grid(pi5m) * 60;

figure('Name',sprintf('Fig 3 — Direct Comparison on %ds window', win_compare));
plot(f5f(band5f), X5f_db, 'b',  'LineWidth', 1.2); hold on;
plot(f5w(band5w), P5w_db, 'r',  'LineWidth', 1.2);
plot(f_grid,      P5m_db, 'g',  'LineWidth', 2.0);
xline(bpm_ref/60, 'w--', sprintf('True ref: %.1f BPM', bpm_ref), 'LineWidth', 1.5);
xlabel('Frequency (Hz)'); ylabel('Normalized dB');
legend('FFT', 'Welch', 'MUSIC');
title(sprintf('All three methods — %ds window  |  MUSIC peak: %.1f BPM', win_compare, bpm5m));
xlim([0.5 3.5]); ylim([-40 2]);
grid on;

% ── Console summary ───────────────────────────────────────────────────────
fprintf('%-12s  %-8s  %-8s  %-8s  %-10s  %-10s  %-10s\n', ...
    'Window(s)', 'FFT', 'Welch', 'MUSIC', 'FFT err', 'Welch err', 'MUSIC err');
fprintf('%s\n', repmat('-', 1, 72));
for k = 1:numel(win_lengths_sec)
    fprintf('%-12d  %-8.1f  %-8.1f  %-8.1f  %-10.1f  %-10.1f  %-10.1f\n', ...
        win_lengths_sec(k), bpm_fft(k), bpm_welch(k), bpm_music(k), ...
        bpm_fft(k)-bpm_ref, bpm_welch(k)-bpm_ref, bpm_music(k)-bpm_ref);
end
fprintf('%-12s  %-8s  %-8s  %-8s  %-10s  %-10s  %-10s  (reference)\n', ...
    'Full', '--', num2str(bpm_ref,'%.1f'), '--', '0.0', '0.0', '--');

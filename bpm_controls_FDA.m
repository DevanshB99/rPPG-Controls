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

% ── Butterworth bandpass (order=4, chosen from filter design study) ───────
f_low = 0.7; f_high = 3.5;
Wn = [f_low f_high] / (fs/2);
[b, a] = butter(4, Wn, 'bandpass');
S_filt = filtfilt(b, a, S_det);

% =========================================================================
%  FREQUENCY DOMAIN ANALYSIS
%  CHANGE: win_lengths_sec — window lengths (s) to compare across methods
%  CHANGE: stft_win_sec    — STFT sliding window size (affects time resolution)
%  CHANGE: stft_ovlp_frac  — STFT overlap fraction (0.5–0.95; higher = smoother)
% =========================================================================
win_lengths_sec = [5, 10, floor(T/fs)];   % last entry = full signal
stft_win_sec    = 8;
stft_ovlp_frac  = 0.90;

nfft = 4096;   % zero-pad for smooth display (does not change true resolution)

% ── Fig 1 — FFT at each window length ─────────────────────────────────────
% True frequency resolution = fs/N Hz per bin (shown in title).
% Zero-padding to nfft only interpolates — it does NOT improve resolution.
figure('Name','Fig 1 — FFT');
bpm_fft = zeros(1, numel(win_lengths_sec));

for k = 1:numel(win_lengths_sec)
    N = min(round(win_lengths_sec(k) * fs), T);
    x = S_filt(1:N) .* hann(N);
    X = abs(fft(x, nfft));
    X = X(1:nfft/2+1);
    f = (0:nfft/2) * fs / nfft;

    band = (f >= f_low) & (f <= f_high);
    [~, pidx] = max(X(band));
    fb = f(band); fp = fb(pidx);
    bpm_fft(k) = fp * 60;

    X_db = 20*log10(X / max(X));

    subplot(numel(win_lengths_sec), 1, k);
    plot(f, X_db, 'b', 'LineWidth', 1.2); hold on;
    xline(f_low,  'k--', 'LineWidth', 1);
    xline(f_high, 'k--', 'LineWidth', 1);
    xline(fp, 'r--', sprintf('peak %.1f BPM', fp*60), 'LabelVerticalAlignment','bottom');
    xlim([0.5 3.5]); ylim([-40 2]);
    ylabel('dB (norm)');
    title(sprintf('FFT  window=%ds  |  bin resolution = %.3f Hz = %.1f BPM/bin', ...
        win_lengths_sec(k), fs/N, fs/N*60));
    grid on;
end
xlabel('Frequency (Hz)');

% ── Fig 2 — Welch at each window length ───────────────────────────────────
figure('Name','Fig 2 — Welch');
bpm_welch = zeros(1, numel(win_lengths_sec));

for k = 1:numel(win_lengths_sec)
    nperseg  = min(round(win_lengths_sec(k) * fs), T);
    noverlap = floor(nperseg / 2);
    [P, f]   = pwelch(S_filt, hann(nperseg), noverlap, nfft, fs);

    band = (f >= f_low) & (f <= f_high);
    [~, pidx] = max(P(band));
    fb = f(band); fp = fb(pidx);
    bpm_welch(k) = fp * 60;

    subplot(numel(win_lengths_sec), 1, k);
    plot(f(band), 10*log10(P(band)), 'r', 'LineWidth', 1.2); hold on;
    xline(fp, 'b--', sprintf('peak %.1f BPM', fp*60), 'LabelVerticalAlignment','bottom');
    ylabel('PSD (dB/Hz)');
    title(sprintf('Welch  nperseg=%ds  |  bin resolution = %.3f Hz = %.1f BPM/bin', ...
        win_lengths_sec(k), fs/nperseg, fs/nperseg*60));
    grid on;
end
xlabel('Frequency (Hz)');

% ── Fig 3 — STFT time-frequency heatmap ───────────────────────────────────
stft_N  = round(stft_win_sec * fs);
stft_ov = floor(stft_N * stft_ovlp_frac);

[~, f_st, t_st, P_st] = spectrogram(S_filt, hann(stft_N), stft_ov, nfft, fs);
band_st = (f_st >= f_low) & (f_st <= f_high);
P_db    = 10*log10(P_st(band_st, :));
f_cb    = f_st(band_st);

figure('Name','Fig 3 — STFT');
imagesc(t_st, f_cb, P_db);
axis xy; colormap(jet); colorbar;
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title(sprintf('STFT  window=%ds  overlap=%.0f%%  |  time-res=%.1fs  freq-res=%.3fHz', ...
    stft_win_sec, stft_ovlp_frac*100, (stft_N-stft_ov)/fs, fs/stft_N));
p_range = prctile(P_db(:), [5 99]);
caxis(p_range);

% Dominant frequency track overlaid as white dashed line
[~, peak_row] = max(P_st(band_st, :), [], 1);
hold on;
plot(t_st, f_cb(peak_row), 'w--', 'LineWidth', 2.0);

% ── Fig 4 — BPM estimate vs window length (motivation for MUSIC) ──────────
% Reference: full-signal Welch (most accurate offline estimate)
[P_ref, f_ref] = pwelch(S_filt, hann(T), floor(T/2), nfft, fs);
band_ref = (f_ref >= f_low) & (f_ref <= f_high);
[~, pidx_ref] = max(P_ref(band_ref));
fb_ref = f_ref(band_ref);
bpm_ref = fb_ref(pidx_ref) * 60;

figure('Name','Fig 4 — BPM vs Window Length');
plot(win_lengths_sec, bpm_fft,   'b-o', 'LineWidth', 1.5, 'MarkerSize', 8); hold on;
plot(win_lengths_sec, bpm_welch, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8);
yline(bpm_ref,   'k--', sprintf('Reference %.1f BPM (full signal Welch)', bpm_ref), 'LineWidth', 1.5);
yline(bpm_ref+5, 'k:',  '+5 BPM error threshold');
yline(bpm_ref-5, 'k:',  '-5 BPM error threshold');
xlabel('Window Length (s)'); ylabel('Estimated BPM');
legend('FFT', 'Welch', 'Location', 'best');
title('BPM Accuracy vs Window Length  —  shorter windows need MUSIC');
xticks(win_lengths_sec);
ylim([bpm_ref-20, bpm_ref+20]);
grid on;

% ── Console summary ───────────────────────────────────────────────────────
fprintf('\n%-12s  %-10s  %-10s  %-12s  %-12s\n', ...
    'Window (s)', 'FFT BPM', 'Welch BPM', 'FFT err', 'Welch err');
fprintf('%s\n', repmat('-', 1, 60));
for k = 1:numel(win_lengths_sec)
    fprintf('%-12d  %-10.1f  %-10.1f  %-12.1f  %-12.1f\n', ...
        win_lengths_sec(k), bpm_fft(k), bpm_welch(k), ...
        bpm_fft(k)-bpm_ref, bpm_welch(k)-bpm_ref);
end
fprintf('%-12s  %-10.1f  %-10.1f  %-12s  %-12s  (reference)\n', ...
    'Full', bpm_ref, bpm_ref, '0.0', '0.0');

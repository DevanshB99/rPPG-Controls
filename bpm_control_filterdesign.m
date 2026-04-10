clear; close all; clc;

VID_PATH = '/Users/devanshbajwala/Documents/VS Code WS/PyQt/DocBOT/rPPG Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/recording_2026-04-07T22-04-35Z.mov';
CSV_PATH = '/Users/devanshbajwala/Documents/VS Code WS/PyQt/DocBOT/rPPG Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/vitals.csv';

% ── Ground truth from CSV ─────────────────────────────────────────────────
csv_data = readtable(CSV_PATH);
gt_time  = csv_data.offset_seconds;
gt_hr    = csv_data.heart_rate;
valid    = ~isnan(gt_hr);
gt_time  = gt_time(valid);
gt_hr    = gt_hr(valid);
bpm_gt_mean   = mean(gt_hr);
bpm_gt_median = median(gt_hr);
fprintf('Ground truth: mean=%.1f BPM  median=%.1f BPM  range=[%d %d] BPM  n=%d samples\n', ...
    bpm_gt_mean, bpm_gt_median, min(gt_hr), max(gt_hr), numel(gt_hr));

% ── Face detection on first video frame ──────────────────────────────────
vid_tmp     = VideoReader(VID_PATH);
first_frame = readFrame(vid_tmp);
[H_vid, W_vid, ~] = size(first_frame);
clear vid_tmp;

try
    detector  = vision.CascadeObjectDetector();
    all_boxes = step(detector, first_frame);
catch
    all_boxes = [];
end

if isempty(all_boxes)
    x1 = floor(W_vid*0.25); y1 = floor(H_vid*0.05);
    x2 = floor(W_vid*0.75); y2 = floor(H_vid*0.75);
    fprintf('Face detector: fallback ROI used\n');
else
    [~, idx] = max(all_boxes(:,3) .* all_boxes(:,4));
    bbox = all_boxes(idx,:);
    x1 = max(bbox(1), 1);
    y1 = max(bbox(2), 1);
    x2 = min(bbox(1)+bbox(3)-1, W_vid);
    y2 = min(bbox(2)+bbox(4)-1, H_vid);
    fprintf('Face detector: bbox [%d %d %d %d]\n', x1, y1, x2, y2);
end

% ── Video: frame-by-frame RGB extraction ─────────────────────────────────
vid = VideoReader(VID_PATH);
fs  = vid.FrameRate;
R_t = []; G_t = []; B_t = [];

while hasFrame(vid)
    frame = readFrame(vid);
    fc    = frame(y1:y2, x1:x2, :);
    fcd   = double(fc);

    % ── Luminance normalisation — removes camera AGC / auto-exposure drift ──
    % Each frame is divided by its spatial mean before computing RGB means.
    % Without this, frame-to-frame gain changes (visible as changing ISO/EV
    % in the CSV) create a slow artifact inside the cardiac band that
    % dominates Welch and makes all filters except Chebyshev II fail.
    frame_lum = mean(fcd(:));
    if frame_lum > 0; fcd = fcd / frame_lum * 128; end  % normalise to 128 DN

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

% ── CHROM ────────────────────────────────────────────────────────────────
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

% =========================================================================
%  IIR FILTER DESIGN COMPARISON
%  CHANGE: f_low, f_high, order, Rp, Rs to explore different designs
% =========================================================================
f_low  = 0.7;   % Hz — lower cutoff
f_high = 3.5;    % Hz — upper cutoff
order  = 2;      % filter order per section (bandpass doubles this)
Rp     = 0.5;    % dB — max passband ripple     (Chebyshev I, Elliptic)
Rs     = 40;     % dB — min stopband attenuation (Chebyshev II, Elliptic)

Wn = [f_low f_high] / (fs/2);

[b_bw, a_bw] = butter(order,       Wn,     'bandpass');
[b_c1, a_c1] = cheby1(order, Rp,   Wn,     'bandpass');
[b_c2, a_c2] = cheby2(order, Rs,   Wn,     'bandpass');
[b_el, a_el] = ellip( order, Rp, Rs, Wn,   'bandpass');

% ── Frequency vectors ─────────────────────────────────────────────────────
f_full = linspace(0, fs/2, 4096);
f_zoom = linspace(0, 6,    8192);

H_bw_full = freqz(b_bw, a_bw, f_full, fs);
H_c1_full = freqz(b_c1, a_c1, f_full, fs);
H_c2_full = freqz(b_c2, a_c2, f_full, fs);
H_el_full = freqz(b_el, a_el, f_full, fs);

H_bw_zoom = freqz(b_bw, a_bw, f_zoom, fs);
H_c1_zoom = freqz(b_c1, a_c1, f_zoom, fs);
H_c2_zoom = freqz(b_c2, a_c2, f_zoom, fs);
H_el_zoom = freqz(b_el, a_el, f_zoom, fs);

% ── Figure 1a: Magnitude BEFORE fix (shows the broken scale) ─────────────
figure('Name','Fig 1a — Magnitude BEFORE fix');
plot(f_full, 20*log10(abs(H_bw_full)), 'b', 'LineWidth',1.5); hold on;
plot(f_full, 20*log10(abs(H_c1_full)), 'r', 'LineWidth',1.5);
plot(f_full, 20*log10(abs(H_c2_full)), 'g', 'LineWidth',1.5);
plot(f_full, 20*log10(abs(H_el_full)), 'm', 'LineWidth',1.5);
xline(f_low,  'k--', 'LineWidth',1);
xline(f_high, 'k--', 'LineWidth',1);
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend('Butterworth','Chebyshev I','Chebyshev II','Elliptic');
title('BEFORE fix — y-axis distorted by -Inf (filter zeros hit log10(0))');
grid on;

% ── Figure 1b: Magnitude AFTER fix (clipped dB, zoomed x-axis) ───────────
dB_floor = -80;
bw_dB = max(20*log10(abs(H_bw_zoom)), dB_floor);
c1_dB = max(20*log10(abs(H_c1_zoom)), dB_floor);
c2_dB = max(20*log10(abs(H_c2_zoom)), dB_floor);
el_dB = max(20*log10(abs(H_el_zoom)), dB_floor);

figure('Name','Fig 1b — Magnitude AFTER fix');
plot(f_zoom, bw_dB, 'b', 'LineWidth',1.5); hold on;
plot(f_zoom, c1_dB, 'r', 'LineWidth',1.5);
plot(f_zoom, c2_dB, 'g', 'LineWidth',1.5);
plot(f_zoom, el_dB, 'm', 'LineWidth',1.5);
xline(f_low,  'k--', 'LineWidth',1);
xline(f_high, 'k--', 'LineWidth',1);
xline(bpm_gt_mean/60, 'k-', sprintf('GT %.1f BPM', bpm_gt_mean), 'LineWidth', 2, 'LabelVerticalAlignment','bottom');
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend('Butterworth','Chebyshev I','Chebyshev II','Elliptic');
title(sprintf('AFTER fix — order=%d  Rp=%.1fdB  Rs=%.1fdB  |  GT=%.1f BPM', order, Rp, Rs, bpm_gt_mean));
ylim([dB_floor 5]); xlim([0 6]);
grid on;

% ── Figure 2: Phase responses ─────────────────────────────────────────────
figure('Name','Fig 2 — Phase Response');
plot(f_zoom, unwrap(angle(H_bw_zoom))*180/pi, 'b', 'LineWidth',1.5); hold on;
plot(f_zoom, unwrap(angle(H_c1_zoom))*180/pi, 'r', 'LineWidth',1.5);
plot(f_zoom, unwrap(angle(H_c2_zoom))*180/pi, 'g', 'LineWidth',1.5);
plot(f_zoom, unwrap(angle(H_el_zoom))*180/pi, 'm', 'LineWidth',1.5);
xline(f_low,  'k--', 'LineWidth',1);
xline(f_high, 'k--', 'LineWidth',1);
xlabel('Frequency (Hz)'); ylabel('Phase (degrees)');
legend('Butterworth','Chebyshev I','Chebyshev II','Elliptic');
title('Phase Response  (jumps in Cheby II / Elliptic are unwrap artifacts at filter zeros)');
grid on; xlim([0 6]);

% ── Figure 3: Group delay ─────────────────────────────────────────────────
gd_bw = grpdelay(b_bw, a_bw, f_zoom, fs);
gd_c1 = grpdelay(b_c1, a_c1, f_zoom, fs);
gd_c2 = grpdelay(b_c2, a_c2, f_zoom, fs);
gd_el = grpdelay(b_el, a_el, f_zoom, fs);

figure('Name','Fig 3 — Group Delay');
plot(f_zoom, gd_bw/fs*1000, 'b', 'LineWidth',1.5); hold on;
plot(f_zoom, gd_c1/fs*1000, 'r', 'LineWidth',1.5);
plot(f_zoom, gd_c2/fs*1000, 'g', 'LineWidth',1.5);
plot(f_zoom, gd_el/fs*1000, 'm', 'LineWidth',1.5);
xline(f_low,  'k--', 'LineWidth',1);
xline(f_high, 'k--', 'LineWidth',1);
xlabel('Frequency (Hz)'); ylabel('Group Delay (ms)');
legend('Butterworth','Chebyshev I','Chebyshev II','Elliptic');
title('Group Delay  (filtfilt cancels this for offline use — matters for real-time)');
grid on; xlim([0 6]); ylim([-200 2000]);

% ── Apply filtfilt ────────────────────────────────────────────────────────
S_bw = filtfilt(b_bw, a_bw, S_det);
S_c1 = filtfilt(b_c1, a_c1, S_det);
S_c2 = filtfilt(b_c2, a_c2, S_det);
S_el = filtfilt(b_el, a_el, S_det);

fprintf('Filter outputs: std bw=%.6f  c1=%.6f  c2=%.6f  el=%.6f\n', ...
    std(S_bw), std(S_c1), std(S_c2), std(S_el));

% ── Figure 4: Filtered signals overlaid ──────────────────────────────────
figure('Name','IIR Filter — Filtered Signals');
plot(t_axis, S_det, 'Color',[0.75 0.75 0.75], 'LineWidth',1.0); hold on;
plot(t_axis, S_bw, 'b', 'LineWidth',1.2);
plot(t_axis, S_c1, 'r', 'LineWidth',1.2);
plot(t_axis, S_c2, 'g', 'LineWidth',1.2);
plot(t_axis, S_el, 'm', 'LineWidth',1.2);
xlabel('Time (s)'); ylabel('Amplitude');
legend('S_{det} (input)','Butterworth','Chebyshev I','Chebyshev II','Elliptic');
title('Filtered BVP Signals — All Four IIR Filters');
grid on;

% ── PSD + peak BPM per filter (full signal Welch) ─────────────────────────
nperseg  = min(round(fs*10), T);
noverlap = floor(nperseg/2);

[p_bw, f_p] = pwelch(S_bw, hann(nperseg), noverlap, [], fs);
[p_c1, ~  ] = pwelch(S_c1, hann(nperseg), noverlap, [], fs);
[p_c2, ~  ] = pwelch(S_c2, hann(nperseg), noverlap, [], fs);
[p_el, ~  ] = pwelch(S_el, hann(nperseg), noverlap, [], fs);

band = (f_p >= f_low) & (f_p <= f_high);

% ── Figure 5: PSD with GT reference line ─────────────────────────────────
figure('Name','IIR Filter — PSD vs Ground Truth');
plot(f_p(band), 10*log10(p_bw(band)), 'b', 'LineWidth',1.5); hold on;
plot(f_p(band), 10*log10(p_c1(band)), 'r', 'LineWidth',1.5);
plot(f_p(band), 10*log10(p_c2(band)), 'g', 'LineWidth',1.5);
plot(f_p(band), 10*log10(p_el(band)), 'm', 'LineWidth',1.5);
xline(bpm_gt_mean/60,   'k-',  sprintf('GT mean %.1f BPM',   bpm_gt_mean),   'LineWidth', 2.5, 'LabelVerticalAlignment','bottom');
xline(bpm_gt_median/60, 'k--', sprintf('GT median %.1f BPM', bpm_gt_median), 'LineWidth', 1.5, 'LabelVerticalAlignment','top');
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
legend('Butterworth','Chebyshev I','Chebyshev II','Elliptic','GT mean','GT median','Location','best');
title(sprintf('PSD — Cardiac Band  |  GT mean=%.1f BPM', bpm_gt_mean));
grid on;

% ── Figure 6: GT HR vs time + sliding-window BPM estimates ───────────────
% Sliding window: step every 1 s, window = 10 s (matching GT 1 Hz cadence)
sw_len    = round(10 * fs);   % 10-second window in frames
sw_step   = round(1  * fs);   % 1-second step
sw_starts = 1 : sw_step : T - sw_len + 1;
n_sw      = numel(sw_starts);

sw_time = zeros(1, n_sw);
sw_bpm  = struct('bw', zeros(1,n_sw), 'c1', zeros(1,n_sw), ...
                 'c2', zeros(1,n_sw), 'el', zeros(1,n_sw));

nfft_sw = 4096;
for k = 1:n_sw
    idx_s = sw_starts(k);
    idx_e = idx_s + sw_len - 1;
    sw_time(k) = (idx_s - 1) / fs + sw_len/(2*fs);  % window centre time

    segs    = {S_bw(idx_s:idx_e), S_c1(idx_s:idx_e), ...
               S_c2(idx_s:idx_e), S_el(idx_s:idx_e)};
    fn      = {'bw','c1','c2','el'};
    for fi = 1:4
        seg = segs{fi};
        np  = min(sw_len, length(seg));
        [pw_k, fw_k] = pwelch(seg, hann(np), floor(np/2), nfft_sw, fs);
        bk  = (fw_k >= f_low) & (fw_k <= f_high);
        [~, pi_] = max(pw_k(bk));
        fb_ = fw_k(bk);
        sw_bpm.(fn{fi})(k) = fb_(pi_) * 60;
    end
end

% ── Interpolate GT onto the sliding-window time grid ─────────────────────
% Each sw_time(k) is the centre of a 10s window.  The GT is at ~1 Hz.
% Linear interpolation gives the expected HR at each window centre.
gt_interp = interp1(gt_time, double(gt_hr), sw_time, 'linear', 'extrap');

sw_arrays    = {sw_bpm.bw, sw_bpm.c1, sw_bpm.c2, sw_bpm.el};
colors       = {'b','r','g','m'};
filter_labels = {'Butterworth','Chebyshev I','Chebyshev II','Elliptic'};
sw_err = zeros(4, n_sw);
for fi = 1:4
    sw_err(fi,:) = sw_arrays{fi} - gt_interp;
end

% ── Figure 6: GT + sliding-window BPM with ±1σ rolling bands ─────────────
figure('Name','Fig 6 — Sliding Window BPM vs Ground Truth');
stairs(gt_time, gt_hr, 'k', 'LineWidth', 2.5); hold on;
for fi = 1:4
    est   = sw_arrays{fi};
    sigma = movstd(est, 11);   % rolling std over ±5 windows = ±5 s
    fill([sw_time, fliplr(sw_time)], ...
         [est+sigma, fliplr(est-sigma)], colors{fi}, ...
         'FaceAlpha', 0.12, 'EdgeColor', 'none');
    plot(sw_time, est, colors{fi}, 'LineWidth', 1.5);
end
yline(bpm_gt_mean, 'k--', sprintf('GT mean %.1f BPM', bpm_gt_mean), 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('BPM');
legend('Ground Truth','','Butterworth','','Chebyshev I','','Chebyshev II','','Elliptic','Location','best');
title(sprintf('Sliding Window (10s) BPM ± 1\\sigma vs GT  |  order=%d  f=[%.1f, %.1f] Hz', ...
    order, f_low, f_high));
ylim([30 130]); xlim([0 max(gt_time)+2]); grid on;

% ── Figure 7: Per-window error with ±5 BPM threshold ─────────────────────
figure('Name','Fig 7 — Per-window BPM Error vs GT');
for fi = 1:4
    err_i  = sw_err(fi,:);
    mae_i  = mean(abs(err_i));
    rmse_i = sqrt(mean(err_i.^2));
    rho_i  = corr(sw_arrays{fi}', gt_interp');

    subplot(4,1,fi);
    fill([sw_time, fliplr(sw_time)], ...
         [err_i+movstd(err_i,11), fliplr(err_i-movstd(err_i,11))], ...
         colors{fi}, 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
    plot(sw_time, err_i, colors{fi}, 'LineWidth', 1.5);
    yline(0, 'k--', 'LineWidth', 1.2);
    yline( 5, 'k:', 'LineWidth', 1);
    yline(-5, 'k:', 'LineWidth', 1);
    ylabel('Error (BPM)');
    title(sprintf('%s  |  MAE=%.1f BPM   RMSE=%.1f BPM   r=%.3f', ...
        filter_labels{fi}, mae_i, rmse_i, rho_i));
    ylim([-65 65]); xlim([0 max(gt_time)+2]); grid on;
end
xlabel('Time (s)');
sgtitle('Per-window Error (rPPG − GT)  |  dotted = ±5 BPM threshold  |  shaded = ±1σ');

% ── Console summary ───────────────────────────────────────────────────────
filters = {'Butterworth','Chebyshev I','Chebyshev II','Elliptic'};
psds    = {p_bw, p_c1, p_c2, p_el};

fprintf('\n%-14s  %7s  %7s  %7s  %7s  %7s  %7s\n', ...
    'Filter','BPM','GT','Err(full)','MAE(sw)','RMSE(sw)','r(sw)');
fprintf('%s\n', repmat('-',1,60));
for k = 1:4
    pb = psds{k}(band);
    [~, pi_] = max(pb); fp = f_p(band); fp = fp(pi_);
    mae_k  = mean(abs(sw_err(k,:)));
    rmse_k = sqrt(mean(sw_err(k,:).^2));
    rho_k  = corr(sw_arrays{k}', gt_interp');
    fprintf('%-14s  %7.1f  %7.1f  %7.1f  %7.1f  %7.1f  %7.3f\n', ...
        filters{k}, fp*60, bpm_gt_mean, fp*60-bpm_gt_mean, mae_k, rmse_k, rho_k);
end
fprintf('\nGT: mean=%.1f  median=%.1f  std=%.1f  range=[%d %d]  n=%d\n', ...
    bpm_gt_mean, bpm_gt_median, std(double(gt_hr)), min(gt_hr), max(gt_hr), numel(gt_hr));

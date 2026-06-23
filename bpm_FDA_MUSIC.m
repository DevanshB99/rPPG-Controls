clear; close all; clc;

addpath('/home/macs/Downloads/macs-matlab-toolbox-master/macs-matlab-toolbox-master');
addpath('/home/macs/Downloads/MUSIC-ESPRIT-Frequency-ID-main/MUSIC-ESPRIT-Frequency-ID-main');

CSV_PATH = '/home/macs/Documents/rPPG-Controls/bpm_sample_output_35.csv';

data           = readtable(CSV_PATH);
t_axis         = data.time_s;
fs             = 1 / (t_axis(2) - t_axis(1));
T              = height(data);
S_bw           = data.BVP_butterworth;
S_c1           = data.BVP_cheby1;
S_c2           = data.BVP_cheby2;
S_el           = data.BVP_elliptic;
lum_t          = data.frame_luminance;
detected_t     = data.face_detected;
skin_count_t   = data.skin_pixel_count;

gt_bpm  = data.gt_bpm;
vld_gt  = ~isnan(gt_bpm);   % NaN = frames outside vitals monitor window; BVP is still valid there
gt_mean = mean(gt_bpm(vld_gt));

fprintf('Loaded %d frames  fs=%.2f Hz  GT coverage=%.0f%%  GT mean=%.1f BPM\n', ...
    T, fs, 100*mean(vld_gt), gt_mean);

f_low  = 0.7;
f_high = 3.5;

% ── Section A: Full-signal spectra ──────────────────────────────────────────
specCal(S_bw, fs);

sce = specCale([S_bw, S_c1, S_c2, S_el], fs);
figure('Name', 'specCale — All Filters');
plot(sce.f(:,1), sce.amp, 'LineWidth', 1.2);
xline(f_low, 'k--', '0.7 Hz');  xline(f_high, 'k--', '3.5 Hz');
xlim([0, f_high + 1]);
legend('Butterworth','Cheby I','Cheby II','Elliptic','Location','best');
xlabel('Frequency (Hz)');  ylabel('Amplitude');
title('Amplitude Spectra — All Filtered BVP Signals');  grid on;

% ── Section B: Sliding-window FFT / Welch / MUSIC / ESPRIT ──────────────────
win_secs      = [2, 3, 5, 10, 20];
nfft          = 4096;
ACC_THR       = 10;     % ±BPM tolerance counted as "accurate"
SNR_THR       = 6;      % dB: cardiac peak vs in-band noise floor
DET_THR       = 0.7;    % fraction of frames with fresh MTCNN detection
SKIN_THR      = 300;    % minimum mean skin pixels per window
LUM_CV_THR    = 0.12;   % luminance coefficient-of-variation (AGC drift proxy)
CONF_THR      = 0.80;   % composite confidence score threshold for HQ classification
% With W_CONF=[0.40,0.30,0.15,0.15], CONF_THR=0.80 requires snr_score>=0.5 (≥6dB) when
% det/skin/lum are perfect (contribution floor = 0.60). This correctly gates on SNR.
W_CONF        = [0.40, 0.30, 0.15, 0.15];  % weights: [SNR, det_rate, skin_mean, lum_cv]
omega_cb      = linspace(2*pi*f_low/fs, 2*pi*f_high/fs, 2048);

fprintf('\n%s\n', repmat('=', 1, 100));
fprintf('  %-6s  %-9s  %-9s  %-9s  %-9s  %-9s  %-9s  %-9s  %-9s\n', ...
    'Win(s)','FFT_MAE','Wch_MAE','MUS_MAE','ESP_MAE','FFT_Acc','Wch_Acc','MUS_Acc','ESP_Acc');
fprintf('%s\n', repmat('=', 1, 100));

for wi = 1:numel(win_secs)
    win_N = round(win_secs(wi) * fs);
    if win_N > T;  continue;  end

    M_sub  = max(round(win_N / 4), 6);  % need M > p+1=4 for p=4 (2 real sinusoids)
    starts = 1 : round(fs) : T - win_N + 1;
    n_sw   = numel(starts);

    sw_time     = zeros(1, n_sw);
    bpm_fft     = zeros(1, n_sw);
    bpm_welch   = zeros(1, n_sw);
    bpm_music   = nan(1, n_sw);
    bpm_esprit  = nan(1, n_sw);
    snr_db      = zeros(1, n_sw);
    det_rate    = zeros(1, n_sw);
    skin_mean   = zeros(1, n_sw);
    lum_cv_win  = zeros(1, n_sw);

    for k = 1:n_sw
        idx           = starts(k) : starts(k) + win_N - 1;
        seg           = S_bw(idx);
        sw_time(k)    = (starts(k) - 1)/fs + win_N/(2*fs);
        bpm_fft(k)    = est_fft(seg, win_N, nfft, fs, f_low, f_high);
        bpm_welch(k)  = est_welch(seg, win_N, nfft, fs, f_low, f_high);
        bpm_music(k)  = est_music(seg, M_sub, omega_cb, fs, f_low, f_high);
        bpm_esprit(k) = est_esprit(seg, M_sub, fs, f_low, f_high);
        [snr_db(k), det_rate(k), skin_mean(k), lum_cv_win(k)] = ...
            win_quality(seg, detected_t(idx), skin_count_t(idx), lum_t(idx), ...
                        win_N, nfft, fs, f_low, f_high);
    end

    gt_sw  = interp1(t_axis(vld_gt), gt_bpm(vld_gt), sw_time, 'linear', 'extrap');

    % Weighted composite confidence score [0,1] per window.
    % Each factor is linearly mapped: reaches 0.5 at its threshold, 1.0 at 2×threshold.
    snr_score  = min(1, max(0, snr_db      / (SNR_THR    * 2)));
    det_score  = det_rate;
    skin_score = min(1, max(0, skin_mean   / (SKIN_THR   * 2)));
    lum_score  = max(0, 1 - lum_cv_win    / (LUM_CV_THR * 2));
    conf_score = W_CONF(1)*snr_score + W_CONF(2)*det_score + ...
                 W_CONF(3)*skin_score + W_CONF(4)*lum_score;
    hq_win = conf_score >= CONF_THR;

    % Per-factor binary failure flags (diagnostic only — not used in hq_win)
    fail_snr  = snr_db     < SNR_THR;
    fail_det  = det_rate   < DET_THR;
    fail_skin = skin_mean  < SKIN_THR;
    fail_lum  = lum_cv_win > LUM_CV_THR;

    mae_fft    = mean(abs(bpm_fft    - gt_sw));
    mae_welch  = mean(abs(bpm_welch  - gt_sw));
    mae_music  = mean(abs(bpm_music  - gt_sw), 'omitnan');
    mae_esprit = mean(abs(bpm_esprit - gt_sw), 'omitnan');
    acc_fft    = mean(abs(bpm_fft    - gt_sw) <= ACC_THR) * 100;
    acc_welch  = mean(abs(bpm_welch  - gt_sw) <= ACC_THR) * 100;
    acc_music  = mean(abs(bpm_music  - gt_sw) <= ACC_THR, 'omitnan') * 100;
    acc_esprit = mean(abs(bpm_esprit - gt_sw) <= ACC_THR, 'omitnan') * 100;

    fprintf('  win=%ds  N=%d  M=%d  HQ=%d/%d (%.0f%%)  |  LQ causes: SNR=%d  Det=%d  Skin=%d  Lum=%d\n', ...
        win_secs(wi), win_N, M_sub, sum(hq_win), n_sw, mean(hq_win)*100, ...
        sum(fail_snr), sum(fail_det), sum(fail_skin), sum(fail_lum));
    fprintf('  %-6d  %-9.2f  %-9.2f  %-9.2f  %-9.2f  %-9.1f  %-9.1f  %-9.1f  %-9.1f\n', ...
        win_secs(wi), mae_fft, mae_welch, mae_music, mae_esprit, ...
        acc_fft, acc_welch, acc_music, acc_esprit);

    if sum(hq_win) >= 5
        mae_hq_f = mean(abs(bpm_fft(hq_win)    - gt_sw(hq_win)));
        mae_hq_w = mean(abs(bpm_welch(hq_win)   - gt_sw(hq_win)));
        mae_hq_m = mean(abs(bpm_music(hq_win)   - gt_sw(hq_win)), 'omitnan');
        mae_hq_e = mean(abs(bpm_esprit(hq_win)  - gt_sw(hq_win)), 'omitnan');
        acc_hq_f = mean(abs(bpm_fft(hq_win)    - gt_sw(hq_win)) <= ACC_THR) * 100;
        acc_hq_w = mean(abs(bpm_welch(hq_win)   - gt_sw(hq_win)) <= ACC_THR) * 100;
        acc_hq_m = mean(abs(bpm_music(hq_win)   - gt_sw(hq_win)) <= ACC_THR, 'omitnan') * 100;
        acc_hq_e = mean(abs(bpm_esprit(hq_win)  - gt_sw(hq_win)) <= ACC_THR, 'omitnan') * 100;
        fprintf('    HQ only: FFT=%.1f/%.0f%%  Welch=%.1f/%.0f%%  MUSIC=%.1f/%.0f%%  ESPRIT=%.1f/%.0f%%\n', ...
            mae_hq_f,acc_hq_f, mae_hq_w,acc_hq_w, mae_hq_m,acc_hq_m, mae_hq_e,acc_hq_e);
    end

    figure('Name', sprintf('BPM Track — %ds window', win_secs(wi)));

    ax1 = subplot(2,1,1);
    stairs(t_axis(vld_gt), gt_bpm(vld_gt), 'k', 'LineWidth', 2.5);  hold on;
    plot(sw_time, bpm_fft,    'b',   'LineWidth', 1.2);
    plot(sw_time, bpm_welch,  'r',   'LineWidth', 1.2);
    plot(sw_time, bpm_music,  'g',   'LineWidth', 1.5);
    plot(sw_time, bpm_esprit, 'm--', 'LineWidth', 1.5);
    yline(gt_mean,           'k:', sprintf('GT %.1f BPM', gt_mean));
    yline(gt_mean + ACC_THR, 'g:');  yline(gt_mean - ACC_THR, 'g:');
    scatter(sw_time(~hq_win), 52*ones(1,sum(~hq_win)), 20, [0.8 0.2 0.2], 'v', 'filled');
    ylabel('BPM');  ylim([50, 145]);  grid on;
    legend('GT','FFT','Welch','MUSIC','ESPRIT','Low-conf▼','Location','best');
    title(sprintf('Window=%ds  FFT bin=%.2fHz=%.1fBPM/bin', win_secs(wi), fs/win_N, fs/win_N*60));

    ax2 = subplot(2,1,2);
    % Individual factor scores as thin dotted lines (background context)
    plot(sw_time, snr_score,  'b:', 'LineWidth', 1.0);  hold on;
    plot(sw_time, det_score,  'g:', 'LineWidth', 1.0);
    plot(sw_time, lum_score,  'r:', 'LineWidth', 1.0);
    plot(sw_time, skin_score, 'k:', 'LineWidth', 1.0);
    % Composite score as prominent solid line
    plot(sw_time, conf_score, 'm', 'LineWidth', 2.5);
    % HQ region shading under the composite curve where it exceeds threshold
    fill([sw_time, fliplr(sw_time)], ...
         [conf_score .* double(hq_win), zeros(1,n_sw)], ...
         'c', 'FaceAlpha', 0.20, 'EdgeColor', 'none');
    yline(CONF_THR, 'm--', sprintf('HQ≥%.2f', CONF_THR), 'LineWidth', 1.5, ...
          'LabelVerticalAlignment', 'bottom');
    ylim([0, 1.1]);  xlabel('Time (s)');  ylabel('Score [0–1]');
    legend('SNR (w=0.40)', 'Det (w=0.30)', 'Lum (w=0.15)', 'Skin (w=0.15)', ...
           'Composite', 'HQ region', 'Location', 'best');
    title(sprintf('Confidence: 0.40·SNR + 0.30·det + 0.15·lum + 0.15·skin  |  HQ ≥ %.2f', CONF_THR));
    grid on;
    linkaxes([ax1 ax2], 'x');
end

fprintf('%s\n', repmat('=', 1, 100));
fprintf('  MAE (BPM) and Acc (%% within ±%d BPM) vs ground truth\n', ACC_THR);
fprintf('%s\n', repmat('=', 1, 100));

% ── Local functions ──────────────────────────────────────────────────────────

function bpm = peak_bpm(spectrum, freqs, f_low, f_high)
% Dominant cardiac peak with sub-harmonic correction.
% rPPG BVP waveforms often show a strong half-frequency artifact from the
% dicrotic notch. Threshold = 0.05: if the 2× frequency has even 5% of the
% sub-harmonic's amplitude, prefer it (aggressive correction needed because
% the cardiac component is typically 5-15× weaker than the sub-harmonic).
    band = freqs >= f_low & freqs <= f_high;
    [peak_val, idx] = max(spectrum(band));
    fb = freqs(band);  fp = fb(idx);
    if fp < 1.2 && 2*fp <= f_high
        [~, i2] = min(abs(freqs - 2*fp));
        if spectrum(i2) > 0.05 * peak_val;  fp = 2*fp;  end
    end
    bpm = fp * 60;
end

function bpm = est_fft(seg, win_N, nfft, fs, f_low, f_high)
    X  = abs(fft(seg .* hann(win_N), nfft));
    fv = (0:nfft/2) * fs / nfft;
    bpm = peak_bpm(X(1:nfft/2+1), fv, f_low, f_high);
end

function bpm = est_welch(seg, win_N, nfft, fs, f_low, f_high)
    [P, fw] = pwelch(seg, hann(win_N), floor(win_N/2), nfft, fs);
    bpm = peak_bpm(P, fw, f_low, f_high);
end

function bpm = est_music(seg, M, omega_cb, fs, f_low, f_high)
% MUSIC with p=4: models both the sub-harmonic (~0.87 Hz) and cardiac (~1.72 Hz)
% as signal components (2 real sinusoids = 4 complex exponentials).
% With p=2 the sub-harmonic monopolises the signal subspace and cardiac lands
% in the noise subspace, making the pseudospectrum blind to the cardiac peak.
    bpm = NaN;
    if length(seg) < M || M < 5;  return;  end
    try
        Px = m_music(seg(:), 4, M, omega_cb);
    catch
        return;
    end
    f_cb = omega_cb / (2*pi) * fs;
    band = f_cb >= f_low & f_cb <= f_high;
    Px_b = Px(band);  fb = f_cb(band);
    [peak_val, idx] = max(Px_b);  fp = fb(idx);
    if fp < 1.2 && 2*fp <= f_high
        % Search ±0.2 Hz neighbourhood around 2×fp — single-point check misses
        % peaks that land slightly off-grid from the exact harmonic location.
        near = f_cb >= (2*fp - 0.2) & f_cb <= (2*fp + 0.2);
        if any(near)
            [nb_val, nb_idx] = max(Px(near));
            if nb_val > peak_val - 10
                nb_f = f_cb(near);
                fp   = nb_f(nb_idx);   % use actual peak near 2×fp
            end
        end
    end
    bpm = fp * 60;
end

function bpm = est_esprit(seg, M, fs, f_low, f_high)
% ESPRIT with p=4: models sub-harmonic and cardiac as separate sinusoids.
% Returns up to 2 frequencies in the cardiac band; if both 0.87 Hz and 1.72 Hz
% appear, picks the higher one (cardiac). Falls back to harmonic correction
% if only one frequency is returned.
    bpm = NaN;
    if length(seg) < M || M < 5;  return;  end
    try
        [~, w_est] = evalc('m_esprit(seg(:), 4, M)');
    catch
        return;
    end
    hz = sort(real(w_est) * fs / (2*pi));
    hz = hz(hz >= f_low & hz <= f_high);
    if isempty(hz);  return;  end
    if numel(hz) >= 2 && hz(end)/hz(1) > 1.5
        bpm = hz(end) * 60;   % two distinct freqs, ratio >1.5: higher is cardiac
    else
        fp = hz(1);
        if fp < 1.2 && 2*fp <= f_high;  fp = 2*fp;  end   % fallback harmonic correction
        bpm = fp * 60;
    end
end

function [snr_db, det_rate, skin_mean, lum_cv] = ...
        win_quality(seg, det_seg, skin_seg, lum_seg, win_N, nfft, fs, f_low, f_high)
% Four per-window quality factors:
%   snr_db    — cardiac peak PSD vs in-band noise floor (dB); low → noisy signal
%   det_rate  — fraction of frames with fresh MTCNN detection; low → stale bbox
%   skin_mean — mean skin pixel count; low → face partially occluded or mis-detected
%   lum_cv    — luminance coefficient of variation; high → AGC/lighting instability
    [P, fw]    = pwelch(seg, hann(win_N), floor(win_N/2), nfft, fs);
    band       = fw >= f_low & fw <= f_high;
    P_band     = P(band);  f_band = fw(band);
    [pk, pi]   = max(P_band);
    noise_mask = abs(f_band - f_band(pi)) >= 0.5;
    if any(noise_mask)
        noise_P = mean(P_band(noise_mask));
    else
        noise_P = mean(P_band);
    end
    snr_db    = 10*log10(max(pk / max(noise_P, eps), 1));
    det_rate  = mean(double(det_seg));
    skin_mean = mean(double(skin_seg));
    mu_lum    = mean(lum_seg);
    lum_cv    = std(lum_seg) / max(mu_lum, eps);
end

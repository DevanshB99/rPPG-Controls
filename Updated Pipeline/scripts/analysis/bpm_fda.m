clear; close all; clc;

addpath('/home/macs/Downloads/macs-matlab-toolbox-master/macs-matlab-toolbox-master');
addpath('/home/macs/Downloads/MUSIC-ESPRIT-Frequency-ID-main/MUSIC-ESPRIT-Frequency-ID-main');

% ── Auto-save setup: results/fda_results/fda_<timestamp>/ ────────────────
pipeline_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
fda_ts   = datestr(now, 'yyyymmdd_HHMMSS');
fda_dir  = fullfile(pipeline_dir, 'results', 'fda_results', sprintf('fda_%s', fda_ts));
mkdir(fda_dir);
diary(fullfile(fda_dir, 'output.txt'));
diary on;

% ── Load latest filterdesign output from results/filter_results/ ─────────
filter_dir = fullfile(pipeline_dir, 'results', 'filter_results');
csv_files  = dir(fullfile(filter_dir, 'filterdesign_*.csv'));
if isempty(csv_files)
    error('No filterdesign CSV found in filter_results/. Run bpm_filterdesign.m first.');
end
[~, newest] = max([csv_files.datenum]);
csv_path    = fullfile(filter_dir, csv_files(newest).name);
fprintf('Loading: %s\n', csv_path);

data           = readtable(csv_path);
t_axis         = data.time_s;
fs             = 1 / median(diff(t_axis));
T              = height(data);

% Primary signal: Hamming N_mid tight passband (fp1=1.0 Hz) — best performer
S_primary      = data.BVP_ham_tight;   % winner from filter design
S_ham_adapt    = data.BVP_ham_adapt;   % Hamming N_mid, adaptive fp1
S_el_tight     = data.BVP_el_tight;   % Elliptic tight, for comparison
S_el_adapt     = data.BVP_el_adapt;   % Elliptic adaptive fp1
f_low_adapt    = data.f_p1_adapt(1);  % adaptive lower cutoff (for ham_adapt/el_adapt)

lum_t          = data.frame_luminance;
detected_t     = data.face_detected;
skin_count_t   = data.skin_pixel_count;

gt_bpm  = data.gt_bpm;
vld_gt  = ~isnan(gt_bpm);
gt_mean = mean(gt_bpm(vld_gt));

% Passband: tight lower edge (1.0 Hz = 60 BPM) removes motion/respiration;
% upper edge from filterdesign's adaptive estimate stored in the CSV.
f_low  = 1.0;                   % tight fp1 matching BVP_ham_tight filter
f_high = data.f_p2_adapt(1);    % adaptive upper cutoff from filterdesign

fprintf('Loaded %d frames  fs=%.2f Hz  GT coverage=%.0f%%  GT mean=%.1f BPM\n', ...
    T, fs, 100*mean(vld_gt), gt_mean);
fprintf('Cardiac band: [%.2f, %.2f] Hz = [%.0f, %.0f] BPM\n', ...
    f_low, f_high, f_low*60, f_high*60);

% ── Section W: Noise whiteness diagnostic ────────────────────────────────────
[P_w, F_w] = pwelch(S_primary, hann(min(round(fs*10), T)), [], 4096, fs);
band_w = F_w >= f_low & F_w <= f_high;
[~, pi_w] = max(P_w(band_w));  F_bw = F_w(band_w);
f_card = F_bw(pi_w);
if f_card < 1.1 && 2*f_card <= f_high;  f_card = 2*f_card;  end

bw_notch = 0.30;
N_notch  = 2*round(1.5*fs) + 1;
b_ns = fir1(N_notch-1, ...
    [max(f_low+0.05, f_card-bw_notch), min(f_high-0.05, f_card+bw_notch)] / (fs/2), ...
    'stop', hamming(N_notch));
S_noise = filtfilt(b_ns, 1, S_primary);

LB_LAGS    = min(20, floor(T/5));
[Q_lb, p_lb] = ljung_box_local(S_noise, LB_LAGS);
is_colored   = p_lb < 0.05;

M_ev  = min(floor(T/4), 60);
x_ev  = S_noise - mean(S_noise);
X_ev  = hankel(x_ev(1:M_ev), x_ev(M_ev:end));
R_ev  = (X_ev * X_ev') / size(X_ev, 2);
ev    = sort(real(eig(R_ev)), 'descend');
ev_ns = ev(5:end);
cv_ev = std(ev_ns) / (mean(ev_ns) + 1e-9);

fprintf('\n── Section W: Noise Whiteness ──────────────────────────────────────\n');
fprintf('  Cardiac fundamental : %.2f Hz (%.0f BPM)\n', f_card, f_card*60);
fprintf('  Ljung-Box (%d lags) : Q=%.1f  p=%.4f  → %s\n', LB_LAGS, Q_lb, p_lb, ...
    iif_str(is_colored, 'COLORED NOISE → AR pre-whitening applied', 'white noise → no pre-whitening needed'));
fprintf('  Eigenvalue CV       : %.3f (M=%d)  → %s\n', cv_ev, M_ev, ...
    iif_str(cv_ev > 0.30, 'non-uniform — colored noise confirmed', 'near-uniform — approximately white'));
fprintf('────────────────────────────────────────────────────────────────────\n\n');

conf_lb = 1.96 / sqrt(T);
max_lag = min(60, floor(T/4));
[acf_n, lags_n] = acf_local(S_noise, max_lag);
[P_n, F_n]      = pwelch(S_noise, hann(min(round(fs*5), length(S_noise))), [], 4096, fs);

figure('Name', 'Fig W1 — Noise Whiteness Diagnostic');
subplot(2,2,1);
plot(F_w, 10*log10(P_w+1e-30), 'b', 'LineWidth', 1.2);
xline(f_card, 'r--', sprintf('f_c=%.2fHz', f_card), 'LineWidth', 1.5);
xline(f_low, 'k--');  xline(f_high, 'k--');
xlabel('Hz');  ylabel('dB/Hz');  title('PSD — S_{primary}');
grid on;  xlim([0, f_high+1]);

subplot(2,2,2);
plot(F_n, 10*log10(P_n+1e-30), 'g', 'LineWidth', 1.2);
xline(f_low, 'k--');  xline(f_high, 'k--');
xlabel('Hz');  ylabel('dB/Hz');
title(sprintf('PSD — noise floor (cardiac notched ±%.1fHz)', bw_notch));
grid on;  xlim([0, f_high+1]);

subplot(2,2,3);
stem(lags_n, acf_n, 'g', 'MarkerSize', 3, 'LineWidth', 0.8);  hold on;
yline(conf_lb, 'r--');  yline(-conf_lb, 'r--');
xlabel('Lag (samples)');  ylabel('ACF');
title(sprintf('ACF — noise floor  (conf = ±%.3f)', conf_lb));  grid on;

subplot(2,2,4);
semilogy(1:length(ev), ev, 'bo-', 'MarkerSize', 4, 'LineWidth', 1.2);  hold on;
semilogy(5:length(ev), ev_ns, 'rs', 'MarkerSize', 4);
xline(4.5, 'k--', 'signal|noise', 'LineWidth', 1.2);
yline(mean(ev_ns), 'g--', sprintf('noise mean (CV=%.2f)', cv_ev));
xlabel('Eigenvalue index');  ylabel('Eigenvalue');
title(sprintf('Covariance eigenvalues  M=%d', M_ev));
legend('all', 'noise subspace');  grid on;
sgtitle(sprintf('Noise Whiteness  |  L-B p=%.4f  |  EV-CV=%.3f  |  %s', ...
    p_lb, cv_ev, iif_str(is_colored, 'COLORED NOISE', 'white noise')), 'FontWeight', 'bold');

% ── Section A: Full-signal spectra ──────────────────────────────────────────
specCal(S_primary, fs);

sce = specCale([S_primary, S_ham_adapt, S_el_tight, S_el_adapt], fs);
figure('Name', 'specCale — Tight vs Adapt');
plot(sce.f(:,1), sce.amp, 'LineWidth', 1.2);
xline(f_low,       'b-',  sprintf('f_p1 tight=%.1fHz', f_low),       'LineWidth', 1.5);
xline(f_low_adapt, 'r--', sprintf('f_p1 adapt=%.2fHz', f_low_adapt), 'LineWidth', 1.5);
xline(f_high,      'k--', sprintf('f_p2=%.2fHz', f_high));
xlim([0, f_high + 1]);
legend('Ham-tight (PRIMARY)','Ham-adapt','El-tight','El-adapt','Location','best');
xlabel('Frequency (Hz)');  ylabel('Amplitude');
title('Amplitude Spectra — tight (1.0 Hz) vs adaptive passband lower edge');  grid on;

% ── Section B: Sliding-window FFT / Welch / MUSIC / ESPRIT ──────────────────
win_secs      = [2, 3, 5, 10, 20];
nfft          = 4096;
P_AR          = 8;      % AR pre-whitening order (Burg method)
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

fprintf('\n%s\n', repmat('=', 1, 110));
fprintf('  %-5s  %-11s  %-11s  %-11s  %-11s  %-10s  %-10s\n', ...
    'Win(s)', 'HamTight', 'HamAdapt', 'ElTight', 'ElAdapt', 'MUSIC', 'ESPRIT');
fprintf('  %-5s  %-11s  %-11s  %-11s  %-11s  %-10s  %-10s\n', ...
    '', '(Welch/MAE)', '(Welch/MAE)', '(Welch/MAE)', '(Welch/MAE)', '(HamT/MAE)', '(HamT/MAE)');
fprintf('%s\n', repmat('=', 1, 110));

for wi = 1:numel(win_secs)
    win_N = round(win_secs(wi) * fs);
    if win_N > T;  continue;  end

    M_sub  = max(round(win_N / 4), 6);  % M > max_p+1; MDL selects p in {2..6}
    starts = 1 : round(fs) : T - win_N + 1;
    n_sw   = numel(starts);

    sw_time      = zeros(1, n_sw);
    bpm_welch    = zeros(1, n_sw);   % Welch on Ham-tight (primary)
    bpm_welch_ha = zeros(1, n_sw);   % Welch on Ham-adapt
    bpm_welch_et = zeros(1, n_sw);   % Welch on El-tight
    bpm_welch_ea = zeros(1, n_sw);   % Welch on El-adapt
    bpm_music    = nan(1, n_sw);     % MUSIC        on Ham-tight
    bpm_esprit   = nan(1, n_sw);     % ESPRIT       on Ham-tight
    bpm_music_ar = nan(1, n_sw);     % MUSIC+AR-white  on Ham-tight
    bpm_esprit_ar= nan(1, n_sw);     % ESPRIT+AR-white on Ham-tight
    snr_db       = zeros(1, n_sw);
    det_rate    = zeros(1, n_sw);
    skin_mean   = zeros(1, n_sw);
    lum_cv_win  = zeros(1, n_sw);

    for k = 1:n_sw
        idx           = starts(k) : starts(k) + win_N - 1;
        seg           = S_primary(idx);
        sw_time(k)      = (starts(k) - 1)/fs + win_N/(2*fs);
        bpm_welch(k)    = est_welch(seg,              win_N, nfft, fs, f_low,       f_high);
        bpm_welch_ha(k) = est_welch(S_ham_adapt(idx), win_N, nfft, fs, f_low_adapt, f_high);
        bpm_welch_et(k) = est_welch(S_el_tight(idx),  win_N, nfft, fs, f_low,       f_high);
        bpm_welch_ea(k) = est_welch(S_el_adapt(idx),  win_N, nfft, fs, f_low_adapt, f_high);
        p_mdl             = mdl_order_local(seg, M_sub, 6);
        seg_w             = ar_prewhiten(seg, fs, f_low, f_high, P_AR);
        bpm_music(k)      = est_music(seg,   p_mdl,        M_sub, omega_cb, fs, f_low, f_high);
        bpm_esprit(k)     = est_esprit(seg,  max(4,p_mdl), M_sub, fs, f_low, f_high);
        bpm_music_ar(k)   = est_music(seg_w, p_mdl,        M_sub, omega_cb, fs, f_low, f_high);
        bpm_esprit_ar(k)  = est_esprit(seg_w,max(4,p_mdl), M_sub, fs, f_low, f_high);
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

    mae_welch   = mean(abs(bpm_welch    - gt_sw));
    mae_wch_ha  = mean(abs(bpm_welch_ha - gt_sw));
    mae_wch_et  = mean(abs(bpm_welch_et - gt_sw));
    mae_wch_ea  = mean(abs(bpm_welch_ea - gt_sw));
    mae_music    = mean(abs(bpm_music    - gt_sw), 'omitnan');
    mae_esprit   = mean(abs(bpm_esprit  - gt_sw), 'omitnan');
    mae_music_ar = mean(abs(bpm_music_ar  - gt_sw), 'omitnan');
    mae_esprit_ar= mean(abs(bpm_esprit_ar - gt_sw), 'omitnan');
    acc_welch    = mean(abs(bpm_welch    - gt_sw) <= ACC_THR) * 100;
    acc_wch_ha   = mean(abs(bpm_welch_ha - gt_sw) <= ACC_THR) * 100;
    acc_wch_et   = mean(abs(bpm_welch_et - gt_sw) <= ACC_THR) * 100;
    acc_wch_ea   = mean(abs(bpm_welch_ea - gt_sw) <= ACC_THR) * 100;
    acc_music    = mean(abs(bpm_music    - gt_sw) <= ACC_THR, 'omitnan') * 100;
    acc_esprit   = mean(abs(bpm_esprit   - gt_sw) <= ACC_THR, 'omitnan') * 100;
    acc_music_ar = mean(abs(bpm_music_ar  - gt_sw) <= ACC_THR, 'omitnan') * 100;
    acc_esprit_ar= mean(abs(bpm_esprit_ar - gt_sw) <= ACC_THR, 'omitnan') * 100;

    fprintf('  win=%ds  N=%d  M=%d  HQ=%d/%d (%.0f%%)  |  LQ: SNR=%d  Det=%d  Skin=%d  Lum=%d\n', ...
        win_secs(wi), win_N, M_sub, sum(hq_win), n_sw, mean(hq_win)*100, ...
        sum(fail_snr), sum(fail_det), sum(fail_skin), sum(fail_lum));
    fprintf('  %-5d  %-11.2f  %-11.2f  %-11.2f  %-11.2f  %-10.2f  %-10.2f   [MAE BPM]\n', ...
        win_secs(wi), mae_welch, mae_wch_ha, mae_wch_et, mae_wch_ea, mae_music, mae_esprit);
    fprintf('  %-5s  %-11.1f  %-11.1f  %-11.1f  %-11.1f  %-10.1f  %-10.1f   [Acc %%]\n', ...
        '', acc_welch, acc_wch_ha, acc_wch_et, acc_wch_ea, acc_music, acc_esprit);
    fprintf('  %-5s  AR-white:  MUSIC=%.2f(%.0f%%)  ESPRIT=%.2f(%.0f%%)  [MAE / Acc]\n', ...
        '', mae_music_ar, acc_music_ar, mae_esprit_ar, acc_esprit_ar);

    if sum(hq_win) >= 5
        mae_hq_w   = mean(abs(bpm_welch(hq_win)     - gt_sw(hq_win)));
        mae_hq_ha  = mean(abs(bpm_welch_ha(hq_win)  - gt_sw(hq_win)));
        mae_hq_et  = mean(abs(bpm_welch_et(hq_win)  - gt_sw(hq_win)));
        mae_hq_ea  = mean(abs(bpm_welch_ea(hq_win)  - gt_sw(hq_win)));
        mae_hq_m   = mean(abs(bpm_music(hq_win)     - gt_sw(hq_win)), 'omitnan');
        mae_hq_e   = mean(abs(bpm_esprit(hq_win)    - gt_sw(hq_win)), 'omitnan');
        mae_hq_mar = mean(abs(bpm_music_ar(hq_win)  - gt_sw(hq_win)), 'omitnan');
        mae_hq_ear = mean(abs(bpm_esprit_ar(hq_win) - gt_sw(hq_win)), 'omitnan');
        fprintf('    HQ: HT=%.1f  HA=%.1f  ET=%.1f  EA=%.1f  MUS=%.1f  ESP=%.1f  [MAE BPM]\n', ...
            mae_hq_w, mae_hq_ha, mae_hq_et, mae_hq_ea, mae_hq_m, mae_hq_e);
        fprintf('    HQ-AR: MUS=%.1f  ESP=%.1f  [MAE BPM]\n', mae_hq_mar, mae_hq_ear);
    end

    figure('Name', sprintf('BPM Track — %ds window', win_secs(wi)));

    ax1 = subplot(2,1,1);
    stairs(t_axis(vld_gt), gt_bpm(vld_gt), 'k', 'LineWidth', 2.5);  hold on;
    plot(sw_time, bpm_welch,    'b',    'LineWidth', 2.0);   % Ham-tight Welch
    plot(sw_time, bpm_welch_ha, 'b--',  'LineWidth', 1.2);   % Ham-adapt Welch
    plot(sw_time, bpm_welch_et, 'r',    'LineWidth', 2.0);   % El-tight Welch
    plot(sw_time, bpm_welch_ea, 'r--',  'LineWidth', 1.2);   % El-adapt Welch
    plot(sw_time, bpm_music,     'g',    'LineWidth', 1.8);   % MUSIC on Ham-tight
    plot(sw_time, bpm_esprit,    'm--',  'LineWidth', 1.5);   % ESPRIT on Ham-tight
    plot(sw_time, bpm_music_ar,  'g--',  'LineWidth', 1.5);   % MUSIC+AR-white
    plot(sw_time, bpm_esprit_ar, 'm',    'LineWidth', 1.5);   % ESPRIT+AR-white
    yline(gt_mean,           'k:', sprintf('GT %.1f BPM', gt_mean));
    yline(gt_mean + ACC_THR, 'g:');  yline(gt_mean - ACC_THR, 'g:');
    scatter(sw_time(~hq_win), 52*ones(1,sum(~hq_win)), 20, [0.8 0.2 0.2], 'v', 'filled');
    ylabel('BPM');  ylim([50, 145]);  grid on;
    legend('GT','HT-Welch','HA-Welch','ET-Welch','EA-Welch','MUSIC','ESPRIT','MUSIC-AR','ESP-AR','Low-conf▼', ...
           'Location','best');
    title(sprintf('Win=%ds | solid=tight(1.0Hz) dashed=adapt(%.2fHz) | MUSIC/ESPRIT on HamTight', ...
        win_secs(wi), f_low_adapt));

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

function bpm = est_music(seg, p, M, omega_cb, fs, f_low, f_high)
    bpm = NaN;
    if length(seg) < M || M < 5;  return;  end
    try
        Px = m_music(seg(:), p, M, omega_cb);
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

function bpm = est_esprit(seg, p, M, fs, f_low, f_high)
    bpm = NaN;
    if length(seg) < M || M < 5;  return;  end
    try
        [~, w_est] = evalc('m_esprit(seg(:), p, M)');
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

function [Q, p] = ljung_box_local(x, m)
    N = length(x);
    x = x - mean(x);
    v = var(x);
    if v < 1e-30;  Q = 0;  p = 1;  return;  end
    x = x / sqrt(v);
    Q = 0;
    for lag = 1:m
        r = dot(x(1:N-lag), x(lag+1:N)) / N;
        Q = Q + r^2 / (N - lag);
    end
    Q = N * (N + 2) * Q;
    p = 1 - chi2cdf(Q, m);
end

function [rho, lags] = acf_local(x, max_lag)
    x = x - mean(x);
    if var(x) < 1e-30;  rho = zeros(max_lag+1,1);  lags = (0:max_lag)';  return;  end
    [c, lgs] = xcorr(x, max_lag, 'coeff');
    idx  = lgs >= 0;
    rho  = c(idx);
    lags = lgs(idx)';
end

function s = iif_str(cond, a, b)
    if cond;  s = a;  else;  s = b;  end
end

function k_opt = mdl_order_local(x, M, max_k)
    x  = x(:) - mean(x);
    N  = length(x) - M;
    if N < 5;  k_opt = 2;  return;  end
    X  = hankel(x(1:M), x(M:end));
    R  = (X * X') / N;
    ev = sort(real(eig(R)), 'descend');
    best = Inf;  k_opt = 2;
    for k = 0 : 2 : max_k
        n_n = M - k;
        if n_n < 3;  break;  end
        lam = max(ev(k+1:end), 1e-30);
        gm  = exp(mean(log(lam)));
        am  = mean(lam);
        if am < 1e-30;  break;  end
        mdl = -N * n_n * log(gm/am) + 0.5*k*(2*M-k)*log(N);
        if mdl < best;  best = mdl;  k_opt = max(2, k);  end
    end
end

function seg_w = ar_prewhiten(seg, fs, f_low, f_high, p_ar)
    x = seg(:);
    N = length(x);
    if N < p_ar + 10;  seg_w = x;  return;  end
    nfft_w    = max(512, 2^nextpow2(4*N));
    [P, F]    = pwelch(x, hann(N), 0, nfft_w, fs);
    band      = F >= f_low & F <= f_high;
    [~, pi_c] = max(P(band));
    Fb        = F(band);
    f_c       = Fb(pi_c);
    if f_c < 1.1 && 2*f_c <= f_high;  f_c = 2*f_c;  end
    f_n       = min(max(f_c / (fs/2), 0.01), 0.99);
    [b_n, a_n, ~] = designNotchPeakIIR('CenterFrequency', f_n, 'Bandwidth', f_n/8);
    x_noise   = filtfilt(b_n, a_n, x);
    [a_ar, ~] = arburg(x_noise, p_ar);
    seg_w     = filter([1, a_ar], 1, x);
end

% ── Auto-save figures and close diary ────────────────────────────────────────
fig_handles = findall(0, 'Type', 'figure');
for fh = fig_handles'
    fname = get(fh, 'Name');
    if isempty(fname);  fname = sprintf('figure_%d', get(fh,'Number'));  end
    fname = regexprep(fname, '[^\w -]', '_');
    saveas(fh, fullfile(fda_dir, [fname '.png']));
    saveas(fh, fullfile(fda_dir, [fname '.pdf']));
end
fprintf('\nFigures saved → %s\n', fda_dir);
diary off;

clear; close all; clc;

% ── Source selection ──────────────────────────────────────────────────────────
% Set USE_PYTHON_CSV = true  → read BVP_detrended from the Python pipeline
%                              (BiSeNet + MediaPipe + adaptive skin extraction)
% Set USE_PYTHON_CSV = false → extract RGB from the raw video file with the
%                              simpler Viola-Jones + fixed-YCbCr MATLAB pipeline
USE_PYTHON_CSV = true;

pipeline_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
input_dir    = fullfile(pipeline_dir, 'results', 'input_results');
run_dirs     = dir(input_dir);
run_dirs     = run_dirs([run_dirs.isdir] & ~ismember({run_dirs.name}, {'.','..'}));
if isempty(run_dirs)
    error('No input_results found. Run rppg_pipeline_refined.py first.');
end
[~, latest_idx] = max([run_dirs.datenum]);
PYTHON_CSV = fullfile(input_dir, run_dirs(latest_idx).name, 'rppg_output.csv');
VID_PATH   = '/home/macs/Documents/rPPG-Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/recording_2026-04-07T22-04-35Z.mov';
CSV_PATH   = '/home/macs/Documents/rPPG-Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/vitals.csv';

% ── Ground truth (same source regardless of signal path) ─────────────────────
csv_data      = readtable(CSV_PATH);
gt_time       = csv_data.offset_seconds;
gt_hr         = csv_data.heart_rate;
valid         = ~isnan(gt_hr);
gt_time       = gt_time(valid);  gt_hr = gt_hr(valid);
bpm_gt_mean   = mean(gt_hr);
bpm_gt_median = median(gt_hr);
fprintf('GT: mean=%.1f  median=%.1f  range=[%d %d]  n=%d\n', ...
    bpm_gt_mean, bpm_gt_median, min(gt_hr), max(gt_hr), numel(gt_hr));

% ── Signal extraction ─────────────────────────────────────────────────────────
if USE_PYTHON_CSV
    % ── Path A: use superior Python pipeline output ───────────────────────────
    % Reads pre-computed BVP_detrended (CHROM + linear detrend already applied).
    % R_normalized / G_normalized / B_normalized are also available if you need
    % to recompute CHROM or inspect individual channels.
    py_data = readtable(PYTHON_CSV);

    t_axis = py_data.time_s;
    T      = height(py_data);
    fs     = 1 / median(diff(t_axis));

    % CHROM BVP signal after linear detrending — use directly as S_det
    S_det  = py_data.BVP_detrended;

    fprintf('Source: Python CSV (%d frames, fs=%.4f Hz)\n', T, fs);
    fprintf('  Columns available: %s\n', strjoin(py_data.Properties.VariableNames, ', '));

else
    % ── Path B: MATLAB extraction from raw video (Viola-Jones + fixed YCbCr) ─
    vid      = VideoReader(VID_PATH);
    fs       = vid.FrameRate;
    H_vid    = vid.Width;   % dimensions swap after 90° CW rotation
    W_vid    = vid.Height;
    detector = vision.CascadeObjectDetector('MinSize', [80 80]);
    bbox     = [];
    R_t=[]; G_t=[]; B_t=[];

    while hasFrame(vid)
        frame = rot90(readFrame(vid), 3);
        boxes = step(detector, frame);
        if ~isempty(boxes)
            [~,i] = max(boxes(:,3) .* boxes(:,4));
            b    = boxes(i,:);
            bbox = [max(b(1),1) max(b(2),1) min(b(3),W_vid-b(1)) min(b(4),H_vid-b(2))];
        end
        if isempty(bbox); continue; end
        fc    = imcrop(frame, bbox);
        fcd   = double(fc);
        lum   = mean(fcd(:));  if lum > 0; fcd = fcd/lum*128; end
        Yf  =  0.299*fcd(:,:,1)    + 0.587*fcd(:,:,2)    + 0.114*fcd(:,:,3);
        Cbf = -0.168736*fcd(:,:,1) - 0.331264*fcd(:,:,2) + 0.5*fcd(:,:,3)      + 128;
        Crf =  0.5*fcd(:,:,1)      - 0.418688*fcd(:,:,2) - 0.081312*fcd(:,:,3) + 128;
        Mf  = (Cbf>=77)&(Cbf<=127)&(Crf>=133)&(Crf<=173)&(Yf>40);
        if sum(Mf(:)) < 50;  continue;  end
        pix = reshape(fcd,[],3);  msk = Mf(:);
        R_t(end+1) = mean(pix(msk,1)); %#ok<SAGROW>
        G_t(end+1) = mean(pix(msk,2)); %#ok<SAGROW>
        B_t(end+1) = mean(pix(msk,3)); %#ok<SAGROW>
    end

    T      = length(R_t);
    t_axis = (0:T-1)'/fs;

    R_n = R_t/mean(R_t);  G_n = G_t/mean(G_t);  B_n = B_t/mean(B_t);
    Xs  = 3*R_n - 2*G_n;
    Ys  = 1.5*R_n + G_n - 1.5*B_n;
    alpha = std(Xs)/std(Ys);
    S     = Xs - alpha*Ys;

    t_vec  = (1:T)';
    coeffs = [t_vec, ones(T,1)] \ S(:);
    S_det  = S(:) - (coeffs(1)*t_vec + coeffs(2));

    fprintf('Source: MATLAB video extraction\n');
end

t_axis = t_axis(:);  % column vector — required by filtfilt and plot
S_det  = S_det(:);
fprintf('Signal: T=%d frames  fs=%.4f Hz\n', T, fs);

% ── Adaptive passband estimation from S_det spectrum ──────────────────────
% Derive f_p1, f_p2 from THIS recording's spectrum rather than hardcoded
% constants.  Order matters: cardiac estimate must come before artifact
% detection because the half-harmonic search uses f_cardiac/2.

% Step 1: broad Welch PSD (30 s window or full signal if shorter).
N_rough            = min(round(fs * 30), T);
[P_rough, F_rough] = pwelch(S_det, hann(N_rough), floor(N_rough/2), 8192, fs);
df_rough           = F_rough(2) - F_rough(1);   % frequency resolution

% Step 2: rough cardiac estimate using a fixed 0.67–3.5 Hz search window.
%   Must be computed BEFORE artifact detection, which needs f_cardiac/2.
broad_mask  = (F_rough >= 0.67) & (F_rough <= 3.5);
[~, hf_i]  = max(P_rough(broad_mask));
F_hf        = F_rough(broad_mask);
f_card_raw  = F_hf(hf_i);
bpm_rough   = f_card_raw * 60;
if bpm_rough < 60 && (f_card_raw * 2) <= 3.5
    bpm_rough = bpm_rough * 2;    % harmonic-correct the rough cardiac estimate
end

% Cardiac peak power (for S/A ratio)
[pv_card, ~] = max(P_rough(broad_mask));

% Step 3: find the HALF-HARMONIC ARTIFACT — the dangerous one for rPPG.
%
%   IIR filter failures follow one pattern: a spectral peak near f_cardiac/2
%   is mistaken for the cardiac frequency.  The half-harmonic artifact is
%   dangerous exactly because it sits at f_cardiac/2 inside the passband.
%
%   Strategy A (targeted): look within ±0.3 Hz of f_cardiac/2.
%                          Accept if power ≥ 20% of cardiac peak.
%   Strategy B (fallback): if no half-harmonic found, use the highest peak
%                          in the general 0.5–1.5 Hz motion zone.
f_half         = (bpm_rough / 2) / 60;          % Hz — target centre
hh_lo          = max(0.40, f_half - 0.30);
hh_hi          = min(1.50, f_half + 0.30);
hh_mask        = (F_rough >= hh_lo) & (F_rough <= hh_hi);
artifact_found = false;

if any(hh_mask)
    [pv_hh, hh_i] = max(P_rough(hh_mask));
    F_hh = F_rough(hh_mask);
    if pv_hh >= 0.20 * pv_card
        f_artifact     = F_hh(hh_i);
        artifact_found = true;
        fprintf('  Half-harmonic artifact: %.2f Hz (%.0f BPM)  [f_c/2 = %.2f Hz]\n', ...
            f_artifact, f_artifact*60, f_half);
    end
end

if ~artifact_found
    motion_mask = (F_rough > 0.50) & (F_rough < 1.50);
    [~, mi]     = max(P_rough(motion_mask));
    F_motion    = F_rough(motion_mask);
    f_artifact  = F_motion(mi);
    fprintf('  Motion-zone artifact : %.2f Hz (%.0f BPM)  [no half-harmonic found]\n', ...
        f_artifact, f_artifact*60);
end

% Signal-to-artifact ratio: cardiac peak vs artifact peak power.
art_bin     = max(1, round(f_artifact / df_rough));
pv_art      = P_rough(art_bin);
sa_ratio_dB = 10 * log10(pv_card / (pv_art + 1e-30));

% Lower passband — 0.25 Hz above the detected artifact, floor at 0.67 Hz.
f_p1 = max(0.67, f_artifact + 0.25);
f_s1 = max(0.30, f_p1 - 0.35);

% Report S/A with context: artifact below f_p1 will be filtered out (not critical).
% Artifact above f_p1 would overlap the passband — that is the dangerous case.
if f_artifact < f_p1
    if sa_ratio_dB < 0
        fprintf('  Note: S/A = %.1f dB — artifact at %.2f Hz is stronger than cardiac peak,\n', ...
            sa_ratio_dB, f_artifact);
        fprintf('        but artifact is BELOW f_p1=%.2f Hz and will be filtered out.\n', f_p1);
        fprintf('        IIR risk is low; FIR Hamming N_mid recommended as primary filter.\n');
    else
        fprintf('  S/A = +%.1f dB — cardiac peak dominates; artifact below passband.\n', sa_ratio_dB);
    end
else
    fprintf('  WARNING: S/A = %.1f dB — artifact at %.2f Hz is INSIDE passband (f_p1=%.2f Hz).\n', ...
        sa_ratio_dB, f_artifact, f_p1);
    fprintf('           All filters will struggle; consider improving skin extraction quality.\n');
end

% Step 5: upper passband — 1.5 Hz margin above cardiac estimate.
f_p2 = min(fs/2 - 0.5, max(bpm_rough/60 + 1.50, 3.0));
f_s2 = min(fs/2 - 0.1, f_p2 + 0.80);

% Step 6: sliding-window length — 10 s default; 5 s for recordings < 30 s.
sw_len_s = 10;
if T/fs < 30;  sw_len_s = 5;  end

fprintf('\nAdaptive spec  (derived from S_det spectrum):\n');
fprintf('  Rough cardiac est : %.1f BPM\n', bpm_rough);
fprintf('  Passband          : [%.2f, %.2f] Hz  =  [%.0f, %.0f] BPM\n', ...
    f_p1, f_p2, f_p1*60, f_p2*60);
fprintf('  Stopband          : [%.2f, %.2f] Hz\n', f_s1, f_s2);
fprintf('  Sliding window    : %d s\n', sw_len_s);

% ── Ripple / attenuation specs (constant — not signal-dependent) ──────────
Rp   = 1.0;   % dB  max passband ripple
Rs   = 40;    % dB  min stopband attenuation

Wp = [f_p1 f_p2]/(fs/2);
Ws = [f_s1 f_s2]/(fs/2);

% ── IIR filters — SOS form avoids numerical instability at high orders ────
[N_bw, Wn_bw]   = buttord(Wp, Ws, Rp, Rs);
N_bw = min(N_bw, 6);  % cap order; full spec needs N_bw from buttord
[sos_bw, g_bw]  = butter(N_bw, Wn_bw, 'bandpass');

[N_c1, Wn_c1]   = cheb1ord(Wp, Ws, Rp, Rs);
[sos_c1, g_c1]  = cheby1(N_c1, Rp, Wn_c1, 'bandpass');

[N_c2, Wn_c2]   = cheb2ord(Wp, Ws, Rp, Rs);
[sos_c2, g_c2]  = cheby2(N_c2, Rs, Wn_c2, 'bandpass');

[N_el, Wn_el]   = ellipord(Wp, Ws, Rp, Rs);
[sos_el, g_el]  = ellip(N_el, Rp, Rs, Wn_el, 'bandpass');

fprintf('\nIIR orders — Butterworth:%d  ChebyI:%d  ChebyII:%d  Elliptic:%d\n', N_bw,N_c1,N_c2,N_el);

% ── FIR order estimate (Kaiser formula) — N >= (Rs-7.95)/(2.285*dw_min) ─
delta_f     = min(f_p1-f_s1, f_s2-f_p2);
delta_omega = 2*pi*delta_f/fs;
N_est = ceil((Rs-7.95)/(2.285*delta_omega));
if mod(N_est,2)==0; N_est=N_est+1; end  % odd order -> Type-I linear phase

% ── FIR comparison orders — scaled relative to fs and recording length ────
% N_lo  : ~0.5 s latency  (fast, wide transition)
% N_mid : ~2.0 s latency  (balanced)
% N_hi  : meets Rs spec   (may be very long for short recordings)
N_lo  = 2*round(0.5*fs/2)+1;    % nearest odd number to 0.5 s at this fs
N_mid = 2*round(2.0*fs/2)+1;    % nearest odd number to 2.0 s at this fs
N_hi  = min(N_est, round(sw_len_s*fs*0.4));  % cap at 40% of window to avoid startup dominance
if mod(N_hi,2)==0; N_hi=N_hi+1; end

fprintf('FIR Kaiser estimate: N=%d  (latency = %.1f s)\n', N_est, (N_est/2)/fs);
fprintf('FIR comparison: N_lo=%d (%.1fs)  N_mid=%d (%.1fs)  N_hi=%d (%.1fs)\n', ...
    N_lo, (N_lo/2)/fs, N_mid, (N_mid/2)/fs, N_hi, (N_hi/2)/fs);

% ── FIR B1: Windowed-sinc with Hamming window ─────────────────────────────
b_ham_lo  = fir1(N_lo -1, Wp, 'bandpass', hamming(N_lo));
b_ham_mid = fir1(N_mid-1, Wp, 'bandpass', hamming(N_mid));
b_ham_hi  = fir1(N_hi -1, Wp, 'bandpass', hamming(N_hi));

% ── FIR B2: Kaiser window — kaiserord computes N and beta from specs ──────
dev_stop = 10^(-Rs/20);
dev_pass = (10^(Rp/10)-1)/(10^(Rp/10)+1);
[N_ksr, Wn_ksr, beta_ksr, ftype_ksr] = kaiserord(...
    [f_s1 f_p1 f_p2 f_s2], [0 1 0], [dev_stop dev_pass dev_stop], fs);
if mod(N_ksr,2)==0; N_ksr=N_ksr+1; end
b_ksr = fir1(N_ksr, Wn_ksr, ftype_ksr, kaiser(N_ksr+1, beta_ksr));
fprintf('Kaiser FIR: N=%d  beta=%.3f\n', N_ksr, beta_ksr);

% ── FIR B3: Parks-McClellan equiripple — optimal for fixed order N ────────
N_pm       = N_ksr;
w_ratio    = 10^((Rs-Rp)/20);  % stopband weighted more than passband
bands_pm   = [0 f_s1 f_p1 f_p2 f_s2 fs/2]/(fs/2);
amps_pm    = [0 0    1    1    0    0   ];
weights_pm = [w_ratio 1 w_ratio];
b_pm = firpm(N_pm-1, bands_pm, amps_pm, weights_pm);
fprintf('Parks-McClellan FIR: N=%d  stopband weight=%.1fx\n', N_pm, w_ratio);

% ── Fig 0: Adaptive spec diagnostic — raw spectrum + derived passband ─────
% This figure shows WHAT the code saw in the signal and WHY it chose these
% passband edges.  When running on a new dataset, check this figure first:
% if the artifact peak and cardiac peak are clearly separated, the adaptive
% spec will work well.  If they overlap, the rPPG signal quality is too low.
figure('Name','Fig 0 — Adaptive Spec Diagnostic','Position',[30 650 900 340]);
plot(F_rough*60, 10*log10(P_rough+1e-30), 'Color',[0.4 0.4 0.4],'LineWidth',1.2);
hold on;
% Mark the detected artifact peak
xline(f_artifact*60, 'r--', sprintf('Motion artifact %.0f BPM (%.2f Hz)', ...
    f_artifact*60, f_artifact), 'LineWidth',1.5, 'LabelVerticalAlignment','bottom');
% Mark adaptive passband edges
xline(f_p1*60, 'b-',  sprintf('f_{p1}=%.2fHz', f_p1), 'LineWidth',2.0);
xline(f_p2*60, 'b-',  sprintf('f_{p2}=%.2fHz', f_p2), 'LineWidth',2.0);
xline(f_s1*60, 'b:',  'LineWidth',1.2);
xline(f_s2*60, 'b:',  'LineWidth',1.2);
% Mark GT BPM
xline(bpm_gt_mean, 'k-', sprintf('GT %.1f BPM', bpm_gt_mean), ...
    'LineWidth', 2.5, 'LabelVerticalAlignment','bottom');
xlabel('BPM'); ylabel('PSD (dB/Hz)');
title(sprintf('Raw S_{det} Spectrum — Artifact at %.0f BPM | Adaptive passband [%.0f–%.0f] BPM', ...
    f_artifact*60, f_p1*60, f_p2*60));
xlim([0 min(300, fs/2*60)]); grid on;
legend('S_{det} PSD','Artifact peak','f_{p1}','f_{p2}','Location','northeast');

% ── Frequency responses ───────────────────────────────────────────────────
f_zoom   = linspace(0, min(8, fs/2-0.01), 8192)';
dB_floor = -80;
safeDB   = @(H) max(20*log10(abs(H)), dB_floor);

H_bw = sosfreqz(sos_bw, g_bw, f_zoom, fs);
H_c1 = sosfreqz(sos_c1, g_c1, f_zoom, fs);
H_c2 = sosfreqz(sos_c2, g_c2, f_zoom, fs);
H_el = sosfreqz(sos_el, g_el, f_zoom, fs);
H_ham_lo  = freqz(b_ham_lo,   1,   f_zoom, fs);
H_ham_mid = freqz(b_ham_mid,  1,   f_zoom, fs);
H_ham_hi  = freqz(b_ham_hi,   1,   f_zoom, fs);
H_ksr     = freqz(b_ksr,      1,   f_zoom, fs);
H_pm      = freqz(b_pm,       1,   f_zoom, fs);

% ── Fig 1: IIR magnitude ──────────────────────────────────────────────────
figure('Name','Fig 1 — IIR Magnitude','Position',[30 600 800 360]);
plot(f_zoom, safeDB(H_bw),'b', 'LineWidth',1.8); hold on;
plot(f_zoom, safeDB(H_c1),'r', 'LineWidth',1.8);
plot(f_zoom, safeDB(H_c2),'g', 'LineWidth',1.8);
plot(f_zoom, safeDB(H_el),'m', 'LineWidth',1.8);
xline(f_p1,'k--','LineWidth',1.2); xline(f_p2,'k--','LineWidth',1.2);
xline(f_s1,'k:','LineWidth',1.0);  xline(f_s2,'k:','LineWidth',1.0);
yline(-Rp,'b:',sprintf('-%ddB pass',Rp),'LineWidth',0.9);
yline(-Rs,'r--',sprintf('-%ddB stop',Rs),'LineWidth',0.9);
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend(sprintf('Butterworth N=%d',N_bw),sprintf('Chebyshev I N=%d',N_c1), ...
       sprintf('Chebyshev II N=%d',N_c2),sprintf('Elliptic N=%d',N_el),'Location','southwest');
title(sprintf('IIR Filters — Magnitude  |  Rp=%.1fdB  Rs=%.0fdB',Rp,Rs));
ylim([dB_floor 5]); xlim([0 min(8,fs/2)]); grid on;

% ── Fig 2: FIR magnitude ──────────────────────────────────────────────────
figure('Name','Fig 2 — FIR Magnitude','Position',[30 180 800 360]);
plot(f_zoom, safeDB(H_ham_lo), 'b--','LineWidth',1.3); hold on;
plot(f_zoom, safeDB(H_ham_mid),'b-.','LineWidth',1.6);
plot(f_zoom, safeDB(H_ham_hi), 'b',  'LineWidth',2.2);
plot(f_zoom, safeDB(H_ksr),    'r',  'LineWidth',1.8);
plot(f_zoom, safeDB(H_pm),     'm',  'LineWidth',1.8);
xline(f_p1,'k--','LineWidth',1.2); xline(f_p2,'k--','LineWidth',1.2);
xline(f_s1,'k:','LineWidth',1.0);  xline(f_s2,'k:','LineWidth',1.0);
yline(-Rp,'b:',sprintf('-%ddB pass',Rp),'LineWidth',0.9);
yline(-Rs,'r--',sprintf('-%ddB stop',Rs),'LineWidth',0.9);
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend(sprintf('Hamming N=%d',N_lo),sprintf('Hamming N=%d',N_mid), ...
       sprintf('Hamming N=%d (spec)',N_hi), ...
       sprintf('Kaiser N=%d b=%.1f',N_ksr,beta_ksr), ...
       sprintf('Parks-McClellan N=%d',N_pm),'Location','southwest');
title('FIR Filters — Magnitude  (higher order = sharper transition)');
ylim([dB_floor 5]); xlim([0 min(8,fs/2)]); grid on;

% ── Fig 3: Phase — linear (FIR) vs nonlinear (IIR) ───────────────────────
figure('Name','Fig 3 — Phase','Position',[870 600 800 420]);
subplot(2,1,1);
plot(f_zoom, unwrap(angle(H_bw))*180/pi,'b','LineWidth',1.5); hold on;
plot(f_zoom, unwrap(angle(H_c1))*180/pi,'r','LineWidth',1.5);
plot(f_zoom, unwrap(angle(H_c2))*180/pi,'g','LineWidth',1.5);
plot(f_zoom, unwrap(angle(H_el))*180/pi,'m','LineWidth',1.5);
xline(f_p1,'k--'); xline(f_p2,'k--');
ylabel('Phase (deg)'); title('IIR — nonlinear phase');
legend('Butterworth','Cheby I','Cheby II','Elliptic','Location','best');
xlim([0 min(8,fs/2)]); grid on;

subplot(2,1,2);
plot(f_zoom, unwrap(angle(H_pm)) *180/pi,'m','LineWidth',1.8); hold on;
plot(f_zoom, unwrap(angle(H_ksr))*180/pi,'r','LineWidth',1.5);
plot(f_zoom, unwrap(angle(H_ham_hi))*180/pi,'b','LineWidth',1.5);
xline(f_p1,'k--'); xline(f_p2,'k--');
ylabel('Phase (deg)'); xlabel('Frequency (Hz)');
title('FIR — exactly linear phase (straight line)');
legend('Parks-McClellan','Kaiser','Hamming','Location','best');
xlim([0 min(8,fs/2)]); grid on;

% ── Fig 4: Group delay ────────────────────────────────────────────────────
figure('Name','Fig 4 — Group Delay','Position',[870 150 800 420]);
subplot(2,1,1);
n_gd = 1024;
[gd_bw, f_gd] = grpdelay(sos_bw, g_bw, n_gd, fs);
[gd_c1,    ~] = grpdelay(sos_c1, g_c1, n_gd, fs);
[gd_c2,    ~] = grpdelay(sos_c2, g_c2, n_gd, fs);
[gd_el,    ~] = grpdelay(sos_el, g_el, n_gd, fs);
plot(f_gd,gd_bw/fs*1000,'b','LineWidth',1.5); hold on;
plot(f_gd,gd_c1/fs*1000,'r','LineWidth',1.5);
plot(f_gd,gd_c2/fs*1000,'g','LineWidth',1.5);
plot(f_gd,gd_el/fs*1000,'m','LineWidth',1.5);
xline(f_p1,'k--'); xline(f_p2,'k--');
ylabel('Group Delay (ms)'); title('IIR — varies with frequency');
legend('Butterworth','Cheby I','Cheby II','Elliptic');
xlim([0 min(8,fs/2)]); ylim([-200 2000]); grid on;

subplot(2,1,2);
gd_pm  = grpdelay(b_pm,    1,f_zoom,fs);
gd_ksr = grpdelay(b_ksr,   1,f_zoom,fs);
gd_ham = grpdelay(b_ham_hi,1,f_zoom,fs);
plot(f_zoom,gd_pm /fs*1000,'m','LineWidth',1.8); hold on;
plot(f_zoom,gd_ksr/fs*1000,'r','LineWidth',1.5);
plot(f_zoom,gd_ham/fs*1000,'b','LineWidth',1.5);
yline(N_pm/(2*fs)*1000,'k--', ...
    sprintf('%.0f ms = N/2 = %d samples',N_pm/(2*fs)*1000,floor(N_pm/2)), ...
    'LineWidth',1.2,'LabelVerticalAlignment','bottom');
xline(f_p1,'k--'); xline(f_p2,'k--');
ylabel('Group Delay (ms)'); xlabel('Frequency (Hz)');
title('FIR — constant (N/2 samples, linear phase confirmed)');
legend('Parks-McClellan','Kaiser','Hamming');
xlim([0 min(8,fs/2)]); grid on;

% ── Apply filters ─────────────────────────────────────────────────────────
S_bw = filtfilt(sos_bw, g_bw, S_det);
S_c1 = filtfilt(sos_c1, g_c1, S_det);
S_c2 = filtfilt(sos_c2, g_c2, S_det);
S_el = filtfilt(sos_el, g_el, S_det);
S_ham_lo  = filtfilt(b_ham_lo, 1,   S_det);
S_ham_mid = filtfilt(b_ham_mid,1,   S_det);
S_ham_hi  = filtfilt(b_ham_hi, 1,   S_det);
S_ksr     = filtfilt(b_ksr,    1,   S_det);
S_pm      = filtfilt(b_pm,     1,   S_det);

% ── Fig 5: Time-domain filtered BVP ──────────────────────────────────────
figure('Name','Fig 5 — Filtered BVP Signals','Position',[30 50 1200 500]);
subplot(1,2,1);
plot(t_axis,S_det,'Color',[0.8 0.8 0.8],'LineWidth',0.8); hold on;
plot(t_axis,S_bw,'b','LineWidth',1.2);
plot(t_axis,S_c1,'r','LineWidth',1.2);
plot(t_axis,S_c2,'g','LineWidth',1.2);
plot(t_axis,S_el,'m','LineWidth',1.2);
xlabel('Time (s)'); ylabel('BVP Amplitude');
legend('S_{det}','Butterworth','Cheby I','Cheby II','Elliptic','Location','best');
title('IIR Filtered BVP'); grid on;

subplot(1,2,2);
plot(t_axis,S_det,    'Color',[0.8 0.8 0.8],'LineWidth',0.8); hold on;
plot(t_axis,S_ham_lo, 'b--','LineWidth',1.2);
plot(t_axis,S_ham_hi, 'b',  'LineWidth',1.5);
plot(t_axis,S_ksr,    'r',  'LineWidth',1.5);
plot(t_axis,S_pm,     'm',  'LineWidth',1.5);
xlabel('Time (s)'); ylabel('BVP Amplitude');
legend('S_{det}',sprintf('Hamming N=%d',N_lo),sprintf('Hamming N=%d',N_hi), ...
       'Kaiser','Parks-McClellan','Location','best');
title('FIR Filtered BVP'); grid on;

% ── Fig 6: PSD in cardiac band vs Ground Truth ────────────────────────────
nperseg  = min(round(fs*10), T);
noverlap = floor(nperseg/2);
[p_bw, f_p] = pwelch(S_bw,     hann(nperseg), noverlap, [], fs);
[p_c2,   ~] = pwelch(S_c2,     hann(nperseg), noverlap, [], fs);
[p_el,   ~] = pwelch(S_el,     hann(nperseg), noverlap, [], fs);
[p_ham,  ~] = pwelch(S_ham_hi, hann(nperseg), noverlap, [], fs);
[p_ksr,  ~] = pwelch(S_ksr,    hann(nperseg), noverlap, [], fs);
[p_pm,   ~] = pwelch(S_pm,     hann(nperseg), noverlap, [], fs);
band = (f_p >= f_p1) & (f_p <= f_p2);

figure('Name','Fig 6 — PSD vs Ground Truth','Position',[870 50 900 380]);
plot(f_p(band)*60, 10*log10(p_bw(band)), 'b',              'LineWidth',1.5); hold on;
plot(f_p(band)*60, 10*log10(p_c2(band)), 'g',              'LineWidth',1.5);
plot(f_p(band)*60, 10*log10(p_el(band)), 'm',              'LineWidth',1.5);
plot(f_p(band)*60, 10*log10(p_pm(band)), 'Color',[0.5 0 0.5],'LineWidth',2.0,'LineStyle','--');
plot(f_p(band)*60, 10*log10(p_ksr(band)),'Color',[0.85 0.33 0],'LineWidth',1.8,'LineStyle',':');
plot(f_p(band)*60, 10*log10(p_ham(band)),'c',              'LineWidth',1.5,'LineStyle','-.');
xline(bpm_gt_mean,  'k-', sprintf('GT mean %.1f BPM',  bpm_gt_mean), 'LineWidth',2.5,'LabelVerticalAlignment','bottom');
xline(bpm_gt_median,'k--',sprintf('GT med %.1f BPM',bpm_gt_median),  'LineWidth',1.5,'LabelVerticalAlignment','top');
xlabel('BPM'); ylabel('PSD (dB/Hz)');
legend('Butterworth','Cheby II','Elliptic','Parks-McClellan','Kaiser', ...
       sprintf('Hamming N=%d',N_hi),'Location','best');
title('PSD in Cardiac Band — peak should align with GT'); grid on;

% ── Quality weights from Python CSV (ones if unavailable / MATLAB path) ────
if USE_PYTHON_CSV && ismember('quality_score', py_data.Properties.VariableNames)
    quality_t = double(py_data.quality_score);
else
    quality_t = ones(T, 1);
end

% ── Sliding-window BPM evaluation ─────────────────────────────────────────
sw_len    = round(sw_len_s * fs);   % adaptive: 10 s normal, 5 s for short recordings
sw_step   = round(1*fs);
sw_starts = 1:sw_step:T-sw_len+1;
n_sw      = numel(sw_starts);
sw_time   = zeros(1, n_sw);
sw_quality= zeros(1, n_sw);   % mean quality weight per window

all_sigs   = {S_bw,S_c1,S_c2,S_el, S_ham_lo,S_ham_mid,S_ham_hi,S_ksr,S_pm};
all_labels = {'Butterworth','Cheby I','Cheby II','Elliptic', ...
              sprintf('Hamming N=%d',N_lo),sprintf('Hamming N=%d',N_mid), ...
              sprintf('Hamming N=%d (spec)',N_hi), ...
              sprintf('Kaiser N=%d',N_ksr),sprintf('Parks-McClellan N=%d',N_pm)};
nF      = numel(all_sigs);
sw_bpm  = zeros(nF, n_sw);
sw_snr  = zeros(nF, n_sw);   % per-window cardiac-band SNR (dB) per filter
nfft_sw = 4096;

for k = 1:n_sw
    idx_s        = sw_starts(k);
    idx_e        = idx_s + sw_len - 1;
    sw_time(k)   = (idx_s-1)/fs + sw_len/(2*fs);
    sw_quality(k)= mean(quality_t(idx_s:idx_e));
    for fi = 1:nF
        seg = all_sigs{fi}(idx_s:idx_e);
        np  = length(seg);
        [pw_k, fw_k] = pwelch(seg, hann(np), floor(np/2), nfft_sw, fs);
        bk        = (fw_k >= f_p1) & (fw_k <= f_p2);
        pw_band   = pw_k(bk);
        fw_band   = fw_k(bk);
        [pk_val, pi_] = max(pw_band);
        bpm_est   = fw_band(pi_) * 60;

        % ── Harmonic correction ───────────────────────────────────────────
        % If dominant peak < 60 BPM, the cardiac signal may be at 2× that
        % frequency (typical when strong near-DC motion artifact exists).
        % Threshold 30%: accept the doubling when 2× bin has at least 30%
        % of the half-freq peak power.  50% was too strict when the artifact
        % is much stronger than the cardiac signal.
        if bpm_est < 60
            f_double = (bpm_est * 2) / 60;   % Hz
            if f_double <= f_p2
                [~, pi2] = min(abs(fw_k - f_double));
                if pw_k(pi2) >= 0.30 * pk_val
                    bpm_est = bpm_est * 2;
                end
            end
        end
        sw_bpm(fi, k) = bpm_est;

        % ── SNR anchored at the harmonic-corrected BPM ───────────────────
        % Evaluate at the corrected cardiac frequency, not at the dominant
        % peak.  Prevents a wrong sharp peak (e.g. Parks-McClellan at 207
        % BPM) from earning a high SNR score.
        [~, pi_c] = min(abs(fw_k - bpm_est/60));
        sw_snr(fi, k) = 10 * log10(pw_k(pi_c) / (mean(pw_band) + 1e-30));
    end
end

gt_interp  = interp1(gt_time, double(gt_hr), sw_time, 'linear', 'extrap');
sw_err     = sw_bpm - gt_interp;

% ── Quality-weighted and normalised metrics ────────────────────────────────
% Normalise weights so they sum to n_sw (preserves absolute MAE scale).
w       = (sw_quality(:)' / (sum(sw_quality) + 1e-9)) * n_sw;
mae_w_all   = (abs(sw_err) * w') / sum(w);   % quality-weighted MAE
mae_all     = mean(abs(sw_err), 2);           % unweighted (reference only)
rmse_all    = sqrt(mean(sw_err.^2, 2));
pct_err     = mae_w_all / bpm_gt_mean * 100;  % %-error: dataset-independent
consistency = std(sw_bpm, 0, 2);              % BPM std across windows
snr_all     = mean(sw_snr, 2);                % mean cardiac-band SNR (dB)
% ── Lin's Concordance Correlation Coefficient (CCC) ──────────────────────
% Pearson r fails when GT barely varies (std≈2 BPM here): a perfect
% estimator shows low r because it correlates noise against a flat line.
% CCC accounts for both covariance (precision) AND mean bias (accuracy).
% CCC≈1 when estimates are close to GT mean even if GT is nearly constant.
ccc_all = arrayfun(@(f) concordance_cc(sw_bpm(f,:)', gt_interp'), 1:nF)';

% ── Composite score (higher = better) ─────────────────────────────────────
% Normalise each component to [0,1] then combine.
%   40% quality-weighted MAE  — accuracy
%   25% BPM consistency       — stability across windows
%   20% cardiac-band SNR      — peak sharpness at corrected BPM
%   15% CCC                   — agreement (handles stable GT correctly)
norm01  = @(x) (x - min(x)) ./ (max(x) - min(x) + 1e-9);
mae_s   = 1 - norm01(mae_w_all);   % lower  MAE  → higher score
cons_s  = 1 - norm01(consistency); % lower  std  → higher score
snr_s   =     norm01(snr_all);     % higher SNR  → higher score
ccc_s   =     norm01(ccc_all);     % higher CCC  → higher score
composite   = 0.40*mae_s + 0.25*cons_s + 0.20*snr_s + 0.15*ccc_s;

[~, best_idx]     = max(composite);
[~, best_iir]     = max(composite(1:4));
[~, best_fir_rel] = max(composite(5:nF));
best_fir          = best_fir_rel + 4;

% ── Fig 7: Sliding BPM vs GT — IIR ───────────────────────────────────────
figure('Name','Fig 7 — Sliding BPM vs GT (IIR)','Position',[30 50 1000 400]);
stairs(gt_time,gt_hr,'k','LineWidth',2.5); hold on;
iir_c = {'b','r','g','m'};
for fi = 1:4
    est = sw_bpm(fi,:); sigma = movstd(est,11);
    fill([sw_time,fliplr(sw_time)],[est+sigma,fliplr(est-sigma)],iir_c{fi},'FaceAlpha',0.12,'EdgeColor','none');
    plot(sw_time,est,iir_c{fi},'LineWidth',1.5);
end
yline(bpm_gt_mean,'k--',sprintf('GT mean %.1f BPM',bpm_gt_mean),'LineWidth',1.5);
xlabel('Time (s)'); ylabel('BPM');
legend('Ground Truth','','Butterworth','','Cheby I','','Cheby II','','Elliptic','Location','best');
title('IIR: Sliding-Window (10 s) BPM +-1sigma vs GT');
ylim([30 130]); xlim([0 max(gt_time)+2]); grid on;

% ── Fig 8: Sliding BPM vs GT — FIR ───────────────────────────────────────
figure('Name','Fig 8 — Sliding BPM vs GT (FIR)','Position',[30 50 1000 400]);
stairs(gt_time,gt_hr,'k','LineWidth',2.5); hold on;
fir_c  = {'b','b','b','r','m'};
fir_ls = {'--','-.', '-','-','-'};
for ii = 1:5
    fi = ii+4;
    plot(sw_time,sw_bpm(fi,:),fir_c{ii},'LineStyle',fir_ls{ii},'LineWidth',1.5);
end
yline(bpm_gt_mean,'k--',sprintf('GT mean %.1f BPM',bpm_gt_mean),'LineWidth',1.5);
xlabel('Time (s)'); ylabel('BPM');
legend(['Ground Truth',all_labels(5:9),{'GT mean'}],'Location','best');
title('FIR: Sliding-Window (10 s) BPM vs GT');
ylim([30 130]); xlim([0 max(gt_time)+2]); grid on;

% ── Fig 9: Per-window error — all filters stacked ────────────────────────
figure('Name','Fig 9 — Per-window Error','Position',[30 50 1200 1050]);
all_c = {'b','r','g','m','c',[0 0.5 0],[0 0.4 0.8],[0.85 0.33 0],[0.5 0 0.5]};

for fi = 1:nF
    err_i = sw_err(fi,:);
    subplot(nF,1,fi);
    fill([sw_time,fliplr(sw_time)], ...
         [err_i+movstd(err_i,11),fliplr(err_i-movstd(err_i,11))], ...
         all_c{fi},'FaceAlpha',0.18,'EdgeColor','none'); hold on;
    plot(sw_time, err_i, 'Color', all_c{fi}, 'LineWidth', 1.5);
    yline(0,'k--','LineWidth',1.1);
    yline( 5,'k:','LineWidth',0.9); yline(-5,'k:','LineWidth',0.9);
    ylabel('Err (BPM)'); ylim([-60 60]); xlim([0 max(gt_time)+2]); grid on;
    best_marker = '';
    if fi == best_idx; best_marker = '  ◄ BEST'; end
    title(sprintf('%s  |  wMAE=%.1f (%.0f%%)  SNR=%.1fdB  std=%.1f  CCC=%.3f  Score=%.2f%s', ...
        all_labels{fi}, mae_w_all(fi), pct_err(fi), snr_all(fi), ...
        consistency(fi), ccc_all(fi), composite(fi), best_marker), 'FontSize', 7.5);
end
xlabel('Time (s)');
sgtitle('Per-window Error (harmonic-corrected)  |  dotted=±5 BPM  |  shaded=±1σ', ...
    'FontWeight','bold');

% ── Summary table ─────────────────────────────────────────────────────────
W = 116;
fprintf('\n%s\n', repmat('=',1,W));
fprintf('%-32s  %4s  %8s  %6s  %6s  %6s  %5s  %5s  %7s  %7s\n', ...
    'Filter','N','wMAE','%err','Consis','SNRdB','CCC','Score','BPM_med','GT');
fprintf('%s\n', repmat('-',1,W));

for fi = 1:nF
    bpm_med = median(sw_bpm(fi,:));  % median across windows (robust peak estimate)
    if fi == 5; fprintf('%s\n', repmat('-',1,W)); end
    tag = '';  if fi == best_idx; tag = '  ◄'; end
    fprintf('%-32s  %4d  %8.2f  %5.1f%%  %6.1f  %6.1f  %5.3f  %5.3f  %7.1f  %7.1f%s\n', ...
        all_labels{fi}, numel(all_sigs{fi})-1, ...
        mae_w_all(fi), pct_err(fi), consistency(fi), snr_all(fi), ...
        ccc_all(fi), composite(fi), bpm_med, bpm_gt_mean, tag);
end
fprintf('%s\n', repmat('=',1,W));
fprintf('GT: mean=%.1f  median=%.1f  std=%.1f  range=[%d %d]  n=%d\n', ...
    bpm_gt_mean, bpm_gt_median, std(double(gt_hr)), min(gt_hr), max(gt_hr), numel(gt_hr));
fprintf('\nComposite = 0.40×wMAE_score + 0.25×Consistency_score + 0.20×SNR_score + 0.15×r_score\n');
fprintf('Best overall : %s  (score=%.3f  wMAE=%.1f BPM  %.0f%%)\n', ...
    all_labels{best_idx}, composite(best_idx), mae_w_all(best_idx), pct_err(best_idx));
fprintf('Best IIR     : %s  (score=%.3f  wMAE=%.1f BPM)\n', ...
    all_labels{best_iir}, composite(best_iir), mae_w_all(best_iir));
fprintf('Best FIR     : %s  (score=%.3f  wMAE=%.1f BPM)\n', ...
    all_labels{best_fir}, composite(best_fir), mae_w_all(best_fir));

% ── Fig 10: Composite score bar chart — generalisable filter ranking ───────
figure('Name','Fig 10 — Composite Score','Position',[30 50 960 400]);
bar_clr = repmat([0.35 0.60 0.85], nF, 1);
bar_clr(best_iir, :)   = [0.85 0.50 0.20];  % best IIR: orange
bar_clr(best_fir, :)   = [0.20 0.72 0.35];  % best FIR: green
if best_idx == best_iir || best_idx == best_fir
    bar_clr(best_idx,:) = [0.10 0.45 0.10];  % overall best: dark green
end
b = bar(composite, 'FaceColor', 'flat');
b.CData = bar_clr;
set(gca, 'XTick', 1:nF, 'XTickLabel', all_labels, 'FontSize', 8);
xtickangle(20);
ylabel('Composite Score  (higher = better)');
title(sprintf('Filter Ranking — Composite Score  |  Best: %s  (%.3f)', ...
    all_labels{best_idx}, composite(best_idx)), 'FontSize', 10);
ylim([0 1.12]); grid on;
yline(composite(best_idx), 'k--', 'LineWidth', 1.3);
xline(4.5, 'k:', 'LineWidth', 1.2);
text(2.5,  1.07, 'IIR',  'FontSize', 9, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(7.0,  1.07, 'FIR',  'FontSize', 9, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% ── Tight lower passband comparison (1.0 Hz / 60 BPM) ────────────────────
% The standard 0.7–3.5 Hz passband lets through 0.7–1.0 Hz motion and
% respiration artifacts.  For adult subjects at rest no true cardiac signal
% exists below 1.0 Hz (60 BPM).  Test tighter lower edges on the two best
% candidates (Elliptic IIR and Hamming N=151 FIR) to see the improvement.
fprintf('\n%s\n', repmat('-',1,W));
fprintf('Tight lower-passband comparison  (fp1=1.0 Hz / 60 BPM  |  fs1=0.65 Hz)\n');
fprintf('%s\n', repmat('-',1,W));

f_p1_t = 1.0;   f_s1_t = 0.65;
Wp_t = [f_p1_t f_p2]/(fs/2);
Ws_t = [f_s1_t f_s2]/(fs/2);

[N_el_t, Wn_el_t]   = ellipord(Wp_t, Ws_t, Rp, Rs);
[sos_el_t, g_el_t]  = ellip(N_el_t, Rp, Rs, Wn_el_t, 'bandpass');
b_ham_t = fir1(N_mid-1, Wp_t, 'bandpass', hamming(N_mid));

S_el_t   = filtfilt(sos_el_t, g_el_t, S_det);
S_ham_t  = filtfilt(b_ham_t,  1,      S_det);

tight_sigs   = {S_el_t, S_ham_t};
tight_labels = {sprintf('Elliptic(tight) N=%d', N_el_t), ...
                sprintf('Hamming(tight) N=%d',  N_mid)};
n_tight = numel(tight_sigs);

sw_bpm_t  = zeros(n_tight, n_sw);
sw_snr_t  = zeros(n_tight, n_sw);

for k = 1:n_sw
    idx_s = sw_starts(k);
    idx_e = idx_s + sw_len - 1;
    for fi = 1:n_tight
        seg = tight_sigs{fi}(idx_s:idx_e);
        np  = length(seg);
        [pw_k, fw_k] = pwelch(seg, hann(np), floor(np/2), nfft_sw, fs);
        bk = (fw_k >= f_p1_t) & (fw_k <= f_p2);
        pw_band = pw_k(bk);  fw_band = fw_k(bk);
        [pk_val, pi_] = max(pw_band);
        bpm_est = fw_band(pi_) * 60;
        if bpm_est < 60
            f_d = bpm_est * 2 / 60;
            if f_d <= f_p2
                [~, pi2] = min(abs(fw_k - f_d));
                if pw_k(pi2) >= 0.30 * pk_val; bpm_est = bpm_est * 2; end
            end
        end
        sw_bpm_t(fi, k) = bpm_est;
        [~, pi_c] = min(abs(fw_k - bpm_est/60));
        sw_snr_t(fi, k) = 10*log10(pw_k(pi_c) / (mean(pw_band)+1e-30));
    end
end

sw_err_t = sw_bpm_t - gt_interp;
for fi = 1:n_tight
    mae_t    = (abs(sw_err_t(fi,:)) * w') / sum(w);
    pct_t    = mae_t / bpm_gt_mean * 100;
    cons_t   = std(sw_bpm_t(fi,:));
    snr_t    = mean(sw_snr_t(fi,:));
    ccc_t    = concordance_cc(sw_bpm_t(fi,:)', gt_interp');
    bpm_med_t= median(sw_bpm_t(fi,:));
    fprintf('%-32s  %4d  %8.2f  %5.1f%%  %6.1f  %6.1f  %5.3f           %7.1f  %7.1f\n', ...
        tight_labels{fi}, numel(tight_sigs{fi})-1, ...
        mae_t, pct_t, cons_t, snr_t, ccc_t, bpm_med_t, bpm_gt_mean);
end
fprintf('%s\n', repmat('=',1,W));
fprintf('fp1=0.7Hz passes 0.7-1.0 Hz motion/respiration artifacts.\n');
fprintf('fp1=1.0Hz removes that band — compare wMAE and CCC above vs standard.\n');

% ── Export MATLAB-filtered signals to data/ for FDA_MUSIC ────────────────
if USE_PYTHON_CSV
    filter_dir = fullfile(pipeline_dir, 'results', 'filter_results');
    if ~exist(filter_dir, 'dir');  mkdir(filter_dir);  end

    out_tbl               = py_data;
    out_tbl.BVP_ham_tight = S_ham_t;    % PRIMARY — best performer (fp1=1.0 Hz)
    out_tbl.BVP_ham_adapt = S_ham_mid;  % Hamming N_mid, adaptive fp1
    out_tbl.BVP_el_tight  = S_el_t;    % Elliptic tight, for comparison
    out_tbl.BVP_el_adapt  = S_el;      % Elliptic adaptive fp1, for comparison
    out_tbl.f_p1_adapt    = repmat(f_p1,  T, 1);   % adaptive lower cutoff used
    out_tbl.f_p2_adapt    = repmat(f_p2,  T, 1);   % upper cutoff used
    out_tbl.N_ham_mid     = repmat(N_mid, T, 1);   % FIR order

    out_csv = fullfile(filter_dir, sprintf('filterdesign_%s.csv', datestr(now,'yyyymmdd_HHMMSS')));
    writetable(out_tbl, out_csv);
    fprintf('\nExported → %s\n', out_csv);
    fprintf('  PRIMARY: BVP_ham_tight (Hamming N=%d, fp1=1.0Hz)\n', N_mid);
    fprintf('  Also: BVP_ham_adapt (fp1=%.2fHz)  BVP_el_tight\n', f_p1);
end

function H = sosfreqz(sos, g, f, fs)
% Cascade freqz section-by-section; freqz always returns a column vector.
    H = ones(numel(f), 1);
    for k = 1:size(sos, 1)
        H = H .* freqz(sos(k,1:3), sos(k,4:6), f(:), fs);
    end
    H = g .* H;
end

function ccc = concordance_cc(x, y)
% Lin's Concordance Correlation Coefficient.
% Measures agreement between two vectors, accounting for both covariance
% (precision) and mean bias (accuracy).  Unlike Pearson r, CCC is high
% only when estimates are close to GT mean AND track GT variation.
% Works correctly when GT is nearly constant (Pearson r degrades there).
    x = x(:);  y = y(:);
    mu_x = mean(x);  mu_y = mean(y);
    s2_x = mean((x - mu_x).^2);   % population variance
    s2_y = mean((y - mu_y).^2);
    s_xy = mean((x - mu_x) .* (y - mu_y));
    ccc  = 2 * s_xy / (s2_x + s2_y + (mu_x - mu_y)^2 + 1e-9);
end

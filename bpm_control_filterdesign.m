clear; close all; clc;

VID_PATH = '/home/macs/Documents/rPPG-Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/recording_2026-04-07T22-04-35Z.mov';
CSV_PATH = '/home/macs/Documents/rPPG-Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/vitals.csv';

csv_data      = readtable(CSV_PATH);
gt_time       = csv_data.offset_seconds;
gt_hr         = csv_data.heart_rate;
valid         = ~isnan(gt_hr);
gt_time       = gt_time(valid);  gt_hr = gt_hr(valid);
bpm_gt_mean   = mean(gt_hr);
bpm_gt_median = median(gt_hr);
fprintf('GT: mean=%.1f  median=%.1f  range=[%d %d]  n=%d\n', ...
    bpm_gt_mean, bpm_gt_median, min(gt_hr), max(gt_hr), numel(gt_hr));

vid_tmp     = VideoReader(VID_PATH);
first_frame = readFrame(vid_tmp);
[H_vid, W_vid, ~] = size(first_frame);  clear vid_tmp;

try
    detector  = vision.CascadeObjectDetector();
    all_boxes = step(detector, first_frame);
catch;  all_boxes = []; end

if isempty(all_boxes)
    x1=floor(W_vid*0.25); y1=floor(H_vid*0.05);
    x2=floor(W_vid*0.75); y2=floor(H_vid*0.75);
else
    [~,idx] = max(all_boxes(:,3).*all_boxes(:,4));
    bbox = all_boxes(idx,:);
    x1=max(bbox(1),1);               y1=max(bbox(2),1);
    x2=min(bbox(1)+bbox(3)-1,W_vid); y2=min(bbox(2)+bbox(4)-1,H_vid);
end

vid = VideoReader(VID_PATH);
fs  = vid.FrameRate;
R_t=[]; G_t=[]; B_t=[];

while hasFrame(vid)
    frame = readFrame(vid);
    fc    = frame(y1:y2, x1:x2, :);
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
t_axis = (0:T-1)/fs;

R_n = R_t/mean(R_t);  G_n = G_t/mean(G_t);  B_n = B_t/mean(B_t);
Xs  = 3*R_n - 2*G_n;
Ys  = 1.5*R_n + G_n - 1.5*B_n;
alpha = std(Xs)/std(Ys);
S     = Xs - alpha*Ys;

t_vec  = (1:T)';
coeffs = [t_vec, ones(T,1)] \ S(:);
S_det  = S(:) - (coeffs(1)*t_vec + coeffs(2));
fprintf('Signal: T=%d frames  fs=%.4f Hz\n', T, fs);

% ── Filter specifications — adjust these to explore different designs ─────
f_p1 = 0.7;   % Hz  lower passband edge (42 BPM)
f_p2 = 3.5;   % Hz  upper passband edge (210 BPM)
f_s1 = 0.4;   % Hz  lower stopband edge
f_s2 = 4.5;   % Hz  upper stopband edge
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

N_lo  = 51;    % low order — wide transition, useful for comparison
N_mid = 151;
N_hi  = N_est; % meets Rs spec

fprintf('FIR Kaiser estimate: N=%d  (latency = %.1f s)\n', N_est, (N_est/2)/fs);

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
[gd_bw, f_gd] = grpdelay(sos_bw, n_gd, fs);
[gd_c1,    ~] = grpdelay(sos_c1, n_gd, fs);
[gd_c2,    ~] = grpdelay(sos_c2, n_gd, fs);
[gd_el,    ~] = grpdelay(sos_el, n_gd, fs);
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

% ── Sliding-window BPM evaluation (10 s window, 1 s step) ─────────────────
sw_len    = round(10*fs);
sw_step   = round(1*fs);
sw_starts = 1:sw_step:T-sw_len+1;
n_sw      = numel(sw_starts);
sw_time   = zeros(1,n_sw);

all_sigs   = {S_bw,S_c1,S_c2,S_el, S_ham_lo,S_ham_mid,S_ham_hi,S_ksr,S_pm};
all_labels = {'Butterworth','Cheby I','Cheby II','Elliptic', ...
              sprintf('Hamming N=%d',N_lo),sprintf('Hamming N=%d',N_mid), ...
              sprintf('Hamming N=%d (spec)',N_hi), ...
              sprintf('Kaiser N=%d',N_ksr),sprintf('Parks-McClellan N=%d',N_pm)};
nF      = numel(all_sigs);
sw_bpm  = zeros(nF, n_sw);
nfft_sw = 4096;

for k = 1:n_sw
    idx_s      = sw_starts(k);
    idx_e      = idx_s + sw_len - 1;
    sw_time(k) = (idx_s-1)/fs + sw_len/(2*fs);
    for fi = 1:nF
        seg = all_sigs{fi}(idx_s:idx_e);
        np  = length(seg);
        [pw_k,fw_k] = pwelch(seg, hann(np), floor(np/2), nfft_sw, fs);
        bk = (fw_k>=f_p1)&(fw_k<=f_p2);
        [~,pi_] = max(pw_k(bk));  fb_ = fw_k(bk);
        sw_bpm(fi,k) = fb_(pi_)*60;
    end
end

gt_interp = interp1(gt_time, double(gt_hr), sw_time, 'linear','extrap');
sw_err    = sw_bpm - gt_interp;

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
figure('Name','Fig 9 — Per-window Error','Position',[30 50 1200 1000]);
all_c    = {'b','r','g','m','c',[0 0.5 0],[0 0.4 0.8],[0.85 0.33 0],[0.5 0 0.5]};
mae_all  = mean(abs(sw_err), 2);
rmse_all = sqrt(mean(sw_err.^2, 2));
rho_all  = arrayfun(@(fi) corr(sw_bpm(fi,:)',gt_interp'), 1:nF)';

for fi = 1:nF
    err_i = sw_err(fi,:);
    subplot(nF,1,fi);
    fill([sw_time,fliplr(sw_time)], ...
         [err_i+movstd(err_i,11),fliplr(err_i-movstd(err_i,11))], ...
         all_c{fi},'FaceAlpha',0.18,'EdgeColor','none'); hold on;
    plot(sw_time,err_i,'Color',all_c{fi},'LineWidth',1.5);
    yline(0,'k--','LineWidth',1.1);
    yline( 5,'k:','LineWidth',0.9); yline(-5,'k:','LineWidth',0.9);
    ylabel('Err (BPM)'); ylim([-60 60]); xlim([0 max(gt_time)+2]); grid on;
    title(sprintf('%s  |  MAE=%.1f  RMSE=%.1f  r=%.3f', ...
        all_labels{fi},mae_all(fi),rmse_all(fi),rho_all(fi)),'FontSize',8);
end
xlabel('Time (s)');
sgtitle('Per-window Error = rPPG BPM - GT  |  dotted=+-5 BPM  |  shaded=+-1sigma','FontWeight','bold');

% ── Summary table ─────────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('=',1,96));
fprintf('%-32s  %4s  %7s  %7s  %+7s  %7s  %7s  %5s\n','Filter','N','BPM','GT','Error','MAE','RMSE','r');
fprintf('%s\n', repmat('-',1,96));

psd_list = {p_bw,[],p_c2,p_el,p_ham,[],[],p_ksr,p_pm};
for fi = 1:nF
    if ~isempty(psd_list{fi})
        pb_band = psd_list{fi}(band);
    else
        [pfi,~] = pwelch(all_sigs{fi}, hann(nperseg), noverlap, [], fs);
        pb_band = pfi(band);
    end
    [~,pi_] = max(pb_band); fp_=f_p(band); fp=fp_(pi_);
    if fi==5; fprintf('%s\n',repmat('-',1,96)); end
    fprintf('%-32s  %4d  %7.1f  %7.1f  %+7.1f  %7.1f  %7.1f  %5.3f\n', ...
        all_labels{fi}, numel(all_sigs{fi})-1, fp*60, bpm_gt_mean, ...
        fp*60-bpm_gt_mean, mae_all(fi), rmse_all(fi), rho_all(fi));
end
fprintf('%s\n', repmat('=',1,96));
fprintf('GT: mean=%.1f  median=%.1f  std=%.1f  range=[%d %d]  n=%d\n', ...
    bpm_gt_mean, bpm_gt_median, std(double(gt_hr)), min(gt_hr), max(gt_hr), numel(gt_hr));

[~,bi]=min(mae_all(1:4));
[~,bf]=min(mae_all(5:9)); bf=bf+4;
fprintf('\nBest IIR: %s  MAE=%.1f BPM  N=%d\n', all_labels{bi},mae_all(bi),numel(all_sigs{bi})-1);
fprintf('Best FIR: %s  MAE=%.1f BPM  N=%d\n',  all_labels{bf},mae_all(bf),numel(all_sigs{bf})-1);

function H = sosfreqz(sos, g, f, fs)
% Cascade freqz section-by-section; freqz always returns a column vector.
    H = ones(numel(f), 1);
    for k = 1:size(sos, 1)
        H = H .* freqz(sos(k,1:3), sos(k,4:6), f(:), fs);
    end
    H = g .* H;
end

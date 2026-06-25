%% bpm_facesegment.m
% rPPG pipeline with HoG + LBP adaptive face segmentation.
% Replaces simple YCbCr bounding-box approach with multi-color-space voting
% followed by HoG gradient energy + LBP variance texture rejection to remove
% hair/eyebrow pixels. Video preview shows light-green skin pixel overlay.
clear; clc; close all;

%% ── Paths ─────────────────────────────────────────────────────────────────
BASE     = '/home/macs/Documents/rPPG-Controls/Measurement Data/DocBOT_2026-04-07_15-04-35';
VID_PATH = fullfile(BASE, 'recording_2026-04-07T22-04-35Z.mov');
CSV_PATH = fullfile(BASE, 'vitals.csv');
OUT_PATH = fullfile('/home/macs/Documents/rPPG-Controls', 'bpm_facesegment_output.csv');

%% ── Ground truth ──────────────────────────────────────────────────────────
csv_data = readtable(CSV_PATH);
gt_time  = csv_data.offset_seconds;
gt_hr    = csv_data.heart_rate;
valid_gt = ~isnan(gt_hr);
gt_time  = gt_time(valid_gt);
gt_hr    = double(gt_hr(valid_gt));
fprintf('Ground truth: %d samples  mean=%.1f BPM  range=[%d %d]\n', ...
    numel(gt_hr), mean(gt_hr), min(gt_hr), max(gt_hr));

%% ── Step 1: Initialize video + first MTCNN detection ──────────────────────
vid_init    = VideoReader(VID_PATH);
fs          = vid_init.FrameRate;
vidDuration = vid_init.Duration;
H_vid       = vid_init.Width;    % swapped after 90° CW rotation
W_vid       = vid_init.Height;
first_frame = rot90(readFrame(vid_init), 3);
clear vid_init;

detector = mtcnn.Detector();
[bboxes, ~, mtcnnLand] = detector.detect(first_frame);
if isempty(bboxes), error('No face detected in frame 1.'); end
[~, best]    = max(bboxes(:,3) .* bboxes(:,4));
faceBox_last = clampBB(round(bboxes(best,:)), H_vid, W_vid);
pts_last     = squeeze(mtcnnLand(best,:,:));
fprintf('First detection: faceBox=[%d %d %d %d]\n', faceBox_last);

%% ── Step 2: Per-frame extraction with skin overlay preview ────────────────
nEst         = ceil(vidDuration * fs) + 20;
R_t          = zeros(1, nEst);
G_t          = zeros(1, nEst);
B_t          = zeros(1, nEst);
lum_t        = zeros(1, nEst);
detected_t   = zeros(1, nEst);
skin_count_t = zeros(1, nEst);
fIdx         = 0;
nFallback    = 0;

vid    = VideoReader(VID_PATH);
player = vision.VideoPlayer('Name', 'HoG+LBP Skin Pixel Overlay');

while hasFrame(vid)
    frame = rot90(readFrame(vid), 3);

    [bboxes_f, ~, mtcnnLand_f] = detector.detect(frame);
    if ~isempty(bboxes_f)
        [~, best_f]  = max(bboxes_f(:,3) .* bboxes_f(:,4));
        faceBox_cur  = clampBB(round(bboxes_f(best_f,:)), H_vid, W_vid);
        pts_cur      = squeeze(mtcnnLand_f(best_f,:,:));
        faceBox_last = faceBox_cur;
        pts_last     = pts_cur;
        det_flag     = 1;
    else
        faceBox_cur = faceBox_last;
        pts_cur     = pts_last;
        det_flag    = 0;
        nFallback   = nFallback + 1;
    end

    faceROI  = imcrop(frame, faceBox_cur);
    skinMask = buildAdvancedSkinMask(faceROI, pts_cur, faceBox_cur);
    n_skin   = nnz(skinMask);

    if n_skin < 50, continue; end

    % Luminance normalization on skin pixels to compensate AGC/auto-exposure
    fcd_raw   = double(faceROI);
    msk       = skinMask(:);
    pix_raw   = reshape(fcd_raw, [], 3);
    frame_lum = mean(pix_raw(msk, :), 'all');
    fcd       = fcd_raw / max(frame_lum, 1) * 128;

    pix  = reshape(fcd, [], 3);
    fIdx = fIdx + 1;
    R_t(fIdx)          = mean(pix(msk, 1));
    G_t(fIdx)          = mean(pix(msk, 2));
    B_t(fIdx)          = mean(pix(msk, 3));
    lum_t(fIdx)        = frame_lum;
    detected_t(fIdx)   = det_flag;
    skin_count_t(fIdx) = n_skin;

    % Light green overlay on detected skin pixels for visual verification
    [fH, fW, ~] = size(faceROI);
    x1 = faceBox_cur(1);  y1 = faceBox_cur(2);
    x2 = min(size(frame,2), x1 + fW - 1);
    y2 = min(size(frame,1), y1 + fH - 1);
    mH = y2 - y1 + 1;  mW = x2 - x1 + 1;
    fullMask = false(size(frame,1), size(frame,2));
    fullMask(y1:y2, x1:x2) = skinMask(1:mH, 1:mW);

    Rd = double(frame(:,:,1));
    Gd = double(frame(:,:,2));
    Bd = double(frame(:,:,3));
    ov = 0.45;                          % overlay alpha
    Rd(fullMask) = Rd(fullMask) * (1-ov);
    Gd(fullMask) = Gd(fullMask) * (1-ov) + 144 * ov;   % light green
    Bd(fullMask) = Bd(fullMask) * (1-ov);
    step(player, uint8(cat(3, Rd, Gd, Bd)));
end

R_t          = R_t(1:fIdx);
G_t          = G_t(1:fIdx);
B_t          = B_t(1:fIdx);
lum_t        = lum_t(1:fIdx);
detected_t   = detected_t(1:fIdx);
skin_count_t = skin_count_t(1:fIdx);
T            = fIdx;
t_axis       = (0:T-1) / fs;
fprintf('Extracted %d frames (%.1fs) at %.2fHz  |  fresh: %d  fallback: %d\n', ...
    T, T/fs, fs, sum(detected_t), nFallback);

%% ── Step 3: CHROM projection + linear detrend ────────────────────────────
R_n      = R_t / mean(R_t);
G_n      = G_t / mean(G_t);
B_n      = B_t / mean(B_t);
Xs       = 3*R_n - 2*G_n;
Ys       = 1.5*R_n + G_n - 1.5*B_n;
if std(Ys) < eps, alpha_c = 1; else, alpha_c = std(Xs)/std(Ys); end
S        = Xs - alpha_c * Ys;

t_vec  = (1:T)';
coeffs = [t_vec, ones(T,1)] \ S(:);
S_det  = S(:) - (coeffs(1)*t_vec + coeffs(2));

%% ── Step 4: Four IIR bandpass filters ─────────────────────────────────────
f_low = 0.7;  f_high = 3.5;
order = 4;    Rp = 0.5;  Rs = 40;
Wn = [f_low f_high] / (fs/2);

[b_bw, a_bw] = butter(order,         Wn, 'bandpass');
[b_c1, a_c1] = cheby1(order, Rp,     Wn, 'bandpass');
[b_c2, a_c2] = cheby2(order, Rs,     Wn, 'bandpass');
[b_el, a_el] = ellip( order, Rp, Rs, Wn, 'bandpass');

S_bw = filtfilt(b_bw, a_bw, S_det);
S_c1 = filtfilt(b_c1, a_c1, S_det);
S_c2 = filtfilt(b_c2, a_c2, S_det);
S_el = filtfilt(b_el, a_el, S_det);

%% ── Step 5: Welch PSD → BPM estimate ─────────────────────────────────────
win_sec  = 10;
winLen   = round(win_sec * fs);
novlp    = round(winLen * 0.5);
[pxx, f_psd] = pwelch(S_bw, hann(winLen), novlp, 4096, fs);

cardiac_mask = f_psd >= f_low & f_psd <= f_high;
f_cardiac    = f_psd(cardiac_mask);
pxx_cardiac  = pxx(cardiac_mask);
[~, pk_idx]  = max(pxx_cardiac);
f_peak       = f_cardiac(pk_idx);
bpm_est      = f_peak * 60;

% Harmonic correction: if peak is below ~1 Hz, check whether 2x is stronger
if f_peak < 1.0
    f2_mask = f_psd >= 1.5*f_peak & f_psd <= 2.5*f_peak & f_psd <= f_high;
    if any(f2_mask)
        [pxx2_max, pk2_idx] = max(pxx(f2_mask));
        if pxx2_max > 0.3 * pxx_cardiac(pk_idx)
            f2_vec  = f_psd(f2_mask);
            bpm_est = f2_vec(pk2_idx) * 60;
        end
    end
end
fprintf('Welch BPM estimate: %.1f BPM\n', bpm_est);

%% ── Step 6: Export CSV ───────────────────────────────────────────────────
gt_bpm_frame = interp1(gt_time, gt_hr, t_axis', 'linear', NaN);

out = table( ...
    (1:T)', t_axis', G_t', G_n', lum_t', detected_t', skin_count_t', ...
    S_det, S_bw, S_c1, S_c2, S_el, gt_bpm_frame, ...
    'VariableNames', {'frame_index','time_s','G_skin_raw','G_normalized', ...
        'frame_luminance','face_detected','skin_pixel_count', ...
        'BVP_detrended','BVP_butterworth','BVP_cheby1','BVP_cheby2','BVP_elliptic','gt_bpm'});
writetable(out, OUT_PATH);
fprintf('Saved: %s  (%d rows x %d cols)\n', OUT_PATH, height(out), width(out));

%% ── Step 7: Summary plots ─────────────────────────────────────────────────
figure('Position',[50 50 1400 750],'Name','rPPG — HoG+LBP Segmentation');

subplot(3,1,1);
plot(t_axis, G_t); xlabel('Time (s)'); ylabel('G mean');
title('Green Channel — HoG+LBP Skin Pixels'); grid on;

subplot(3,1,2);
plot(t_axis, S_bw); xlabel('Time (s)'); ylabel('BVP');
title('BVP Signal (Butterworth filtered)'); grid on;

subplot(3,1,3);
plot(f_psd*60, pxx); xlim([40 220]);
xline(bpm_est, 'r--', 'LineWidth', 1.5, 'Label', sprintf('%.1f BPM', bpm_est));
xlabel('BPM'); ylabel('PSD'); title('Welch PSD — Butterworth BVP'); grid on;

figure('Position',[50 830 1400 280],'Name','Skin Pixel Count');
plot(t_axis, skin_count_t); xlabel('Time (s)'); ylabel('Pixel count');
title('Skin pixels per frame (HoG+LBP mask)'); grid on;

%% ── Helper functions ──────────────────────────────────────────────────────

function skinMask = buildAdvancedSkinMask(faceROI, pts, faceBox)
% Multi-CS voting skin mask refined by HoG gradient energy + LBP variance
% texture rejection. Removes hair/eyebrow regions that pass color tests.
% Avoids per-frame GMM fitting for video-speed operation: jointly rejects
% pixels in the top-70th-percentile of both gradient energy and LBP variance.
    [faceH, faceW, ~] = size(faceROI);
    x0 = faceBox(1);  y0 = faceBox(2);

    clampPt  = @(p) min(max(p,[1 1]),[faceW faceH]);
    lEye_f   = clampPt(pts(1,:) - [x0-1, y0-1]);
    rEye_f   = clampPt(pts(2,:) - [x0-1, y0-1]);
    nose_f   = clampPt(pts(3,:) - [x0-1, y0-1]);
    lMouth_f = clampPt(pts(4,:) - [x0-1, y0-1]);
    rMouth_f = clampPt(pts(5,:) - [x0-1, y0-1]);
    eyeMid   = (lEye_f + rEye_f) / 2;
    mouthMid = (lMouth_f + rMouth_f) / 2;
    eyeDist  = norm(rEye_f - lEye_f);
    if eyeDist < 5, eyeDist = faceW * 0.35; end   % fallback for degenerate detections

    [xx, yy] = meshgrid(1:faceW, 1:faceH);

    %-- Face ellipse mask -------------------------------------------------
    ellCx    = faceW / 2;
    ellCy    = (eyeMid(2) + mouthMid(2)) / 2;
    faceMask = ((xx-ellCx)/(faceW*0.48)).^2 + ((yy-ellCy)/(faceH*0.50)).^2 <= 1;
    faceMask = imclose(imfill(faceMask,'holes'), strel('disk',5));

    %-- Feature exclusion (eyes, brows, nose, lips) -----------------------
    EW = round(eyeDist*0.30);  EH = round(eyeDist*0.15);
    BW = round(eyeDist*0.35);  BH = round(eyeDist*0.10);  BYOFF = eyeDist*0.24;
    MW = round(eyeDist*0.45);  MH = round(eyeDist*0.18);
    nW    = eyeDist * 0.55;
    nTopY = eyeMid(2)  + eyeDist * 0.05;
    nBotY = nose_f(2)  + eyeDist * 0.15;

    lEyeMask  = bbToRectMask(clampBB([lEye_f(1)-EW,   lEye_f(2)-EH,       2*EW, 2*EH], faceH, faceW), faceH, faceW, 6);
    rEyeMask  = bbToRectMask(clampBB([rEye_f(1)-EW,   rEye_f(2)-EH,       2*EW, 2*EH], faceH, faceW), faceH, faceW, 6);
    lBrowMask = bbToRectMask(clampBB([lEye_f(1)-BW,   lEye_f(2)-BYOFF-BH, 2*BW, 2*BH], faceH, faceW), faceH, faceW, 6);
    rBrowMask = bbToRectMask(clampBB([rEye_f(1)-BW,   rEye_f(2)-BYOFF-BH, 2*BW, 2*BH], faceH, faceW), faceH, faceW, 6);
    mthMask   = bbToRectMask(clampBB([mouthMid(1)-MW, mouthMid(2)-MH,     2*MW, 2*MH], faceH, faceW), faceH, faceW, 6);

    nosePts = [nose_f(1)-nW*0.22, nTopY; nose_f(1)+nW*0.22, nTopY; ...
               nose_f(1)+nW*0.55, nose_f(2); nose_f(1)+nW*0.58, nBotY; ...
               nose_f(1)-nW*0.58, nBotY;    nose_f(1)-nW*0.55, nose_f(2)];
    nosePts(:,1) = min(max(nosePts(:,1),1),faceW);
    nosePts(:,2) = min(max(nosePts(:,2),1),faceH);
    noseMask  = imdilate(poly2mask(nosePts(:,1),nosePts(:,2),faceH,faceW), strel('disk',5));

    excludeMask = lEyeMask | rEyeMask | lBrowMask | rBrowMask | mthMask | noseMask;

    %-- Candidate skin zone (forehead + left/right cheeks) ----------------
    browBottomY = min(lEye_f(2),rEye_f(2)) - round(eyeDist*0.10);
    foreTopY    = max(1, round(browBottomY - eyeDist*0.60));
    botY        = min(faceH-2, round(mouthMid(2) + eyeDist*0.55));
    mkPoly = @(P) poly2mask(min(max(P(:,1),1),faceW), min(max(P(:,2),1),faceH), faceH, faceW);

    candidateZone = ...
        mkPoly([faceW*0.04,foreTopY;                 faceW*0.96,foreTopY;              faceW*0.96,browBottomY-2;          faceW*0.04,browBottomY-2]) | ...
        mkPoly([faceW*0.02,browBottomY;              nose_f(1)-eyeDist*0.10,browBottomY; nose_f(1)-eyeDist*0.10,botY;    faceW*0.02,botY]) | ...
        mkPoly([nose_f(1)+eyeDist*0.10,browBottomY;  faceW*0.98,browBottomY;           faceW*0.98,botY;                  nose_f(1)+eyeDist*0.10,botY]);
    candidateZone = candidateZone & faceMask;

    %-- Multi-CS voting: per-side adaptive YCbCr + RGB + HSV + shadow -----
    ycbcr = rgb2ycbcr(faceROI);
    Y_ch  = double(ycbcr(:,:,1));
    Cb_ch = double(ycbcr(:,:,2));
    Cr_ch = double(ycbcr(:,:,3));

    leftHalf = xx < nose_f(1);
    seedBase = candidateZone & ~excludeMask & ...
        (Cb_ch>=77 & Cb_ch<=127) & (Cr_ch>=133 & Cr_ch<=173) & (Y_ch>=20 & Y_ch<=245);
    ycbcrSkin = false(faceH, faceW);
    for side = 1:2
        if side == 1, halfM = leftHalf; else, halfM = ~leftHalf; end
        sMask = seedBase & halfM;
        if nnz(sMask) < 30, sMask = candidateZone & ~excludeMask & halfM; end
        if nnz(sMask) >= 10
            sCb = Cb_ch(sMask);  sCr = Cr_ch(sMask);  sY = Y_ch(sMask);
            cLo = max(55,  mean(sCb)-2.5*std(sCb));
            cHi = min(148, mean(sCb)+2.5*std(sCb));
            rLo = max(110, mean(sCr)-2.5*std(sCr));
            rHi = min(195, mean(sCr)+2.5*std(sCr));
            yL  = max(8,   prctile(sY,0.5)-10);
            yH  = min(252, prctile(sY,99.5)+10);
        else
            cLo=77; cHi=127; rLo=133; rHi=173; yL=20; yH=245;
        end
        ycbcrSkin = ycbcrSkin | (halfM & Cb_ch>=cLo & Cb_ch<=cHi & ...
            Cr_ch>=rLo & Cr_ch<=rHi & Y_ch>=yL & Y_ch<=yH);
    end

    Rd = double(faceROI(:,:,1));
    Gd = double(faceROI(:,:,2));
    Bd = double(faceROI(:,:,3));
    rgbSkin = (Rd>95) & (Gd>40) & (Bd>20) & ...
        (max(cat(3,Rd,Gd,Bd),[],3) - min(cat(3,Rd,Gd,Bd),[],3) > 15) & ...
        (abs(Rd-Gd)>15) & (Rd>Gd) & (Rd>Bd);

    hsv     = rgb2hsv(faceROI);
    H_ch    = hsv(:,:,1) * 360;
    S_ch    = hsv(:,:,2);
    V_ch    = hsv(:,:,3);
    hsvSkin = ((H_ch>=0 & H_ch<=50) | (H_ch>=330 & H_ch<=360)) & ...
        (S_ch>=0.10 & S_ch<=0.75) & (V_ch>=0.15);

    shadowSkin = (Cb_ch>=60 & Cb_ch<=145) & (Cr_ch>=115 & Cr_ch<=190) & ...
        (Y_ch>=10 & Y_ch<45) & (S_ch>=0.08);

    voteMap   = double(ycbcrSkin) + double(rgbSkin) + double(hsvSkin);
    votedSkin = ((voteMap >= 2) | shadowSkin) & candidateZone & faceMask & ~excludeMask;
    votedSkin = imclose(imfill(bwareaopen(votedSkin,15),'holes'), strel('disk',3));

    %-- HoG gradient energy + LBP variance texture rejection --------------
    grayFace  = double(rgb2gray(faceROI));
    localMean = imgaussfilt(grayFace, max(15, round(eyeDist*0.4)));
    normGray  = (grayFace ./ (localMean + 1)) * 128;

    [Gmag, ~]  = imgradient(normGray);
    gradEnergy = imgaussfilt(Gmag, 1.5);

    offsets = [-1 -1; -1 0; -1 1; 0 1; 1 1; 1 0; 1 -1; 0 -1];
    padGray = padarray(normGray, [1 1], 'replicate');
    lbpMap  = zeros(faceH, faceW);
    for k = 1:8
        shifted = padGray(1+1+offsets(k,1):faceH+1+offsets(k,1), ...
                          1+1+offsets(k,2):faceW+1+offsets(k,2));
        lbpMap = lbpMap + (shifted >= normGray) * 2^(k-1);
    end
    lbpVar = stdfilt(lbpMap, ones(7));

    % Reject pixels where BOTH gradient energy and LBP variance are high
    % (joint high value = hair/eyebrow texture signature)
    workIdx = find(votedSkin);
    if numel(workIdx) >= 100
        ge_vals  = gradEnergy(workIdx);
        lbp_vals = lbpVar(workIdx);
        ge_thr   = prctile(ge_vals,  70);
        lbp_thr  = prctile(lbp_vals, 70);
        isTexture = (ge_vals > ge_thr) & (lbp_vals > lbp_thr);
        textureMask = false(faceH, faceW);
        textureMask(workIdx(isTexture)) = true;
        cleanSkin = votedSkin & ~textureMask;
    else
        cleanSkin = votedSkin;
    end

    %-- Specular highlight removal + final morphology ---------------------
    specular = (Y_ch > 240) | (Rd > 248 & Gd > 248);
    cleanSkin = cleanSkin & ~specular;
    skinMask  = imfill(imclose(bwareaopen(cleanSkin, 50), strel('disk',3)), 'holes');
end

function bb = clampBB(bb, H, W)
    bb    = round(bb);
    bb(1) = max(1, bb(1));
    bb(2) = max(1, bb(2));
    bb(3) = max(1, min(bb(3), W-bb(1)));
    bb(4) = max(1, min(bb(4), H-bb(2)));
end

function mask = bbToRectMask(bb, H, W, dil)
    mask = false(H, W);
    if isempty(bb), return; end
    x1 = max(1, bb(1));       y1 = max(1, bb(2));
    x2 = min(W, bb(1)+bb(3)); y2 = min(H, bb(2)+bb(4));
    if x2 >= x1 && y2 >= y1, mask(y1:y2, x1:x2) = true; end
    if dil > 0, mask = imdilate(mask, strel('disk',dil)); end
end

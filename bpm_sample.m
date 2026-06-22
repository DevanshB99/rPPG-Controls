%% bpm_sample.m
clear; clc; close all;

%% ── Paths ────────────────────────────────────────────────────────────────
BASE     = '/Users/devanshbajwala/Documents/VS Code WS/rPPG/rPPG Controls/Measurement Data/DocBOT_2026-04-07_15-50-04';
VID_PATH = fullfile(BASE, 'recording_2026-04-07T22-50-04Z.mov');
CSV_PATH = fullfile(BASE, 'vitals.csv');
OUT_PATH = fullfile('/Users/devanshbajwala/Documents/VS Code WS/rPPG/rPPG Controls', 'bpm_sample_output.csv');

%% ── Ground truth ──────────────────────────────────────────────────────────
csv_data = readtable(CSV_PATH);
gt_time  = csv_data.offset_seconds;
gt_hr    = csv_data.heart_rate;
valid_gt = ~isnan(gt_hr);
gt_time  = gt_time(valid_gt);
gt_hr    = double(gt_hr(valid_gt));
fprintf('Ground truth: %d samples  mean=%.1f BPM  range=[%d %d]\n', ...
    numel(gt_hr), mean(gt_hr), min(gt_hr), max(gt_hr));

%% ── Step 1: Initialize — video metadata + first MTCNN detection for fallback
vid_init    = VideoReader(VID_PATH);
fs          = vid_init.FrameRate;
vidDuration = vid_init.Duration;
first_frame = readFrame(vid_init);
[H_vid, W_vid, ~] = size(first_frame);
clear vid_init;

detector = mtcnn.Detector();
[bboxes, ~, mtcnnLand] = detector.detect(first_frame);
if isempty(bboxes), error('No face detected in frame 1.'); end
[~, best]    = max(bboxes(:,3) .* bboxes(:,4));
faceBox_last = clampBB(round(bboxes(best,:)), H_vid, W_vid);
pts_last     = squeeze(mtcnnLand(best,:,:));
fprintf('First detection: faceBox=[%d %d %d %d]\n', faceBox_last);

%% ── Step 2: Per-frame MTCNN tracking + skin detection + channel extraction
nEst       = ceil(vidDuration * fs) + 20;
R_t        = zeros(1, nEst);
G_t        = zeros(1, nEst);
B_t        = zeros(1, nEst);
lum_t      = zeros(1, nEst);
detected_t = zeros(1, nEst);  % 1 = fresh MTCNN detection, 0 = fallback
fIdx       = 0;
nFallback  = 0;

vid = VideoReader(VID_PATH);
while hasFrame(vid)
    frame = readFrame(vid);

    % MTCNN detection on this frame — updates bounding box and landmarks
    [bboxes_f, ~, mtcnnLand_f] = detector.detect(frame);
    if ~isempty(bboxes_f)
        [~, best_f]  = max(bboxes_f(:,3) .* bboxes_f(:,4));
        faceBox_cur  = clampBB(round(bboxes_f(best_f,:)), H_vid, W_vid);
        pts_cur      = squeeze(mtcnnLand_f(best_f,:,:));
        faceBox_last = faceBox_cur;
        pts_last     = pts_cur;
        det_flag     = 1;
    else
        % Fallback: reuse last known detection
        faceBox_cur = faceBox_last;
        pts_cur     = pts_last;
        det_flag    = 0;
        nFallback   = nFallback + 1;
    end

    faceROI  = imcrop(frame, faceBox_cur);
    skinMask = buildSkinMask(faceROI, pts_cur, faceBox_cur);

    if nnz(skinMask) < 50, continue; end

    % Luminance normalisation before channel extraction (removes AGC drift)
    fcd       = double(faceROI);
    frame_lum = mean(fcd(:));
    if frame_lum > 0, fcd = fcd / frame_lum * 128; end

    msk  = skinMask(:);
    pix  = reshape(fcd, [], 3);
    fIdx = fIdx + 1;
    R_t(fIdx)        = mean(pix(msk,1));
    G_t(fIdx)        = mean(pix(msk,2));
    B_t(fIdx)        = mean(pix(msk,3));
    lum_t(fIdx)      = frame_lum;
    detected_t(fIdx) = det_flag;
end

R_t        = R_t(1:fIdx);
G_t        = G_t(1:fIdx);
B_t        = B_t(1:fIdx);
lum_t      = lum_t(1:fIdx);
detected_t = detected_t(1:fIdx);
T          = fIdx;
t_axis     = (0:T-1) / fs;
fprintf('Extracted %d frames (%.1fs) at %.2fHz  |  MTCNN fresh: %d  fallback: %d\n', ...
    T, T/fs, fs, sum(detected_t), nFallback);

%% ── Step 3: CHROM projection + linear detrend ─────────────────────────────
R_n   = R_t / mean(R_t);
G_n   = G_t / mean(G_t);
B_n   = B_t / mean(B_t);
Xs    = 3*R_n - 2*G_n;
Ys    = 1.5*R_n + G_n - 1.5*B_n;
if std(Ys) < eps, alpha = 1; else, alpha = std(Xs) / std(Ys); end
S     = Xs - alpha * Ys;

t_vec  = (1:T)';
coeffs = [t_vec, ones(T,1)] \ S(:);
S_det  = S(:) - (coeffs(1)*t_vec + coeffs(2));

%% ── Step 4: Four IIR bandpass filters ─────────────────────────────────────
f_low = 0.7;  f_high = 3.5;
order = 2;    Rp = 0.5;  Rs = 40;
Wn = [f_low f_high] / (fs/2);

[b_bw, a_bw] = butter(order,         Wn, 'bandpass');
[b_c1, a_c1] = cheby1(order, Rp,     Wn, 'bandpass');
[b_c2, a_c2] = cheby2(order, Rs,     Wn, 'bandpass');
[b_el, a_el] = ellip( order, Rp, Rs, Wn, 'bandpass');

S_bw = filtfilt(b_bw, a_bw, S_det);
S_c1 = filtfilt(b_c1, a_c1, S_det);
S_c2 = filtfilt(b_c2, a_c2, S_det);
S_el = filtfilt(b_el, a_el, S_det);

%% ── Step 5: Export CSV ────────────────────────────────────────────────────
gt_bpm_frame = interp1(gt_time, gt_hr, t_axis', 'linear', NaN);

out = table( ...
    (1:T)', t_axis', G_t', G_n', lum_t', detected_t', S_det, S_bw, S_c1, S_c2, S_el, gt_bpm_frame, ...
    'VariableNames', {'frame_index','time_s','G_skin_raw','G_normalized', ...
        'frame_luminance','face_detected','BVP_detrended','BVP_butterworth','BVP_cheby1', ...
        'BVP_cheby2','BVP_elliptic','gt_bpm'});

writetable(out, OUT_PATH);
fprintf('Saved: %s  (%d rows x %d cols)\n', OUT_PATH, height(out), width(out));

%% ── Helper functions ──────────────────────────────────────────────────────

function skinMask = buildSkinMask(faceROI, pts, faceBox)
% Computes skin mask for one face crop using FaceSegment_HoG_LBP multi-CS
% voting (§3–5). pts: 5×2 MTCNN landmarks in full-frame coordinates.
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

    [xx, yy] = meshgrid(1:faceW, 1:faceH);

    % Face ellipse (§3)
    ellCx    = faceW / 2;
    ellCy    = (eyeMid(2) + mouthMid(2)) / 2;
    faceMask = ((xx-ellCx)/(faceW*0.48)).^2 + ((yy-ellCy)/(faceH*0.50)).^2 <= 1;
    faceMask = imclose(imfill(faceMask,'holes'), strel('disk',5));

    % Feature exclusion — eyes, brows, nose, lips (§4)
    EW = round(eyeDist*0.30);  EH = round(eyeDist*0.15);
    BW = round(eyeDist*0.35);  BH = round(eyeDist*0.10);  BYOFF = eyeDist*0.24;
    MW = round(eyeDist*0.45);  MH = round(eyeDist*0.18);
    nW    = eyeDist*0.55;
    nTopY = eyeMid(2)  + eyeDist*0.05;
    nBotY = nose_f(2)  + eyeDist*0.15;

    lEyeMask  = bbToRectMask(clampBB([lEye_f(1)-EW,   lEye_f(2)-EH,       2*EW, 2*EH], faceH, faceW), faceH, faceW, 6);
    rEyeMask  = bbToRectMask(clampBB([rEye_f(1)-EW,   rEye_f(2)-EH,       2*EW, 2*EH], faceH, faceW), faceH, faceW, 6);
    lBrowMask = bbToRectMask(clampBB([lEye_f(1)-BW,   lEye_f(2)-BYOFF-BH, 2*BW, 2*BH], faceH, faceW), faceH, faceW, 6);
    rBrowMask = bbToRectMask(clampBB([rEye_f(1)-BW,   rEye_f(2)-BYOFF-BH, 2*BW, 2*BH], faceH, faceW), faceH, faceW, 6);
    mthMask   = bbToRectMask(clampBB([mouthMid(1)-MW,  mouthMid(2)-MH,     2*MW, 2*MH], faceH, faceW), faceH, faceW, 6);

    nosePts = [nose_f(1)-nW*0.22, nTopY; nose_f(1)+nW*0.22, nTopY; ...
               nose_f(1)+nW*0.55, nose_f(2); nose_f(1)+nW*0.58, nBotY; ...
               nose_f(1)-nW*0.58, nBotY;    nose_f(1)-nW*0.55, nose_f(2)];
    nosePts(:,1) = min(max(nosePts(:,1),1),faceW);
    nosePts(:,2) = min(max(nosePts(:,2),1),faceH);
    noseMask  = imdilate(poly2mask(nosePts(:,1),nosePts(:,2),faceH,faceW), strel('disk',5));

    excludeMask = lEyeMask | rEyeMask | lBrowMask | rBrowMask | mthMask | noseMask;

    % Candidate skin zone — forehead + cheeks (§5)
    browBottomY   = min(lEye_f(2),rEye_f(2)) - round(eyeDist*0.10);
    foreTopY      = max(1, round(browBottomY - eyeDist*0.60));
    botY          = min(faceH-2, round(mouthMid(2) + eyeDist*0.55));
    mkPoly        = @(P) poly2mask(min(max(P(:,1),1),faceW), min(max(P(:,2),1),faceH), faceH, faceW);
    candidateZone = ...
        mkPoly([faceW*0.04,foreTopY;                faceW*0.96,foreTopY;              faceW*0.96,browBottomY-2;          faceW*0.04,browBottomY-2]) | ...
        mkPoly([faceW*0.02,browBottomY;             nose_f(1)-eyeDist*0.10,browBottomY; nose_f(1)-eyeDist*0.10,botY;    faceW*0.02,botY]) | ...
        mkPoly([nose_f(1)+eyeDist*0.10,browBottomY; faceW*0.98,browBottomY;           faceW*0.98,botY;                  nose_f(1)+eyeDist*0.10,botY]);
    candidateZone = candidateZone & faceMask;

    % Method 1: Per-side adaptive YCbCr (§5)
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

    % Method 2: RGB rule-based (§5)
    Rd = double(faceROI(:,:,1));  Gd = double(faceROI(:,:,2));  Bd = double(faceROI(:,:,3));
    rgbSkin = (Rd>95) & (Gd>40) & (Bd>20) & ...
        (max(cat(3,Rd,Gd,Bd),[],3) - min(cat(3,Rd,Gd,Bd),[],3) > 15) & ...
        (abs(Rd-Gd)>15) & (Rd>Gd) & (Rd>Bd);

    % Method 3: HSV skin range (§5)
    hsv     = rgb2hsv(faceROI);
    H_ch    = hsv(:,:,1) * 360;
    S_ch    = hsv(:,:,2);
    V_ch    = hsv(:,:,3);
    hsvSkin = ((H_ch>=0 & H_ch<=50) | (H_ch>=330 & H_ch<=360)) & ...
        (S_ch>=0.10 & S_ch<=0.75) & (V_ch>=0.15);

    % Shadow recovery (§5)
    shadowSkin = (Cb_ch>=60 & Cb_ch<=145) & (Cr_ch>=115 & Cr_ch<=190) & ...
        (Y_ch>=10 & Y_ch<45) & (S_ch>=0.08);

    % Voting: pixel is skin if 2+ methods agree, or shadow recovery (§5)
    voteMap  = double(ycbcrSkin) + double(rgbSkin) + double(hsvSkin);
    skinMask = ((voteMap >= 2) | shadowSkin) & candidateZone & faceMask & ~excludeMask;
    skinMask = imclose(imfill(bwareaopen(skinMask,15),'holes'), strel('disk',3));
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

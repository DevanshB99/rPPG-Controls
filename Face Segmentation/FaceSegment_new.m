%% Adaptive Face Segmentation Pipeline v2
% Pure skin pixel extraction with ROI classification.
% Toolboxes: Computer Vision, Image Processing, Deep Learning, Statistics & ML, Facial Landmarks.
clear; clc; close all;

%% 1. Input Image
I = imread('Photo on 4-5-26 at 10.31.jpg');
if size(I,3)==1, I = repmat(I,[1 1 3]); end

%% 2. MTCNN Face Detection & Crop
detector = mtcnn.Detector();
[bboxes, ~, mtcnnLand] = detector.detect(I);
if isempty(bboxes), error('No face detected.'); end
[~, best] = max(bboxes(:,3).*bboxes(:,4));
faceBox   = round(bboxes(best,:));
pts       = squeeze(mtcnnLand(best,:,:));
faceROI   = imcrop(I, faceBox);
x0 = faceBox(1);  y0 = faceBox(2);
[faceH, faceW, ~] = size(faceROI);

clampPt  = @(p) min(max(p,[1,1]),[faceW,faceH]);
lEye_f   = clampPt(pts(1,:)-[x0-1,y0-1]);
rEye_f   = clampPt(pts(2,:)-[x0-1,y0-1]);
nose_f   = clampPt(pts(3,:)-[x0-1,y0-1]);
lMouth_f = clampPt(pts(4,:)-[x0-1,y0-1]);
rMouth_f = clampPt(pts(5,:)-[x0-1,y0-1]);
eyeMid   = (lEye_f + rEye_f)/2;
mouthMid = (lMouth_f + rMouth_f)/2;
eyeDist  = norm(rEye_f - lEye_f);

%% 3. Pixel-wise Background Removal (YCbCr skin + face shape)
[xx, yy] = meshgrid(1:faceW, 1:faceH);
faceYCbCr = rgb2ycbcr(faceROI);
Y  = double(faceYCbCr(:,:,1));
Cb = double(faceYCbCr(:,:,2));
Cr = double(faceYCbCr(:,:,3));
roughSkin = (Cb>=77 & Cb<=127) & (Cr>=133 & Cr<=173);
% Generous ellipse — keep all face features, reject only clear background
ellCx = faceW/2;  ellCy = (eyeMid(2)+mouthMid(2))/2;
ellipse = ((xx-ellCx)/(faceW*0.48)).^2 + ((yy-ellCy)/(faceH*0.50)).^2 <= 1;
% Combine: pixel is face if inside ellipse OR skin-colored near face center
faceMask = ellipse | (roughSkin & ...
    ((xx-ellCx)/(faceW*0.55)).^2 + ((yy-ellCy)/(faceH*0.58)).^2 <= 1);
faceMask = imclose(imfill(faceMask,'holes'), strel('disk',5));
faceMask = bwareaopen(faceMask, round(faceH*faceW*0.05));
bgRemoved = faceROI .* uint8(repmat(faceMask,[1 1 3]));

%% 4. Landmark Exclusion (Eyes, Brows, Nose, Lips)
clipPoly = @(P) [min(max(P(:,1),1),faceW), min(max(P(:,2),1),faceH)];
makeMask = @(poly,m) imdilate(poly2mask(poly(:,1),poly(:,2),faceH,faceW), strel('disk',m));

EW = round(eyeDist*0.30);  EH = round(eyeDist*0.15);
BW = round(eyeDist*0.35);  BH = round(eyeDist*0.10);  BYOFF = eyeDist*0.24;
MW = round(eyeDist*0.45);  MH = round(eyeDist*0.18);  MAR = 6;

lEyeBB  = clampBB([lEye_f(1)-EW, lEye_f(2)-EH, 2*EW, 2*EH], faceH, faceW);
rEyeBB  = clampBB([rEye_f(1)-EW, rEye_f(2)-EH, 2*EW, 2*EH], faceH, faceW);
lBrowBB = clampBB([lEye_f(1)-BW, lEye_f(2)-BYOFF-BH, 2*BW, 2*BH], faceH, faceW);
rBrowBB = clampBB([rEye_f(1)-BW, rEye_f(2)-BYOFF-BH, 2*BW, 2*BH], faceH, faceW);
mthBB   = clampBB([mouthMid(1)-MW, mouthMid(2)-MH, 2*MW, 2*MH], faceH, faceW);

[lEyeLand,lEyeOK]   = runToolbox(faceROI,lEyeBB,@eyesProcessing,5,faceH,faceW);
[rEyeLand,rEyeOK]   = runToolbox(faceROI,rEyeBB,@eyesProcessing,5,faceH,faceW);
[mthLand,mthOK]      = runToolbox(faceROI,mthBB,@mouthProcessing,5,faceH,faceW);
[lBrowLand,lBrowOK]  = runToolbox(faceROI,lBrowBB,@eyebrowsProcessing,2,faceH,faceW);
[rBrowLand,rBrowOK]  = runToolbox(faceROI,rBrowBB,@eyebrowsProcessing,2,faceH,faceW);

lEyeMask = buildFeatureMask(lEyeOK,lEyeLand,lEyeBB,faceH,faceW,MAR,clipPoly,makeMask);
rEyeMask = buildFeatureMask(rEyeOK,rEyeLand,rEyeBB,faceH,faceW,MAR,clipPoly,makeMask);

if mthOK
    lipMask = makeMask(clipPoly([mthLand(1),mthLand(2); mthLand(5),mthLand(6); ...
        mthLand(3),mthLand(4); mthLand(7),mthLand(8)]), MAR);
else
    lipMask = makeMask(clipPoly([lMouth_f(1)-eyeDist*0.06,mouthMid(2); ...
        mouthMid(1),mouthMid(2)-eyeDist*0.14; rMouth_f(1)+eyeDist*0.06,mouthMid(2); ...
        mouthMid(1),mouthMid(2)+eyeDist*0.20]), MAR);
end

lBrowMask = buildBrowMask(lBrowOK,lBrowLand,lBrowBB,faceH,faceW,MAR,clipPoly,makeMask);
rBrowMask = buildBrowMask(rBrowOK,rBrowLand,rBrowBB,faceH,faceW,MAR,clipPoly,makeMask);

nW = eyeDist*0.55;  nTopY = eyeMid(2)+eyeDist*0.05;  nBotY = nose_f(2)+eyeDist*0.15;
noseMask = makeMask(clipPoly([nose_f(1)-nW*0.22,nTopY; nose_f(1)+nW*0.22,nTopY; ...
    nose_f(1)+nW*0.55,nose_f(2); nose_f(1)+nW*0.58,nBotY; ...
    nose_f(1)-nW*0.58,nBotY; nose_f(1)-nW*0.55,nose_f(2)]), 5);

precisionExcludeMask = lEyeMask | rEyeMask | lBrowMask | rBrowMask | noseMask | lipMask;

%% 5. Multi-Color-Space Skin Detection with Shadow Compensation
grayFace = double(rgb2gray(faceROI));
R = double(faceROI(:,:,1)); G = double(faceROI(:,:,2)); B = double(faceROI(:,:,3));

% Local illumination normalization
localMean = imgaussfilt(grayFace, max(15, round(eyeDist*0.4)));
normGray  = (grayFace ./ (localMean + 1)) * 128;

faceHSV = rgb2hsv(faceROI);
H_ch = faceHSV(:,:,1) * 360;  % Hue in degrees
S_ch = faceHSV(:,:,2);         % Saturation [0-1]
V_ch = faceHSV(:,:,3);         % Value [0-1]

% --- ROI zone definitions ---
browBottomY = min(lEye_f(2),rEye_f(2)) - round(eyeDist*0.10);
foreTopY    = max(1, round(browBottomY - eyeDist*0.60));
eyeBottomY  = max(lEye_f(2),rEye_f(2)) + round(eyeDist*0.12);
botY        = min(faceH-2, round(mouthMid(2) + eyeDist*0.55));

fp = clipPoly([faceW*0.04,foreTopY; faceW*0.96,foreTopY; ...
    faceW*0.96,browBottomY-2; faceW*0.04,browBottomY-2]);
foreheadMask = poly2mask(fp(:,1),fp(:,2),faceH,faceW) & faceMask;

lp = clipPoly([faceW*0.02,browBottomY; nose_f(1)-eyeDist*0.10,browBottomY; ...
    nose_f(1)-eyeDist*0.10,botY; faceW*0.02,botY]);
lSideMask = poly2mask(lp(:,1),lp(:,2),faceH,faceW) & faceMask;

rp = clipPoly([nose_f(1)+eyeDist*0.10,browBottomY; faceW*0.98,browBottomY; ...
    faceW*0.98,botY; nose_f(1)+eyeDist*0.10,botY]);
rSideMask = poly2mask(rp(:,1),rp(:,2),faceH,faceW) & faceMask;

candidateMask = foreheadMask | lSideMask | rSideMask;

% Temple / cheek split — wider temple coverage
tempBotY    = eyeBottomY + round(eyeDist*0.25);
lTempleMask = lSideMask & (yy <= tempBotY) & (xx <= lEye_f(1)+round(eyeDist*0.08));
rTempleMask = rSideMask & (yy <= tempBotY) & (xx >= rEye_f(1)-round(eyeDist*0.08));
lCheekMask  = lSideMask & ~lTempleMask;
rCheekMask  = rSideMask & ~rTempleMask;

% --- Method 1: Per-side Adaptive YCbCr ---
% Split face at nose to handle asymmetric lighting
leftHalf  = xx < nose_f(1);
rightHalf = xx >= nose_f(1);

seedBase = candidateMask & ~precisionExcludeMask & ...
    (Cb>=77 & Cb<=127) & (Cr>=133 & Cr<=173) & (Y>=20 & Y<=245);

ycbcrSkin = false(faceH, faceW);
for side = 1:2
    if side == 1, halfM = leftHalf; else, halfM = rightHalf; end
    sMask = seedBase & halfM;
    if nnz(sMask) < 30
        sMask = candidateMask & ~precisionExcludeMask & halfM;
    end
    if nnz(sMask) >= 10
        sY = Y(sMask); sCb = Cb(sMask); sCr = Cr(sMask);
        cLo = max(55,  mean(sCb)-2.5*std(sCb));
        cHi = min(148, mean(sCb)+2.5*std(sCb));
        rLo = max(110, mean(sCr)-2.5*std(sCr));
        rHi = min(195, mean(sCr)+2.5*std(sCr));
        yL  = max(8,   prctile(sY,0.5)-10);
        yH  = min(252, prctile(sY,99.5)+10);
    else
        cLo=77; cHi=127; rLo=133; rHi=173; yL=20; yH=245;
    end
    ycbcrSkin = ycbcrSkin | (halfM & ...
        (Cb>=cLo & Cb<=cHi) & (Cr>=rLo & Cr<=rHi) & (Y>=yL & Y<=yH));
end

% --- Method 2: RGB rule-based (from paper) ---
rgbSkin = (R>95) & (G>40) & (B>20) & ...
    ((max(cat(3,R,G,B),[],3) - min(cat(3,R,G,B),[],3)) > 15) & ...
    (abs(R-G) > 15) & (R > G) & (R > B);

% --- Method 3: HSV skin range ---
% Skin hue: 0-50 deg (reddish-yellowish), or 330-360 deg (reddish)
hsvSkin = ((H_ch >= 0 & H_ch <= 50) | (H_ch >= 330 & H_ch <= 360)) & ...
    (S_ch >= 0.10 & S_ch <= 0.75) & (V_ch >= 0.15);

% --- Shadow recovery: Cb/Cr in skin range but Y is low ---
shadowSkin = (Cb>=60 & Cb<=145) & (Cr>=115 & Cr<=190) & ...
    (Y >= 10 & Y < 45) & (S_ch >= 0.08);

% --- Combine: pixel is skin if at least 2 of 3 methods agree, or shadow recovery ---
voteMap = double(ycbcrSkin) + double(rgbSkin) + double(hsvSkin);
combinedSkin = (voteMap >= 2) | shadowSkin;

% Apply to candidate region
ycbcrSkinMask = combinedSkin & candidateMask & faceMask;
ycbcrSkinMask = imclose(imfill(bwareaopen(ycbcrSkinMask,15),'holes'), strel('disk',3));

%% 6. HoG Gradient Energy & LBP Texture on Skin Regions
[Gmag, ~] = imgradient(normGray);
gradEnergy = imgaussfilt(Gmag, 1.5);

% Per-pixel LBP (8-neighbor, radius 1)
offsets = [-1 -1; -1 0; -1 1; 0 1; 1 1; 1 0; 1 -1; 0 -1];
padGray = padarray(normGray, [1 1], 'replicate');
lbpMap  = zeros(faceH, faceW);
for k = 1:8
    shifted = padGray(1+1+offsets(k,1):faceH+1+offsets(k,1), ...
                      1+1+offsets(k,2):faceW+1+offsets(k,2));
    lbpMap = lbpMap + (shifted >= normGray) * 2^(k-1);
end
lbpVar = stdfilt(lbpMap, ones(7));

%% 7. GMM Clustering with Adaptive K (replaces K-Means)
workMask = ycbcrSkinMask & ~precisionExcludeMask & faceMask;
workIdx  = find(workMask);

if numel(workIdx) >= 300
    feat = [gradEnergy(workIdx), lbpVar(workIdx), normGray(workIdx), ...
            Cb(workIdx), Cr(workIdx)];
    feat = (feat - min(feat)) ./ (max(feat) - min(feat) + eps);
    feat(:,1:2) = feat(:,1:2) * 2.0;

    rng(42);

    % --- K-Means K=2 for comparison plot ---
    [cIdx2, C2] = kmeans(feat, 2, 'Replicates', 5, 'MaxIter', 200);
    [~, skinC2] = min(C2(:,1));
    skinMask_K2 = false(faceH, faceW);
    skinMask_K2(workIdx(cIdx2 == skinC2)) = true;
    hairMask_K2 = false(faceH, faceW);
    hairMask_K2(workIdx(cIdx2 ~= skinC2)) = true;

    % --- GMM: fit K=2,3,4 and pick best BIC ---
    bestBIC = inf;  bestK = 2;  bestGMM = [];
    for tryK = 2:4
        try
            gm = fitgmdist(feat, tryK, 'Replicates', 3, ...
                'RegularizationValue', 0.01, 'Options', statset('MaxIter',300));
            if gm.BIC < bestBIC
                bestBIC = gm.BIC;  bestK = tryK;  bestGMM = gm;
            end
        catch
        end
    end
    fprintf('GMM selected K=%d (BIC=%.0f)\n', bestK, bestBIC);

    % Posterior probabilities for each component
    posteriors = bestGMM.posterior(feat);
    [~, gmmIdx] = max(posteriors, [], 2);

    % Identify skin component: lowest mean gradient energy
    [~, skinComp] = min(bestGMM.mu(:,1));

    % Classify by posterior probability threshold
    skinProb = posteriors(:, skinComp);
    isSkin = skinProb >= 0.4;  % Soft threshold — captures borderline skin

    smoothSkinMask = false(faceH, faceW);
    smoothSkinMask(workIdx(isSkin)) = true;

    % Identify hair component: highest mean gradient energy
    [~, hairComp] = max(bestGMM.mu(:,1));
    facialHairMask = false(faceH, faceW);
    facialHairMask(workIdx(gmmIdx == hairComp & ~isSkin)) = true;

    % Mixed: everything else
    mixedMask = false(faceH, faceW);
    mixedMask(workIdx(~isSkin & gmmIdx ~= hairComp)) = true;

    % Recover mixed pixels with low gradient (shadow skin)
    mixedIdx = workIdx(~isSkin & gmmIdx ~= hairComp);
    if ~isempty(mixedIdx)
        mixGrad = gradEnergy(mixedIdx);
        skinGradThresh = prctile(gradEnergy(workIdx(isSkin)), 75);
        recoverable = mixGrad < skinGradThresh;
        smoothSkinMask(mixedIdx(recoverable)) = true;
        mixedMask(mixedIdx(recoverable)) = false;
    end
else
    smoothSkinMask = workMask;
    mixedMask = false(faceH, faceW);
    facialHairMask = false(faceH, faceW);
    skinMask_K2 = workMask;
    hairMask_K2 = false(faceH, faceW);
    bestK = 0;
end

%% 8. Reject Non-Skin & Assemble Final Mask
specularMask = (Y > 240) | (R > 248 & G > 248);
finalSkinMask = smoothSkinMask & ~precisionExcludeMask & ~specularMask & faceMask;
finalSkinMask = imfill(imclose(bwareaopen(finalSkinMask,80), strel('disk',3)), 'holes');

%% 9. ROI Labeling & Overlay Verification
roiLabelMap = zeros(faceH, faceW);
roiLabelMap(finalSkinMask & foreheadMask) = 1;
roiLabelMap(finalSkinMask & lTempleMask)  = 2;
roiLabelMap(finalSkinMask & rTempleMask)  = 3;
roiLabelMap(finalSkinMask & lCheekMask)   = 4;
roiLabelMap(finalSkinMask & rCheekMask)   = 5;

skinOnlyRGB = faceROI;
skinOnlyRGB(repmat(~finalSkinMask,[1 1 3])) = 0;

roiNames = {'Forehead','L.Temple','R.Temple','L.Cheek','R.Cheek'};
fprintf('\n--- ROI Statistics ---\n');
for r = 1:5
    m = roiLabelMap == r;
    if nnz(m) > 0
        fprintf('  %-10s: %5d px  meanY=%.1f  meanCr=%.1f\n', ...
            roiNames{r}, nnz(m), mean(Y(m)), mean(Cr(m)));
    end
end
fprintf('  Total skin: %d px (%.1f%% of face)\n', ...
    nnz(finalSkinMask), 100*nnz(finalSkinMask)/(faceH*faceW));

%% 10. Pipeline Visualization (8 panels)
figure('Position',[50 50 1600 800],'Name','Face Segmentation Pipeline');

subplot(2,4,1); imshow(I); title('1. Original Image');

subplot(2,4,2);
Iviz = insertShape(I,'Rectangle',faceBox,'Color','green','LineWidth',3);
imshow(Iviz); title('2. Face Detected');

subplot(2,4,3); imshow(bgRemoved); title('3. BG Removed');

subplot(2,4,4);
excViz = faceROI;
rv=excViz(:,:,1); gv=excViz(:,:,2); bv=excViz(:,:,3);
rv(precisionExcludeMask)=255; gv(precisionExcludeMask)=0; bv(precisionExcludeMask)=80;
imshow(cat(3,rv,gv,bv)); title('4. Landmark Exclusion');

subplot(2,4,5);
ycViz = faceROI .* uint8(repmat(ycbcrSkinMask,[1 1 3]));
imshow(ycViz); title('5. Multi-CS Skin');

subplot(2,4,6);
kmViz = faceROI;
rv=kmViz(:,:,1); gv=kmViz(:,:,2); bv=kmViz(:,:,3);
rv(smoothSkinMask)=0;    gv(smoothSkinMask)=200;  bv(smoothSkinMask)=0;
rv(mixedMask)=255;       gv(mixedMask)=200;       bv(mixedMask)=0;
rv(facialHairMask)=255;  gv(facialHairMask)=80;   bv(facialHairMask)=0;
imshow(cat(3,rv,gv,bv));
title(sprintf('7. GMM K=%d: Skin(G) Mixed(Y) Hair(O)', bestK));

subplot(2,4,7);
roiColors = [1 1 0; 0 1 1; 1 0 1; 0 0.8 0; 0 0.4 1];
roiViz = double(faceROI)/255;
for r = 1:5
    m = roiLabelMap == r;
    for ch = 1:3
        plane = roiViz(:,:,ch);
        plane(m) = plane(m)*0.4 + roiColors(r,ch)*0.6;
        roiViz(:,:,ch) = plane;
    end
end
imshow(roiViz); title('9. ROI Labels');

subplot(2,4,8); imshow(skinOnlyRGB); title('9. Skin Only');

%% 11. Diagnostic Plots
% Figure 2: HoG gradient energy overlay + LBP variance + K-means comparison
figure('Position',[50 50 1600 500],'Name','Texture & Clustering Diagnostics');

subplot(1,4,1);
imshow(faceROI); hold on;
hogViz = gradEnergy .* double(faceMask);
hIm = imagesc(hogViz); hIm.AlphaData = 0.5 * double(faceMask);
colormap(gca,'hot'); colorbar; title('HoG Gradient Energy Overlay');
hold off;

subplot(1,4,2);
lbpViz = lbpVar .* double(faceMask);
imshow(faceROI); hold on;
hLbp = imagesc(lbpViz); hLbp.AlphaData = 0.5 * double(faceMask);
colormap(gca,'parula'); colorbar; title('LBP Variance Overlay');
hold off;

subplot(1,4,3);
k2Viz = zeros(faceH, faceW, 3);
k2Viz(:,:,2) = double(skinMask_K2);
k2Viz(:,:,1) = double(hairMask_K2);
imshow(k2Viz*0.6 + double(faceROI)/255*0.4); title('K=2 Clusters');

subplot(1,4,4);
k3Viz = zeros(faceH, faceW, 3);
k3Viz(:,:,2) = double(smoothSkinMask);
k3Viz(:,:,1) = double(facialHairMask)*0.8 + double(mixedMask)*0.5;
k3Viz(:,:,3) = double(mixedMask)*0.5;
imshow(k3Viz*0.6 + double(faceROI)/255*0.4);
title(sprintf('GMM K=%d Clusters', bestK));

% Figure 3: Gradient histogram — skin vs hair vs mixed
figure('Position',[50 600 900 400],'Name','Gradient Analysis');
subplot(1,2,1);
if nnz(smoothSkinMask) > 0
    histogram(gradEnergy(smoothSkinMask), 50, 'FaceColor','g', 'FaceAlpha',0.5); hold on;
end
if nnz(facialHairMask) > 0
    histogram(gradEnergy(facialHairMask), 50, 'FaceColor','r', 'FaceAlpha',0.5);
end
if nnz(mixedMask) > 0
    histogram(gradEnergy(mixedMask), 50, 'FaceColor',[1 0.8 0], 'FaceAlpha',0.5);
end
legend('Skin','Hair','Mixed'); xlabel('Gradient Energy'); ylabel('Pixel Count');
title('HoG Gradient Distribution'); hold off;

subplot(1,2,2);
if nnz(smoothSkinMask) > 0
    histogram(lbpVar(smoothSkinMask), 50, 'FaceColor','g', 'FaceAlpha',0.5); hold on;
end
if nnz(facialHairMask) > 0
    histogram(lbpVar(facialHairMask), 50, 'FaceColor','r', 'FaceAlpha',0.5);
end
if nnz(mixedMask) > 0
    histogram(lbpVar(mixedMask), 50, 'FaceColor',[1 0.8 0], 'FaceAlpha',0.5);
end
legend('Skin','Hair','Mixed'); xlabel('LBP Variance'); ylabel('Pixel Count');
title('LBP Variance Distribution'); hold off;

%% ── Helper Functions ──────────────────────────────────────────────────────

function mask = buildFeatureMask(ok, land, bb, faceH, faceW, mar, clipPoly, makeMask)
    if ok && length(land) >= 8
        mask = makeMask(clipPoly([land(1),land(2); land(5),land(6); ...
            land(3),land(4); land(7),land(8)]), mar);
    else
        mask = bbToMask(bb, faceH, faceW, mar);
    end
end

function mask = buildBrowMask(ok, land, bb, faceH, faceW, mar, clipPoly, makeMask)
    bH = max(6, bb(4));
    if ok && length(land) >= 4
        mask = makeMask(clipPoly([land(1),land(2); land(3),land(4); ...
            land(3),land(4)-bH; land(1),land(2)-bH]), mar);
    else
        mask = bbToMask(bb, faceH, faceW, mar);
    end
end

function bb = clampBB(bb, H, W)
    bb = round(bb);
    bb(1) = max(1,bb(1));  bb(2) = max(1,bb(2));
    bb(3) = max(1, min(bb(3), W-bb(1)));
    bb(4) = max(1, min(bb(4), H-bb(2)));
end

function roi = safeROI(img, bb)
    [H,W,~] = size(img);
    x1=max(1,bb(1)); y1=max(1,bb(2));
    x2=min(W,bb(1)+bb(3)); y2=min(H,bb(2)+bb(4));
    if x2<=x1 || y2<=y1, roi=[]; else, roi=img(y1:y2,x1:x2,:); end
end

function mask = bbToMask(bb, H, W, dil)
    mask = false(H,W);
    if isempty(bb), return; end
    x1=max(1,bb(1)); y1=max(1,bb(2));
    x2=min(W,bb(1)+bb(3)); y2=min(H,bb(2)+bb(4));
    if x2>=x1 && y2>=y1, mask(y1:y2,x1:x2)=true; end
    if dil>0, mask=imdilate(mask,strel('disk',dil)); end
end

function [land, ok] = runToolbox(faceROI, bb, processFn, landcont, faceH, faceW)
    land = []; ok = false;
    if isempty(bb), return; end
    roi = safeROI(faceROI, bb);
    if isempty(roi) || size(roi,1)<8 || size(roi,2)<8, return; end
    try
        [V, C] = processFn(roi, landcont);
        land = Landmarks(V, C, bb, landcont);
        xs = land(1:2:end); ys = land(2:2:end);
        ok = all(xs>=1 & xs<=faceW & ys>=1 & ys<=faceH) && ...
             (max(xs)-min(xs))>5 && (max(ys)-min(ys))>3;
    catch
    end
end

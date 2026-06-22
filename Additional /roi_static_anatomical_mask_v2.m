%% roi_static_anatomical_mask_v2.m
% Improved static-frame anatomical usable-skin mask for rPPG
% Main changes vs v1:
% - adaptive YCbCr + normalized rg chromaticity candidate model
% - tighter anatomical exclusion masks
% - ROI-local reconstruction with fill/cleanup
% - post-growth rescue of nearby candidate components
% - cleaner lower-face/beard handling
%
% Outputs:
% - roi_summary.csv
% - per-ROI pixel CSVs
% - masks and overlay PNGs
% - MAT bundle

clear; clc; close all;

%% settings
imagePath = '/Users/devanshbajwala/Downloads/Photo on 4-5-26 at 10.31.jpg'; 
outDir = 'roi_static_anatomical_mask_outputs_v2';
useManualFaceBox = false;
manualFaceBox = []; % [x y w h]

if ~exist(outDir,'dir'); mkdir(outDir); end

%% image
Iu8 = imread(imagePath);
if size(Iu8,3)==1
    Iu8 = repmat(Iu8,[1 1 3]);
end
Id = im2double(Iu8);

%% face box
if useManualFaceBox
    bb = round(manualFaceBox);
else
    bb = [];
    try
        fd = vision.CascadeObjectDetector();
        bboxes = step(fd, Iu8);
        if ~isempty(bboxes)
            [~,idx] = max(bboxes(:,3).*bboxes(:,4));
            bb = round(bboxes(idx,:));
        end
    catch
    end
    if isempty(bb)
        bb = [1 1 size(Iu8,2) size(Iu8,1)];
    end
end

x = max(1,bb(1)); y = max(1,bb(2));
w = min(bb(3), size(Iu8,2)-x+1);
h = min(bb(4), size(Iu8,1)-y+1);
bb = [x y w h];

Iface_u8 = imcrop(Iu8,[x y w-1 h-1]);
Iface_d  = im2double(Iface_u8);
[H,W,~] = size(Iface_u8);

%% channels
YCbCr = rgb2ycbcr(Iface_u8);
Y  = double(YCbCr(:,:,1));
Cb = double(YCbCr(:,:,2));
Cr = double(YCbCr(:,:,3));

R = Iface_d(:,:,1);
G = Iface_d(:,:,2);
B = Iface_d(:,:,3);

HSV = rgb2hsv(Iface_d);
V = HSV(:,:,3);
grayI = im2gray(Iface_d);
Gmag = imgradient(grayI);

rgbSum = R + G + B + eps;
rNorm = R ./ rgbSum;
gNorm = G ./ rgbSum;

[Xg,Yg] = meshgrid(1:W,1:H);
xn = (Xg - W/2)/(W/2);
yn = (Yg - H/2)/(H/2);
faceEllipse = (xn/0.90).^2 + (yn/1.02).^2 <= 1;

%% anchors
anchors = estimateAnchors(Iface_u8);

%% adaptive skin candidate
seedYCbCr = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173) & (Y > 40) & faceEllipse;
seedYCbCr = bwareaopen(seedYCbCr, 60);

if nnz(seedYCbCr) > 250
    cbVals = Cb(seedYCbCr); crVals = Cr(seedYCbCr); yVals = Y(seedYCbCr);
    rVals  = rNorm(seedYCbCr); gVals = gNorm(seedYCbCr);

    cbMu = mean(cbVals); cbSd = std(cbVals);
    crMu = mean(crVals); crSd = std(crVals);
    rMu  = mean(rVals);  rSd  = std(rVals);
    gMu  = mean(gVals);  gSd  = std(gVals);

    cbLo = max(65,  cbMu - 1.45*cbSd); cbHi = min(140, cbMu + 1.45*cbSd);
    crLo = max(125, crMu - 1.45*crSd); crHi = min(180, crMu + 1.45*crSd);
    yLo  = max(38,  prctile(yVals,4) - 4);  yHi = min(242, prctile(yVals,99) + 2);

    rLo = max(0.20, rMu - 1.6*rSd); rHi = min(0.55, rMu + 1.6*rSd);
    gLo = max(0.20, gMu - 1.6*gSd); gHi = min(0.50, gMu + 1.6*gSd);

    maskYCbCr = (Cb >= cbLo & Cb <= cbHi) & (Cr >= crLo & Cr <= crHi) & (Y >= yLo & Y <= yHi);
    maskChrom = (rNorm >= rLo & rNorm <= rHi) & (gNorm >= gLo & gNorm <= gHi);

    adaptiveMask = faceEllipse & maskYCbCr & maskChrom;
else
    adaptiveMask = seedYCbCr;
end

adaptiveMask = imopen(adaptiveMask, strel('disk',1));
adaptiveMask = imclose(adaptiveMask, strel('disk',2));
adaptiveMask = bwareaopen(adaptiveMask, 50);

%% exclusion masks
leftEyeMask  = ellipseMask(H,W,anchors.leftEyeCenter,  anchors.eyeRx, anchors.eyeRy);
rightEyeMask = ellipseMask(H,W,anchors.rightEyeCenter, anchors.eyeRx, anchors.eyeRy);
eyeMask = leftEyeMask | rightEyeMask;

browMask = false(H,W);
browY1 = max(1, round(min([anchors.leftEyeCenter(2), anchors.rightEyeCenter(2)]) - 0.95*anchors.eyeRy));
browY2 = min(H, round(browY1 + 0.38*anchors.eyeRy));
browX1 = max(1, round(anchors.leftEyeCenter(1)  - 0.95*anchors.eyeRx));
browX2 = min(W, round(anchors.rightEyeCenter(1) + 0.95*anchors.eyeRx));
browMask(browY1:browY2, browX1:browX2) = true;

mouthMask = ellipseMask(H,W,anchors.mouthCenter, anchors.mouthRx, anchors.mouthRy);

hairlineMask = false(H,W);
hairlineY = max(1, round(0.12*H));
hairlineMask(1:hairlineY,:) = true;
hairlineMask = hairlineMask & (Y < max(75, prctile(Y(seedYCbCr),30))) & faceEllipse;

earMask = false(H,W);
earMask(:,1:max(1,round(0.05*W))) = true;
earMask(:,min(W,round(0.95*W)):W) = true;
earMask = earMask & (Yg > 0.24*H) & (Yg < 0.80*H) & faceEllipse;

beardMask = false(H,W);
beardMask = (Yg > 0.68*H) & ...
            (Y < max(60, prctile(Y(seedYCbCr),12))) & ...
            (Gmag > prctile(Gmag(:),55)) & faceEllipse;

specularMask = (Y > 245) | (V > 0.985);

excludeMask = eyeMask | browMask | mouthMask | hairlineMask | earMask | beardMask | specularMask;

candidateMask = adaptiveMask & ~excludeMask;
candidateMask = imopen(candidateMask, strel('disk',1));
candidateMask = imclose(candidateMask, strel('disk',2));
candidateMask = bwareaopen(candidateMask, 35);

%% ROIs
roi = struct();
roi.forehead    = boundedRect(round(0.24*W), round(0.09*H), round(0.52*W), round(0.15*H), W, H);
roi.left_cheek  = boundedRect(round(anchors.leftEyeCenter(1)  - 1.05*anchors.eyeRx), round(anchors.leftEyeCenter(2)  + 0.62*anchors.eyeRy), round(0.23*W), round(0.19*H), W, H);
roi.right_cheek = boundedRect(round(anchors.rightEyeCenter(1) - 0.82*anchors.eyeRx), round(anchors.rightEyeCenter(2) + 0.62*anchors.eyeRy), round(0.23*W), round(0.19*H), W, H);
roi.nose        = boundedRect(round(anchors.noseCenter(1) - 0.08*W), round(anchors.leftEyeCenter(2) + 0.22*anchors.eyeRy), round(0.16*W), round(0.20*H), W, H);
roi.full_face   = [1 1 W H];
roiNames = fieldnames(roi);

seedCenter.forehead    = [round(0.50*W), round(0.17*H)];
seedCenter.left_cheek  = [round(anchors.leftEyeCenter(1)),  round(anchors.leftEyeCenter(2)  + 0.95*anchors.eyeRy)];
seedCenter.right_cheek = [round(anchors.rightEyeCenter(1)), round(anchors.rightEyeCenter(2) + 0.95*anchors.eyeRy)];
seedCenter.nose        = [round(anchors.noseCenter(1)),     round(anchors.noseCenter(2))];

%% ROI-local reconstruction + rescue
usableUnion = false(H,W);
usableROI = struct();

for k = 1:numel(roiNames)
    name = roiNames{k};
    if strcmp(name,'full_face')
        continue;
    end

    r = roi.(name);
    roiMask = rectToMask(r,W,H);
    candidateROI = candidateMask & roiMask;

    seedDisk = diskMask(H,W,seedCenter.(name), max(3,round(0.014*min(H,W))));
    seedMaskROI = seedDisk & candidateROI;

    if ~any(seedMaskROI(:))
        [sr,sc] = nearestCandidate(candidateROI, seedCenter.(name));
        if ~isempty(sr)
            seedMaskROI = diskMask(H,W,[sc sr], max(2,round(0.012*min(H,W)))) & candidateROI;
        end
    end

    if any(seedMaskROI(:))
        rg = imreconstruct(seedMaskROI, candidateROI);
        rg = imfill(rg,'holes');
        rg = imclose(rg, strel('disk',2));

        % rescue nearby candidate components close to current mask
        if any(rg(:))
            nearMask = imdilate(rg, strel('disk',10)) & candidateROI;
            rg = nearMask;
            rg = largestCentralComponent(rg, seedCenter.(name));
            rg = imfill(rg,'holes');
        end
    else
        rg = false(H,W);
    end

    usableROI.(name) = rg & roiMask;
    usableUnion = usableUnion | usableROI.(name);
end

usableROI.full_face = usableUnion;
usableMask = usableUnion;

%% per-pixel tables + summary
summaryRows = {};
overlay = Iface_u8;

for k = 1:numel(roiNames)
    name = roiNames{k};
    r = roi.(name);
    roiMask = rectToMask(r,W,H);

    adaptivePixels = candidateMask & roiMask;
    usablePixels   = usableROI.(name) & roiMask;
    contamPixels   = adaptivePixels & ~usablePixels;

    [yy,xx] = find(roiMask);
    lin = sub2ind([H W],yy,xx);

    T = table;
    T.x = xx;
    T.y = yy;
    T.R = R(lin);
    T.G = G(lin);
    T.B = B(lin);
    T.Y = Y(lin);
    T.Cb = Cb(lin);
    T.Cr = Cr(lin);
    T.isAdaptiveSkin = adaptivePixels(lin);
    T.isUsable = usablePixels(lin);
    T.isEye = eyeMask(lin);
    T.isBrow = browMask(lin);
    T.isMouth = mouthMask(lin);
    T.isHairline = hairlineMask(lin);
    T.isEar = earMask(lin);
    T.isBeard = beardMask(lin);
    T.isSpecular = specularMask(lin);

    q1 = prctile(T.Y,33); q2 = prctile(T.Y,66);
    lc = strings(height(T),1);
    lc(T.Y <= q1) = "dark";
    lc(T.Y > q1 & T.Y <= q2) = "medium";
    lc(T.Y > q2) = "bright";
    T.lightClass = lc;

    writetable(T, fullfile(outDir, sprintf('%s_pixels.csv',name)));

    summaryRows(end+1,:) = {name, nnz(roiMask), nnz(adaptivePixels), nnz(usablePixels), ...
        nnz(contamPixels), nnz(eyeMask & adaptivePixels), nnz(browMask & adaptivePixels), ...
        nnz(mouthMask & adaptivePixels), nnz(hairlineMask & adaptivePixels), ...
        nnz(earMask & adaptivePixels), nnz(beardMask & adaptivePixels), ...
        nnz(specularMask & adaptivePixels)}; %#ok<AGROW>
end

summaryTable = cell2table(summaryRows, 'VariableNames', ...
    {'roi','totalPixels','adaptiveSkinPixels','usablePixels','contaminatedPixels', ...
     'eyePixels','browPixels','mouthPixels','hairlinePixels','earPixels','beardPixels','specularPixels'});

writetable(summaryTable, fullfile(outDir,'roi_summary.csv'));

%% overlay + save
redMask = candidateMask & ~usableMask;
greenMask = usableMask;

overlay(:,:,1) = uint8(double(overlay(:,:,1)).*(~redMask) + 255*double(redMask));
overlay(:,:,2) = uint8(double(overlay(:,:,2)).*(~greenMask) + 255*double(greenMask));
overlay(:,:,3) = uint8(double(overlay(:,:,3)).*(~(redMask | greenMask)));

imwrite(Iface_u8, fullfile(outDir,'face_roi.png'));
imwrite(seedYCbCr, fullfile(outDir,'seed_mask.png'));
imwrite(adaptiveMask, fullfile(outDir,'adaptive_mask.png'));
imwrite(candidateMask, fullfile(outDir,'candidate_mask.png'));
imwrite(excludeMask, fullfile(outDir,'exclude_mask.png'));
imwrite(usableMask, fullfile(outDir,'usable_mask.png'));
imwrite(overlay, fullfile(outDir,'usable_overlay.png'));

save(fullfile(outDir,'roi_static_anatomical_mask_results_v2.mat'), ...
    'bb','anchors','roi','seedYCbCr','adaptiveMask','candidateMask','excludeMask','usableMask','summaryTable');

%% display
figure('Name','Static anatomical usable-skin mask v2');
subplot(2,3,1); imshow(Id); title('Input image'); hold on;
rectangle('Position',bb,'EdgeColor','g','LineWidth',1.2);

subplot(2,3,2); imshow(Iface_u8); title('Face ROI'); hold on;
plot(anchors.leftEyeCenter(1),anchors.leftEyeCenter(2),'g+','MarkerSize',8,'LineWidth',1.5);
plot(anchors.rightEyeCenter(1),anchors.rightEyeCenter(2),'g+','MarkerSize',8,'LineWidth',1.5);
plot(anchors.noseCenter(1),anchors.noseCenter(2),'g+','MarkerSize',8,'LineWidth',1.5);
plot(anchors.mouthCenter(1),anchors.mouthCenter(2),'g+','MarkerSize',8,'LineWidth',1.5);
for k = 1:numel(roiNames)
    rectangle('Position',roi.(roiNames{k}),'LineWidth',1.0);
end

subplot(2,3,3); imshow(adaptiveMask); title('Adaptive skin candidate');
subplot(2,3,4); imshow(excludeMask); title('Exclusions');
subplot(2,3,5); imshow(usableMask); title('Usable pixels');
subplot(2,3,6); imshow(overlay); title('Green=usable, Red=filtered');

disp(summaryTable);
fprintf('Saved outputs to: %s\n', outDir);

%% local functions
function anchors = estimateAnchors(Iface_u8)
    [H,W,~] = size(Iface_u8);
    anchors.leftEyeCenter  = [round(0.35*W), round(0.38*H)];
    anchors.rightEyeCenter = [round(0.65*W), round(0.38*H)];
    anchors.noseCenter     = [round(0.50*W), round(0.55*H)];
    anchors.mouthCenter    = [round(0.50*W), round(0.74*H)];
    anchors.eyeRx = round(0.08*W);
    anchors.eyeRy = round(0.045*H);
    anchors.mouthRx = round(0.11*W);
    anchors.mouthRy = round(0.05*H);

    try
        led = vision.CascadeObjectDetector('LeftEye');
        red = vision.CascadeObjectDetector('RightEye');
        nd  = vision.CascadeObjectDetector('Nose');
        md  = vision.CascadeObjectDetector('Mouth');

        LB = step(led, Iface_u8);
        RB = step(red, Iface_u8);
        NB = step(nd,  Iface_u8);
        MB = step(md,  Iface_u8);

        if ~isempty(LB)
            [~,i] = max(LB(:,3).*LB(:,4)); b = LB(i,:);
            anchors.leftEyeCenter = [b(1)+b(3)/2, b(2)+b(4)/2];
            anchors.eyeRx = max(anchors.eyeRx, round(0.36*b(3)));
            anchors.eyeRy = max(anchors.eyeRy, round(0.28*b(4)));
        end
        if ~isempty(RB)
            [~,i] = max(RB(:,3).*RB(:,4)); b = RB(i,:);
            anchors.rightEyeCenter = [b(1)+b(3)/2, b(2)+b(4)/2];
            anchors.eyeRx = max(anchors.eyeRx, round(0.36*b(3)));
            anchors.eyeRy = max(anchors.eyeRy, round(0.28*b(4)));
        end
        if ~isempty(NB)
            [~,i] = max(NB(:,3).*NB(:,4)); b = NB(i,:);
            anchors.noseCenter = [b(1)+b(3)/2, b(2)+b(4)/2];
        end
        if ~isempty(MB)
            [~,i] = max(MB(:,3).*MB(:,4)); b = MB(i,:);
            anchors.mouthCenter = [b(1)+b(3)/2, b(2)+b(4)/2];
            anchors.mouthRx = max(anchors.mouthRx, round(0.34*b(3)));
            anchors.mouthRy = max(anchors.mouthRy, round(0.22*b(4)));
        end
    catch
    end
end

function r = boundedRect(x,y,w,h,W,H)
    x = max(1, round(x));
    y = max(1, round(y));
    w = max(1, round(min(w, W-x+1)));
    h = max(1, round(min(h, H-y+1)));
    r = [x y w h];
end

function mask = rectToMask(r,W,H)
    mask = false(H,W);
    x1 = r(1); y1 = r(2);
    x2 = min(W, x1+r(3)-1); y2 = min(H, y1+r(4)-1);
    mask(y1:y2, x1:x2) = true;
end

function mask = ellipseMask(H,W,center,rx,ry)
    [X,Y] = meshgrid(1:W,1:H);
    cx = center(1); cy = center(2);
    mask = ((X-cx).^2)/(max(rx,1)^2) + ((Y-cy).^2)/(max(ry,1)^2) <= 1;
end

function mask = diskMask(H,W,center,r)
    [X,Y] = meshgrid(1:W,1:H);
    cx = center(1); cy = center(2);
    mask = (X-cx).^2 + (Y-cy).^2 <= max(r,1)^2;
end

function [row,col] = nearestCandidate(mask, center)
    [yy,xx] = find(mask);
    if isempty(xx)
        row = []; col = [];
        return;
    end
    d2 = (xx-center(1)).^2 + (yy-center(2)).^2;
    [~,i] = min(d2);
    row = yy(i); col = xx(i);
end

function out = largestCentralComponent(mask, center)
    CC = bwconncomp(mask);
    if CC.NumObjects == 0
        out = false(size(mask));
        return;
    end
    scores = inf(CC.NumObjects,1);
    for i = 1:CC.NumObjects
        [yy,xx] = ind2sub(size(mask), CC.PixelIdxList{i});
        cx = mean(xx); cy = mean(yy);
        scores(i) = (cx-center(1))^2 + (cy-center(2))^2;
    end
    [~,idx] = min(scores);
    out = false(size(mask));
    out(CC.PixelIdxList{idx}) = true;
end

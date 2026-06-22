%% roi_anchor_regiongrow_filter.m
% Pixelwise usable-skin masking with:
% 1) face detection
% 2) anchor estimation (eyes / nose / mouth if available, else geometric fallback)
% 3) adaptive YCbCr skin candidate mask
% 4) exclusion masks (eyes, brows, mouth/lips, ears, hairline)
% 5) region growing inside each ROI
% 6) ROI pixel tables + summary stats + overlay outputs

clear; clc; close all;

%% settings
imagePath = '/Users/devanshbajwala/Downloads/Photo on 4-5-26 at 10.31.jpg'; 
outDir = 'roi_anchor_regiongrow_outputs';
useManualFaceBox = false;
manualFaceBox = []; % [x y w h]

if ~exist(outDir,'dir'); mkdir(outDir); end

%% read image
Iu8 = imread(imagePath);
if size(Iu8,3)==1
    Iu8 = repmat(Iu8,[1 1 3]);
end
Id = im2double(Iu8);

%% face detection
if useManualFaceBox
    bb = round(manualFaceBox);
else
    bb = [];
    % RetinaFace face detector (R2025a+)
    try
        det = faceDetector("small-network");
        bboxes = detect(det,Iu8);
        if ~isempty(bboxes)
            [~,idx] = max(bboxes(:,3).*bboxes(:,4));
            bb = round(bboxes(idx,:));
        end
    catch
    end

    % fallback to Viola-Jones if needed
    if isempty(bb)
        try
            fd = vision.CascadeObjectDetector();
            bboxes = step(fd,Iu8);
            if ~isempty(bboxes)
                [~,idx] = max(bboxes(:,3).*bboxes(:,4));
                bb = round(bboxes(idx,:));
            end
        catch
        end
    end

    if isempty(bb)
        bb = [1 1 size(Iu8,2) size(Iu8,1)];
    end
end

x = max(1,bb(1));
y = max(1,bb(2));
w = min(bb(3), size(Iu8,2)-x+1);
h = min(bb(4), size(Iu8,1)-y+1);
bb = [x y w h];

Iface_u8 = imcrop(Iu8,[x y w-1 h-1]);
Iface_d  = im2double(Iface_u8);
[H,W,~] = size(Iface_u8);

%% anchor estimation inside face ROI
anchors = estimateFaceAnchors(Iface_u8);

%% color spaces
YCbCr = rgb2ycbcr(Iface_u8);
Y  = double(YCbCr(:,:,1));
Cb = double(YCbCr(:,:,2));
Cr = double(YCbCr(:,:,3));
R = Iface_d(:,:,1);
G = Iface_d(:,:,2);
B = Iface_d(:,:,3);

%% initial YCbCr seed
seedMask = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173) & (Y > 40);

[Xg,Yg] = meshgrid(1:W,1:H);
xn = (Xg - W/2)/(W/2);
yn = (Yg - H/2)/(H/2);
faceEllipse = (xn/0.90).^2 + (yn/1.02).^2 <= 1;
seedMask = seedMask & faceEllipse;

%% adaptive YCbCr refinement from seed
if nnz(seedMask) > 100
    cbVals = Cb(seedMask); crVals = Cr(seedMask); yVals = Y(seedMask);
    cbMu = mean(cbVals); cbSd = std(cbVals);
    crMu = mean(crVals); crSd = std(crVals);

    cbLo = max(60,  cbMu - 1.7*cbSd);
    cbHi = min(145, cbMu + 1.7*cbSd);
    crLo = max(120, crMu - 1.7*crSd);
    crHi = min(185, crMu + 1.7*crSd);
    yLo  = max(35, prctile(yVals, 4)  - 5);
    yHi  = min(245,prctile(yVals, 98) + 5);

    adaptiveMask = (Cb >= cbLo & Cb <= cbHi) & ...
                   (Cr >= crLo & Cr <= crHi) & ...
                   (Y  >= yLo  & Y  <= yHi) & faceEllipse;
else
    adaptiveMask = seedMask;
end

%% exclusion masks from anchors
leftEyeMask  = ellipseMask(H,W,anchors.leftEyeCenter,  anchors.eyeRx, anchors.eyeRy);
rightEyeMask = ellipseMask(H,W,anchors.rightEyeCenter, anchors.eyeRx, anchors.eyeRy);
eyeMask = leftEyeMask | rightEyeMask;

browMask = false(H,W);
browY1 = max(1, round(min([anchors.leftEyeCenter(2),anchors.rightEyeCenter(2)]) - 1.3*anchors.eyeRy));
browY2 = max(browY1, min(H, round(browY1 + 0.65*anchors.eyeRy)));
browX1 = max(1, round(anchors.leftEyeCenter(1)  - 1.5*anchors.eyeRx));
browX2 = min(W, round(anchors.rightEyeCenter(1) + 1.5*anchors.eyeRx));
browMask(browY1:browY2, browX1:browX2) = true;

mouthMask = ellipseMask(H,W,anchors.mouthCenter, anchors.mouthRx, anchors.mouthRy);

hairlineMask = false(H,W);
hairlineY = max(1, round(anchors.foreheadTopY));
hairlineMask(1:hairlineY,:) = true;

earMask = false(H,W);
earMask(:,1:max(1,round(0.08*W))) = true;
earMask(:,min(W,round(0.92*W)):W) = true;
earMask = earMask & faceEllipse;

specularMask = (Y > 245) | (R > 0.98) | (G > 0.98) | (B > 0.98);
darkMask = Y < max(55, prctile(Y(adaptiveMask),10));
beardBand = Yg > round(0.62*H);
beardMask = darkMask & beardBand;

excludeMask = eyeMask | browMask | mouthMask | hairlineMask | earMask | beardMask | specularMask;
candidateMask = adaptiveMask & ~excludeMask;

%% ROI definitions from anchors
roi = struct();
roi.forehead    = boundedRect(round(0.23*W), max(1, round(anchors.foreheadBottomY - 0.20*H)), round(0.54*W), round(0.18*H), W, H);
roi.left_cheek  = boundedRect(max(1, round(anchors.leftEyeCenter(1)  - 1.35*anchors.eyeRx)), round(anchors.leftEyeCenter(2)  + 0.55*anchors.eyeRy), round(0.26*W), round(0.23*H), W, H);
roi.right_cheek = boundedRect(round(anchors.rightEyeCenter(1) - 0.85*anchors.eyeRx), round(anchors.rightEyeCenter(2) + 0.55*anchors.eyeRy), round(0.26*W), round(0.23*H), W, H);
roi.nose        = boundedRect(round(anchors.noseCenter(1) - 0.10*W), round(anchors.leftEyeCenter(2) + 0.20*anchors.eyeRy), round(0.20*W), round(0.28*H), W, H);
roi.full_face   = [1 1 W H];
roiNames = fieldnames(roi);

%% region growing per ROI
regionGrowMasks = struct();
usableUnion = false(H,W);

for k = 1:numel(roiNames)
    name = roiNames{k};
    r = roi.(name);
    roiMask = rectToMask(r,W,H);

    localCandidates = candidateMask & roiMask;
    [seedRow, seedCol] = chooseSeed(localCandidates, roiMask);

    if isempty(seedRow)
        rg = false(H,W);
    else
        tol = estimateTolerance(Y, localCandidates, roiMask);
        try
            rg = grayconnected(uint8(Y), seedRow, seedCol, tol);
        catch
            % fallback for older versions: simple intensity band around seed
            seedVal = Y(seedRow,seedCol);
            rg = abs(Y - seedVal) <= tol;
        end
        rg = rg & localCandidates;
    end

    regionGrowMasks.(name) = rg;
    if ~strcmp(name,'full_face')
        usableUnion = usableUnion | rg;
    end
end

regionGrowMasks.full_face = usableUnion;
usableMask = usableUnion;

%% per-pixel tables and summary
summaryRows = {};
overlay = Iface_u8;

for k = 1:numel(roiNames)
    name = roiNames{k};
    r = roi.(name);
    roiMask = rectToMask(r,W,H);

    adaptivePixels = candidateMask & roiMask;
    usablePixels   = regionGrowMasks.(name);
    contamPixels   = adaptivePixels & ~usablePixels;

    combinedEyeMask = eyeMask;

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
    T.isEye = combinedEyeMask(lin);
    T.isBrow = browMask(lin);
    T.isMouth = mouthMask(lin);
    T.isHairline = hairlineMask(lin);
    T.isEar = earMask(lin);
    T.isBeard = beardMask(lin);
    T.isSpecular = specularMask(lin);

    yvals = T.Y;
    q1 = prctile(yvals,33);
    q2 = prctile(yvals,66);
    lightClass = strings(height(T),1);
    lightClass(yvals <= q1) = "dark";
    lightClass(yvals > q1 & yvals <= q2) = "medium";
    lightClass(yvals > q2) = "bright";
    T.lightClass = lightClass;

    writetable(T, fullfile(outDir, sprintf('%s_pixels.csv', name)));

    summaryRows(end+1,:) = {name, nnz(roiMask), nnz(adaptivePixels), nnz(usablePixels), ...
        nnz(contamPixels), nnz(combinedEyeMask & adaptivePixels), nnz(browMask & adaptivePixels), ...
        nnz(mouthMask & adaptivePixels), nnz(hairlineMask & adaptivePixels), ...
        nnz(earMask & adaptivePixels), nnz(beardMask & adaptivePixels), ...
        nnz(specularMask & adaptivePixels)}; %#ok<AGROW>
end

summaryTable = cell2table(summaryRows, 'VariableNames', ...
    {'roi','totalPixels','adaptiveSkinPixels','usablePixels','contaminatedPixels', ...
     'eyePixels','browPixels','mouthPixels','hairlinePixels','earPixels','beardPixels','specularPixels'});

writetable(summaryTable, fullfile(outDir,'roi_summary.csv'));

%% overlay image
redMask = candidateMask & ~usableMask;
greenMask = usableMask;
overlay(:,:,1) = uint8(double(overlay(:,:,1)).*(~redMask) + 255*double(redMask));
overlay(:,:,2) = uint8(double(overlay(:,:,2)).*(~greenMask) + 255*double(greenMask));
overlay(:,:,3) = uint8(double(overlay(:,:,3)).*(~(redMask | greenMask)));

imwrite(Iface_u8,    fullfile(outDir,'face_roi.png'));
imwrite(seedMask,    fullfile(outDir,'seed_mask.png'));
imwrite(adaptiveMask,fullfile(outDir,'adaptive_mask.png'));
imwrite(candidateMask,fullfile(outDir,'candidate_mask.png'));
imwrite(usableMask,  fullfile(outDir,'usable_mask.png'));
imwrite(overlay,     fullfile(outDir,'usable_overlay.png'));

save(fullfile(outDir,'roi_anchor_regiongrow_results.mat'), ...
    'bb','anchors','roi','seedMask','adaptiveMask','candidateMask','usableMask', ...
    'eyeMask','browMask','mouthMask','hairlineMask','earMask','beardMask', ...
    'specularMask','summaryTable');

%% display
figure('Name','Anchor-guided region-growing usable skin mask');
subplot(2,3,1); imshow(Id); title('Input image'); hold on;
rectangle('Position',bb,'EdgeColor','g','LineWidth',1.2);

subplot(2,3,2); imshow(Iface_u8); title('Face ROI'); hold on;
plot(anchors.leftEyeCenter(1),anchors.leftEyeCenter(2),'g+','MarkerSize',8,'LineWidth',1.5);
plot(anchors.rightEyeCenter(1),anchors.rightEyeCenter(2),'g+','MarkerSize',8,'LineWidth',1.5);
plot(anchors.noseCenter(1),anchors.noseCenter(2),'g+','MarkerSize',8,'LineWidth',1.5);
plot(anchors.mouthCenter(1),anchors.mouthCenter(2),'g+','MarkerSize',8,'LineWidth',1.5);
for k = 1:numel(roiNames)
    r = roi.(roiNames{k});
    rectangle('Position',r,'LineWidth',1.0);
end

subplot(2,3,3); imshow(adaptiveMask); title('Adaptive YCbCr mask');
subplot(2,3,4); imshow(excludeMask); title('Anchor-based exclusions');
subplot(2,3,5); imshow(usableMask); title('Usable pixels');
subplot(2,3,6); imshow(overlay); title('Green=usable, Red=filtered');

disp(summaryTable);
fprintf('Saved outputs to: %s\n', outDir);

%% -------- local functions --------
function anchors = estimateFaceAnchors(Iface_u8)
    [H,W,~] = size(Iface_u8);

    % defaults
    anchors.leftEyeCenter  = [round(0.33*W), round(0.38*H)];
    anchors.rightEyeCenter = [round(0.67*W), round(0.38*H)];
    anchors.noseCenter     = [round(0.50*W), round(0.53*H)];
    anchors.mouthCenter    = [round(0.50*W), round(0.73*H)];

    eyeRx = round(0.09*W); eyeRy = round(0.05*H);
    mouthRx = round(0.14*W); mouthRy = round(0.07*H);

    % eyes
    try
        led = vision.CascadeObjectDetector('LeftEye');
        red = vision.CascadeObjectDetector('RightEye');
        leftBoxes = step(led, Iface_u8);
        rightBoxes = step(red, Iface_u8);

        if ~isempty(leftBoxes)
            [~,idx] = max(leftBoxes(:,3).*leftBoxes(:,4));
            b = leftBoxes(idx,:);
            anchors.leftEyeCenter = [b(1)+b(3)/2, b(2)+b(4)/2];
            eyeRx = max(eyeRx, round(0.7*b(3)/2));
            eyeRy = max(eyeRy, round(0.8*b(4)/2));
        end
        if ~isempty(rightBoxes)
            [~,idx] = max(rightBoxes(:,3).*rightBoxes(:,4));
            b = rightBoxes(idx,:);
            anchors.rightEyeCenter = [b(1)+b(3)/2, b(2)+b(4)/2];
            eyeRx = max(eyeRx, round(0.7*b(3)/2));
            eyeRy = max(eyeRy, round(0.8*b(4)/2));
        end
    catch
    end

    % nose
    try
        nd = vision.CascadeObjectDetector('Nose');
        noseBoxes = step(nd, Iface_u8);
        if ~isempty(noseBoxes)
            [~,idx] = max(noseBoxes(:,3).*noseBoxes(:,4));
            b = noseBoxes(idx,:);
            anchors.noseCenter = [b(1)+b(3)/2, b(2)+b(4)/2];
        end
    catch
    end

    % mouth
    try
        md = vision.CascadeObjectDetector('Mouth');
        mouthBoxes = step(md, Iface_u8);
        if ~isempty(mouthBoxes)
            [~,idx] = max(mouthBoxes(:,3).*mouthBoxes(:,4));
            b = mouthBoxes(idx,:);
            anchors.mouthCenter = [b(1)+b(3)/2, b(2)+b(4)/2];
            mouthRx = max(mouthRx, round(0.65*b(3)/2));
            mouthRy = max(mouthRy, round(0.85*b(4)/2));
        end
    catch
    end

    anchors.eyeRx = eyeRx;
    anchors.eyeRy = eyeRy;
    anchors.mouthRx = mouthRx;
    anchors.mouthRy = mouthRy;
    anchors.foreheadTopY = round(max(1, min([anchors.leftEyeCenter(2), anchors.rightEyeCenter(2)]) - 1.9*eyeRy));
    anchors.foreheadBottomY = round(max(1, min([anchors.leftEyeCenter(2), anchors.rightEyeCenter(2)]) - 0.2*eyeRy));
end

function mask = ellipseMask(H,W,center,rx,ry)
    [X,Y] = meshgrid(1:W,1:H);
    cx = center(1); cy = center(2);
    mask = ((X-cx).^2)/(max(rx,1)^2) + ((Y-cy).^2)/(max(ry,1)^2) <= 1;
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
    x1 = r(1); y1 = r(2); x2 = min(W, x1+r(3)-1); y2 = min(H, y1+r(4)-1);
    mask(y1:y2, x1:x2) = true;
end

function [seedRow,seedCol] = chooseSeed(localCandidates, roiMask)
    seedRow = []; seedCol = [];
    [yy,xx] = find(localCandidates);
    if isempty(xx)
        return;
    end
    [H,W] = size(roiMask);
    cx = W/2; cy = H/2;
    d2 = (xx-cx).^2 + (yy-cy).^2;
    [~,idx] = min(d2);
    seedRow = yy(idx);
    seedCol = xx(idx);
end

function tol = estimateTolerance(Y, localCandidates, roiMask)
    vals = Y(localCandidates);
    if isempty(vals)
        vals = Y(roiMask);
    end
    if isempty(vals)
        tol = 8;
        return;
    end
    tol = max(6, min(18, round(0.8*std(vals) + 6)));
end

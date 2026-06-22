%% roi_adaptive_ycbcr_contamination_filter.m
clear; clc; close all;

imagePath = '/Users/devanshbajwala/Downloads/Photo on 4-5-26 at 10.31.jpg'; 
useManualFaceBox = false;
faceBox = [];                        % [x y w h]
outDir = 'roi_mask_outputs';

if ~exist(outDir,'dir'); mkdir(outDir); end

Iu8 = imread(imagePath);
if size(Iu8,3)==1, Iu8 = repmat(Iu8,[1 1 3]); end
Id = im2double(Iu8);

%% face ROI
if useManualFaceBox
    bb = round(faceBox);
else
    det = [];
    try
        fd = vision.CascadeObjectDetector();
        det = step(fd, Iu8);
    catch
    end
    if isempty(det)
        bb = [1 1 size(Iu8,2) size(Iu8,1)];
    else
        [~,idx] = max(det(:,3).*det(:,4));
        bb = det(idx,:);
    end
end

x = max(1,bb(1)); y = max(1,bb(2));
w = min(bb(3), size(Iu8,2)-x+1);
h = min(bb(4), size(Iu8,1)-y+1);
bb = [x y w h];

Iface_u8 = imcrop(Iu8,[x y w-1 h-1]);
Iface_d  = im2double(Iface_u8);
[H,W,~] = size(Iface_u8);

%% base color spaces
YCbCr = rgb2ycbcr(Iface_u8);
Y  = double(YCbCr(:,:,1));
Cb = double(YCbCr(:,:,2));
Cr = double(YCbCr(:,:,3));

R = Id(y:y+h-1, x:x+w-1, 1);
G = Id(y:y+h-1, x:x+w-1, 2);
B = Id(y:y+h-1, x:x+w-1, 3);

%% coarse seed mask
seedMask = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173) & (Y > 40);

% restrict outer boundary to suppress ears/hair/background
[Xg,Yg] = meshgrid(1:W,1:H);
xn = (Xg - W/2)/(W/2);
yn = (Yg - H/2)/(H/2);
faceEllipse = (xn/0.88).^2 + (yn/0.98).^2 <= 1;

seedMask = seedMask & faceEllipse;

%% adaptive thresholds from seed pixels
if nnz(seedMask) < 200
    adaptiveMask = seedMask;
else
    cbVals = Cb(seedMask); crVals = Cr(seedMask); yVals = Y(seedMask);
    cbMu = mean(cbVals); cbSd = std(cbVals);
    crMu = mean(crVals); crSd = std(crVals);
    yLo  = max(35, prctile(yVals,5)  - 5);
    yHi  = min(245, prctile(yVals,98) + 5);

    cbLo = max(60,  cbMu - 1.6*cbSd); cbHi = min(140, cbMu + 1.6*cbSd);
    crLo = max(120, crMu - 1.6*crSd); crHi = min(180, crMu + 1.6*crSd);

    adaptiveMask = (Cb >= cbLo & Cb <= cbHi) & ...
                   (Cr >= crLo & Cr <= crHi) & ...
                   (Y  >= yLo  & Y  <= yHi) & faceEllipse;
end

%% contamination heuristics
% dark hair / beard / eyebrows
darkMask = Y < max(55, prctile(Y(adaptiveMask | seedMask),10));

% specular / saturated pixels
satMask = (Y > 245) | (R > 0.98) | (G > 0.98) | (B > 0.98);

% lip-like pixels: higher Cr and lower Y in lower middle face
lowerFace = Yg > round(0.52*H);
mouthBand = lowerFace & abs(Xg - W/2) < 0.20*W;
lipMask = (Cr > max(150, prctile(Cr(seedMask | adaptiveMask),75))) & mouthBand;

% eye / eyelid / eyebrow region: dark and upper-mid face
eyeBand = Yg > round(0.22*H) & Yg < round(0.52*H);
eyeWindow = eyeBand & abs(Xg - W/2) < 0.42*W;
eyeMask = darkMask & eyeWindow;

% forehead hairline: dark pixels in top forehead band
topBand = Yg < round(0.22*H);
hairlineMask = darkMask & topBand;

% ears: far left/right bands
earMask = abs(Xg - W/2) > 0.42*W;

% coarse nonskin color residuals
nonskinColorMask = ~adaptiveMask;

% final usable skin
contamMask = darkMask | satMask | lipMask | eyeMask | hairlineMask | earMask | nonskinColorMask;
usableMask = adaptiveMask & ~contamMask;

%% ROI definitions
roi.forehead    = round([0.24*W, 0.08*H, 0.52*W, 0.20*H]);
roi.left_cheek  = round([0.10*W, 0.40*H, 0.28*W, 0.25*H]);
roi.right_cheek = round([0.62*W, 0.40*H, 0.28*W, 0.25*H]);
roi.nose        = round([0.41*W, 0.32*H, 0.18*W, 0.28*H]);
roi.full_face   = [1 1 W H];
roiNames = fieldnames(roi);

%% per-pixel tables and stats
summaryRows = {};
usableOverlay = Iface_u8;

for k = 1:numel(roiNames)
    name = roiNames{k};
    r = roi.(name);
    r(1) = max(1,r(1)); r(2) = max(1,r(2));
    r(3) = min(r(3),W-r(1)+1); r(4) = min(r(4),H-r(2)+1);

    x1 = r(1); y1 = r(2); x2 = x1 + r(3) - 1; y2 = y1 + r(4) - 1;
    roiMask = false(H,W);
    roiMask(y1:y2, x1:x2) = true;

    totalPixels = nnz(roiMask);
    roiAdaptive = adaptiveMask & roiMask;
    roiUsable   = usableMask & roiMask;

    hairContam  = (darkMask | hairlineMask) & roiMask;
    lipContam   = lipMask & roiMask;
    eyeContam   = eyeMask & roiMask;
    earContam   = earMask & roiMask;
    satContam   = satMask & roiMask;
    nonskinCont = nonskinColorMask & roiMask;

    [yy,xx] = find(roiMask);
    lin = sub2ind([H W],yy,xx);

    T = table;
    T.x = xx;
    T.y = yy;
    T.R = reshape(R(lin),[],1);
    T.G = reshape(G(lin),[],1);
    T.B = reshape(B(lin),[],1);
    T.Y = reshape(Y(lin),[],1);
    T.Cb = reshape(Cb(lin),[],1);
    T.Cr = reshape(Cr(lin),[],1);
    T.isAdaptiveSkin = reshape(roiAdaptive(lin),[],1);
    T.isUsable = reshape(roiUsable(lin),[],1);
    T.isHairLike = reshape(hairContam(lin),[],1);
    T.isLipLike = reshape(lipContam(lin),[],1);
    T.isEyeLike = reshape(eyeContam(lin),[],1);
    T.isEarLike = reshape(earContam(lin),[],1);
    T.isSaturated = reshape(satContam(lin),[],1);
    T.isNonSkinColor = reshape(nonskinCont(lin),[],1);

    % lighting / reflectance classes preserved for later study
    yVals = T.Y;
    q1 = prctile(yVals,33); q2 = prctile(yVals,66);
    lightClass = strings(height(T),1);
    lightClass(yVals <= q1) = "dark";
    lightClass(yVals > q1 & yVals <= q2) = "medium";
    lightClass(yVals > q2) = "bright";
    T.lightClass = lightClass;

    writetable(T, fullfile(outDir, sprintf('%s_pixels.csv',name)));

    usableCount = nnz(roiUsable);
    adaptiveCount = nnz(roiAdaptive);
    contamCount = adaptiveCount - usableCount;

    summaryRows(end+1,:) = {name,totalPixels,adaptiveCount,usableCount,contamCount, ...
        nnz(hairContam & roiAdaptive), nnz(lipContam & roiAdaptive), ...
        nnz(eyeContam & roiAdaptive), nnz(earContam & roiAdaptive), ...
        nnz(satContam & roiAdaptive), nnz(nonskinCont & roiMask)}; %#ok<AGROW>

    % overlay usable pixels in green, contaminated adaptive-skin pixels in red
    roiContamAdaptive = roiAdaptive & ~roiUsable;
    usableOverlay(:,:,1) = uint8(double(usableOverlay(:,:,1)).*(~roiContamAdaptive) + 255*double(roiContamAdaptive));
    usableOverlay(:,:,2) = uint8(double(usableOverlay(:,:,2)).*(~roiUsable) + 255*double(roiUsable));
    usableOverlay(:,:,3) = uint8(double(usableOverlay(:,:,3)).*(~(roiUsable|roiContamAdaptive)) + 0*double(roiUsable|roiContamAdaptive));
end

summaryTable = cell2table(summaryRows, 'VariableNames', ...
    {'roi','totalPixels','adaptiveSkinPixels','usablePixels','contaminatedPixels', ...
     'hairLikePixels','lipLikePixels','eyeLikePixels','earLikePixels', ...
     'saturatedPixels','nonSkinColorPixels'});
writetable(summaryTable, fullfile(outDir,'roi_summary.csv'));

save(fullfile(outDir,'roi_masks.mat'), 'bb','seedMask','adaptiveMask','usableMask', ...
     'darkMask','satMask','lipMask','eyeMask','hairlineMask','earMask', ...
     'faceEllipse','summaryTable','roi');

imwrite(Iface_u8, fullfile(outDir,'face_roi.png'));
imwrite(seedMask, fullfile(outDir,'seed_mask.png'));
imwrite(adaptiveMask, fullfile(outDir,'adaptive_mask.png'));
imwrite(usableMask, fullfile(outDir,'usable_mask.png'));
imwrite(usableOverlay, fullfile(outDir,'usable_overlay.png'));

%% display
figure('Name','Adaptive YCbCr contamination filtering');
subplot(2,3,1); imshow(Id); title('Input image'); hold on;
rectangle('Position',bb,'EdgeColor','g','LineWidth',1.3);

subplot(2,3,2); imshow(Iface_u8); title('Face ROI'); hold on;
for k = 1:numel(roiNames)
    r = roi.(roiNames{k});
    rectangle('Position',[r(1) r(2) r(3) r(4)],'LineWidth',1.1);
end

subplot(2,3,3); imshow(seedMask); title('Initial YCbCr seed');
subplot(2,3,4); imshow(adaptiveMask); title('Adaptive skin mask');
subplot(2,3,5); imshow(usableMask); title('Usable skin pixels');
subplot(2,3,6); imshow(usableOverlay); title('Green=usable, Red=filtered');

disp(summaryTable);
fprintf('Saved outputs to folder: %s\n', outDir);

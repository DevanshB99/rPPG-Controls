%% rppg_stage1B_optical_measurement.m
clear; clc; close all;

%% user input
imagePath = '/Users/devanshbajwala/Downloads/Photo on 4-5-26 at 10.31.jpg';   % change to your image path
useManualFaceBox = false;            % true if you want to set faceBox manually
faceBox = [];                        % [x y w h] if useManualFaceBox = true

%% plant parameters
fHR_bpm = 72;
fHR_hz = fHR_bpm/60;
omega_p = 2*pi*fHR_hz;
zeta_p = 0.12;
Kw = 1.0;

fs = 30;
Ts = 1/fs;
T  = 10;
t  = 0:Ts:T-Ts;

%% load image
I = imread(imagePath);
if size(I,3) == 1
    I = repmat(I,[1 1 3]);
end
Irgb = im2double(I);

%% face ROI
if useManualFaceBox
    assert(numel(faceBox)==4,'faceBox must be [x y w h].');
    bb = round(faceBox);
else
    det = [];
    try
        fd = vision.CascadeObjectDetector();
        det = step(fd, I);
    catch
    end
    if isempty(det)
        bb = [1 1 size(I,2) size(I,1)];
    else
        [~,idx] = max(det(:,3).*det(:,4));
        bb = det(idx,:);
    end
end

x = bb(1); y = bb(2); w = bb(3); h = bb(4);
x = max(1,x); y = max(1,y);
w = min(w,size(I,2)-x+1); h = min(h,size(I,1)-y+1);

Iface = imcrop(Irgb,[x y w-1 h-1]);

%% YCbCr skin mask
YCbCr = rgb2ycbcr(Iface);
Y  = double(YCbCr(:,:,1));
Cb = double(YCbCr(:,:,2));
Cr = double(YCbCr(:,:,3));

skinMask = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173) & (Y > 40);

%% geometric sub-ROIs inside face
[H,W,~] = size(Iface);

roi.forehead    = round([0.25*W, 0.08*H, 0.50*W, 0.22*H]);
roi.left_cheek  = round([0.12*W, 0.42*H, 0.26*W, 0.24*H]);
roi.right_cheek = round([0.62*W, 0.42*H, 0.26*W, 0.24*H]);
roi.nose        = round([0.40*W, 0.34*H, 0.20*W, 0.28*H]);
roi.full_face   = [1 1 W H];

roiNames = fieldnames(roi);

%% mean extraction helper
stats = struct();
for k = 1:numel(roiNames)
    name = roiNames{k};
    r = roi.(name);
    r(1) = max(1,r(1)); r(2) = max(1,r(2));
    r(3) = min(r(3),W-r(1)+1); r(4) = min(r(4),H-r(2)+1);

    patch = imcrop(Iface,[r(1) r(2) r(3)-1 r(4)-1]);
    patchMask = imcrop(skinMask,[r(1) r(2) r(3)-1 r(4)-1]);

    R = patch(:,:,1);
    G = patch(:,:,2);
    B = patch(:,:,3);

    idx = patchMask > 0;
    if any(idx(:))
        stats.(name).meanRGB = [mean(R(idx)); mean(G(idx)); mean(B(idx))];
        stats.(name).meanY   = mean(double(rgb2ycbcr(patch(:,:,1:3))));
        stats.(name).skinPixels = nnz(idx);
    else
        stats.(name).meanRGB = [mean(R(:)); mean(G(:)); mean(B(:))];
        stats.(name).meanY   = mean(Y(:));
        stats.(name).skinPixels = 0;
    end
    stats.(name).totalPixels = numel(idx);
    stats.(name).skinRatio = stats.(name).skinPixels / stats.(name).totalPixels;
    stats.(name).box = r;
end

%% operating point c0 from full-face mean
c0 = stats.full_face.meanRGB;

%% latent physiological plant
Gp = tf(Kw,[1 2*zeta_p*omega_p omega_p^2]);

u = sin(2*pi*fHR_hz*t);
xp = lsim(Gp,u,t);            % latent physiological state

%% optical measurement equation
% c(t) = c0 + H_p*x_p(t) + H_m*x_m(t) + H_l*x_l(t) + n(t)
%
% H_p, H_m, H_l are 3x1 sensitivity vectors in RGB space.

Hp = [ 0.012; -0.020; -0.006] .* c0;   % physiological sensitivity
Hm = [ 0.015;  0.015;  0.015] .* c0;   % motion/specular-like brightness direction
Hl = [ 0.010;  0.008;  0.006] .* c0;   % illumination/exposure drift direction

xm = 0.20*sin(2*pi*0.35*t + 0.7);      % motion disturbance
xl = 0.30*sin(2*pi*0.08*t);            % slow illumination drift

rng(1);
nR = 0.002*randn(size(t));
nG = 0.002*randn(size(t));
nB = 0.002*randn(size(t));
N  = [nR; nG; nB] .* c0;

C = c0 + Hp*xp.' + Hm*xm + Hl*xl + N;

Rbar = C(1,:);
Gbar = C(2,:);
Bbar = C(3,:);

Rhat = Rbar/mean(Rbar);
Ghat = Gbar/mean(Gbar);
Bhat = Bbar/mean(Bbar);

Xs = 3*Rhat - 2*Ghat;
Ys = 1.5*Rhat + Ghat - 1.5*Bhat;
alpha = std(Xs)/std(Ys);
S = Xs - alpha*Ys;

%% plots
figure('Name','Stage 1B: static image + optical measurement model');
subplot(2,3,1);
imshow(Irgb); title('Input image'); hold on;
rectangle('Position',bb,'EdgeColor','g','LineWidth',1.5);

subplot(2,3,2);
imshow(Iface); title('Face ROI'); hold on;
for k = 1:numel(roiNames)
    r = stats.(roiNames{k}).box;
    rectangle('Position',[r(1) r(2) r(3) r(4)],'LineWidth',1.2);
end

subplot(2,3,3);
imshow(skinMask); title('YCbCr skin mask');

subplot(2,3,4);
plot(t,xp,'LineWidth',1.3); grid on;
xlabel('t [s]'); ylabel('x_p(t)');
title('Latent physiological state');

subplot(2,3,5);
plot(t,Rhat,'LineWidth',1.1); hold on;
plot(t,Ghat,'LineWidth',1.1);
plot(t,Bhat,'LineWidth',1.1);
grid on;
xlabel('t [s]'); ylabel('normalized RGB');
legend('R','G','B','Location','best');
title('Synthetic measured RGB traces');

subplot(2,3,6);
plot(t,S,'LineWidth',1.4); grid on;
xlabel('t [s]'); ylabel('S(t)');
title('CHROM output');

%% console output
fprintf('\n=== Stage 1B optical measurement model ===\n');
fprintf('Face box [x y w h] = [%d %d %d %d]\n',bb(1),bb(2),bb(3),bb(4));
fprintf('\nOperating point c0 (full-face mean RGB):\n');
disp(c0);

for k = 1:numel(roiNames)
    name = roiNames{k};
    fprintf('%s meanRGB = [%.4f %.4f %.4f]^T, skinRatio = %.4f\n', ...
        name, stats.(name).meanRGB(1), stats.(name).meanRGB(2), ...
        stats.(name).meanRGB(3), stats.(name).skinRatio);
end

fprintf('\nOptical measurement equation used:\n');
fprintf('c(t) = c0 + H_p*x_p(t) + H_m*x_m(t) + H_l*x_l(t) + n(t)\n');

fprintf('\nSensitivity vectors:\n');
fprintf('H_p = [%.6f %.6f %.6f]^T\n',Hp(1),Hp(2),Hp(3));
fprintf('H_m = [%.6f %.6f %.6f]^T\n',Hm(1),Hm(2),Hm(3));
fprintf('H_l = [%.6f %.6f %.6f]^T\n',Hl(1),Hl(2),Hl(3));

fprintf('\nCHROM alpha = %.6f\n',alpha);

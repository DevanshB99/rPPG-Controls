%% rppg_stage1B_optical_measurement_v2.m
clear; clc; close all;

imagePath = '/Users/devanshbajwala/Downloads/Photo on 4-5-26 at 10.31.jpg'; 
useManualFaceBox = false;
faceBox = [];

fHR_bpm = 72;
fHR_hz  = fHR_bpm/60;
omega_p = 2*pi*fHR_hz;
zeta_p  = 0.12;
Kw      = 1.0;

fs = 30; Ts = 1/fs; T = 10;
t = 0:Ts:T-Ts;

I = imread(imagePath);
if size(I,3) == 1, I = repmat(I,[1 1 3]); end
Iu8 = I;
Id  = im2double(Iu8);

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

% Use uint8 YCbCr because thresholds are in 0..255 units
YCbCr_u8 = rgb2ycbcr(Iface_u8);
Y  = YCbCr_u8(:,:,1);
Cb = YCbCr_u8(:,:,2);
Cr = YCbCr_u8(:,:,3);

skinMask = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173) & (Y > 40);

[H,W,~] = size(Iface_d);
roi.forehead    = round([0.25*W, 0.08*H, 0.50*W, 0.22*H]);
roi.left_cheek  = round([0.12*W, 0.42*H, 0.26*W, 0.24*H]);
roi.right_cheek = round([0.62*W, 0.42*H, 0.26*W, 0.24*H]);
roi.nose        = round([0.40*W, 0.34*H, 0.20*W, 0.28*H]);
roi.full_face   = [1 1 W H];

roiNames = fieldnames(roi);
stats = struct();

for k = 1:numel(roiNames)
    name = roiNames{k};
    r = roi.(name);
    r(1) = max(1,r(1)); r(2) = max(1,r(2));
    r(3) = min(r(3),W-r(1)+1); r(4) = min(r(4),H-r(2)+1);

    patch_d    = imcrop(Iface_d, [r(1) r(2) r(3)-1 r(4)-1]);
    patchMask  = imcrop(skinMask,[r(1) r(2) r(3)-1 r(4)-1]);
    patchYCbCr = imcrop(YCbCr_u8,[r(1) r(2) r(3)-1 r(4)-1]);

    R = patch_d(:,:,1); G = patch_d(:,:,2); B = patch_d(:,:,3);
    idx = patchMask > 0;

    if any(idx(:))
        meanRGB = [mean(R(idx)); mean(G(idx)); mean(B(idx))];
        meanY   = mean(double(patchYCbCr(:,:,1)).*double(idx),'all') / mean(double(idx(:)));
    else
        meanRGB = [mean(R(:)); mean(G(:)); mean(B(:))];
        meanY   = mean(double(patchYCbCr(:,:,1)),'all');
    end

    stats.(name).meanRGB    = meanRGB;
    stats.(name).meanY      = meanY;
    stats.(name).skinPixels = nnz(idx);
    stats.(name).totalPixels = numel(idx);
    stats.(name).skinRatio  = nnz(idx)/numel(idx);
    stats.(name).box        = r;
end

% c0 should come from skin pixels only when available
c0 = stats.full_face.meanRGB;

Gp = tf(Kw,[1 2*zeta_p*omega_p omega_p^2]);
u  = sin(2*pi*fHR_hz*t);
xp = lsim(Gp,u,t);

Hp = [ 0.012; -0.020; -0.006] .* c0;
Hm = [ 0.015;  0.015;  0.015] .* c0;
Hl = [ 0.010;  0.008;  0.006] .* c0;

xm = 0.20*sin(2*pi*0.35*t + 0.7);
xl = 0.30*sin(2*pi*0.08*t);

rng(1);
N = [0.002*randn(size(t)); 0.002*randn(size(t)); 0.002*randn(size(t))] .* c0;

C = c0 + Hp*xp.' + Hm*xm + Hl*xl + N;
Rbar = C(1,:); Gbar = C(2,:); Bbar = C(3,:);

Rhat = Rbar/mean(Rbar);
Ghat = Gbar/mean(Gbar);
Bhat = Bbar/mean(Bbar);

Xs = 3*Rhat - 2*Ghat;
Ys = 1.5*Rhat + Ghat - 1.5*Bhat;
alpha = std(Xs)/std(Ys);
S = Xs - alpha*Ys;
S = S - mean(S);

figure('Name','Stage 1B corrected');
subplot(2,3,1); imshow(Id); title('Input image'); hold on;
rectangle('Position',bb,'EdgeColor','g','LineWidth',1.5);

subplot(2,3,2); imshow(Iface_d); title('Face ROI'); hold on;
for k = 1:numel(roiNames)
    r = stats.(roiNames{k}).box;
    rectangle('Position',[r(1) r(2) r(3) r(4)],'LineWidth',1.2);
end

subplot(2,3,3); imshow(skinMask); title('YCbCr skin mask');

subplot(2,3,4); plot(t,xp,'LineWidth',1.2); grid on;
xlabel('t [s]'); ylabel('x_p(t)'); title('Latent physiological state');

subplot(2,3,5);
plot(t,Rhat,'LineWidth',1.0); hold on;
plot(t,Ghat,'LineWidth',1.0);
plot(t,Bhat,'LineWidth',1.0);
grid on; xlabel('t [s]'); ylabel('normalized RGB');
legend('R','G','B','Location','best'); title('Synthetic measured RGB');

subplot(2,3,6); plot(t,S,'LineWidth',1.2); grid on;
xlabel('t [s]'); ylabel('S(t)'); title('CHROM output (zero-mean)');

fprintf('Face box [x y w h] = [%d %d %d %d]\n',bb(1),bb(2),bb(3),bb(4));
fprintf('c0 = [%.4f %.4f %.4f]^T\n',c0(1),c0(2),c0(3));
for k = 1:numel(roiNames)
    name = roiNames{k};
    fprintf('%s: meanRGB=[%.4f %.4f %.4f]^T, meanY=%.2f, skinRatio=%.4f\n', ...
        name, stats.(name).meanRGB(1), stats.(name).meanRGB(2), stats.(name).meanRGB(3), ...
        stats.(name).meanY, stats.(name).skinRatio);
end
fprintf('alpha = %.6f\n',alpha);

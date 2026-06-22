%% rppg_step1D_multi_roi_plant.m
clear; clc; close all;

%% time / latent plant
fHR_bpm = 72;
fHR_hz  = fHR_bpm/60;
omega_p = 2*pi*fHR_hz;
zeta_p  = 0.12;
Kw      = 1.0;

fs = 30; Ts = 1/fs; T = 20;
t = 0:Ts:T-Ts;
N = numel(t);

Gp = tf(Kw,[1 2*zeta_p*omega_p omega_p^2]);
u_p = sin(2*pi*fHR_hz*t);
x_p = lsim(Gp,u_p,t).';                       % shared latent physiological state

%% ROIs
roiNames = {'forehead','left_cheek','right_cheek','nose'};
nROI = numel(roiNames);

% operating points from your current image-driven result
c0.forehead    = [0.4714; 0.3766; 0.3273];
c0.left_cheek  = [0.7125; 0.6092; 0.5814];
c0.right_cheek = [0.4350; 0.3286; 0.2732];
c0.nose        = [0.5897; 0.4764; 0.4324];

% usable-pixel counts from your current ROI adaptive result
usablePix.forehead    = 20580;
usablePix.left_cheek  = 8797;
usablePix.right_cheek = 13480;
usablePix.nose        = 7009;

% contamination ratios from your current ROI adaptive result
contamRatio.forehead    = 378/20958;
contamRatio.left_cheek  = 20/8817;
contamRatio.right_cheek = 466/13946;
contamRatio.nose        = 296/7305;

%% ROI-specific measurement model
% c_i(t) = c0_i + h_p_i*x_p(t) + h_m_i*x_m_i(t) + h_l_i*x_l_i(t) + n_i(t)

% shared CHROM projection
L = [3 -2 0;
     1.5 1 -1.5];

results = struct();
rng(1);

for k = 1:nROI
    roi = roiNames{k};
    c0i = c0.(roi);

    % ROI-specific sensitivity vectors
    % physiology strongest in forehead and cheeks, slightly weaker in nose
    switch roi
        case 'forehead'
            hp = [ 0.014; -0.021; -0.007] .* c0i;
            hm = [ 0.010;  0.010;  0.010] .* c0i;
            hl = [ 0.009;  0.007;  0.006] .* c0i;
            xm = 0.14*sin(2*pi*0.30*t + 0.4);
            xl = 0.22*sin(2*pi*0.08*t + 0.0);
            noiseScale = 0.0018;
        case 'left_cheek'
            hp = [ 0.012; -0.019; -0.006] .* c0i;
            hm = [ 0.016;  0.016;  0.016] .* c0i;
            hl = [ 0.012;  0.010;  0.008] .* c0i;
            xm = 0.22*sin(2*pi*0.36*t + 0.9);
            xl = 0.30*sin(2*pi*0.08*t + 0.2);
            noiseScale = 0.0025;
        case 'right_cheek'
            hp = [ 0.013; -0.020; -0.006] .* c0i;
            hm = [ 0.018;  0.018;  0.018] .* c0i;
            hl = [ 0.013;  0.011;  0.009] .* c0i;
            xm = 0.20*sin(2*pi*0.38*t + 0.7);
            xl = 0.34*sin(2*pi*0.08*t + 0.5);
            noiseScale = 0.0023;
        case 'nose'
            hp = [ 0.010; -0.016; -0.005] .* c0i;
            hm = [ 0.020;  0.020;  0.020] .* c0i;
            hl = [ 0.014;  0.012;  0.010] .* c0i;
            xm = 0.24*sin(2*pi*0.42*t + 0.5);
            xl = 0.32*sin(2*pi*0.08*t + 0.1);
            noiseScale = 0.0028;
    end

    eta = noiseScale * randn(3,N) .* c0i;

    C = c0i + hp*x_p + hm*xm + hl*xl + eta;

    R = C(1,:); G = C(2,:); B = C(3,:);
    chat = [R/mean(R); G/mean(G); B/mean(B)];

    Z = L*chat;
    Xs = Z(1,:);
    Ys = Z(2,:);
    alpha = std(Xs)/std(Ys);

    w = [1 -alpha];
    S = w*Z;
    S = S - mean(S);

    % separated projected components
    Cp = hp*x_p;
    Cm = hm*xm;
    Cl = hl*xl;

    Cp_hat = [Cp(1,:)/mean(R); Cp(2,:)/mean(G); Cp(3,:)/mean(B)];
    Cm_hat = [Cm(1,:)/mean(R); Cm(2,:)/mean(G); Cm(3,:)/mean(B)];
    Cl_hat = [Cl(1,:)/mean(R); Cl(2,:)/mean(G); Cl(3,:)/mean(B)];

    Sp = w*(L*Cp_hat);
    Sm = w*(L*Cm_hat);
    Sl = w*(L*Cl_hat);

    gain_p = rms(Sp)/rms(x_p);
    motion_ratio = rms(Sm)/rms(xm);
    light_ratio  = rms(Sl)/rms(xl);

    % simple ROI confidence
    usableNorm = usablePix.(roi) / max(struct2array(usablePix));
    contamPenalty = 1 - contamRatio.(roi);
    projScore = gain_p / (motion_ratio + light_ratio + eps);
    quality = usableNorm * contamPenalty * projScore;

    results.(roi).c0 = c0i;
    results.(roi).hp = hp;
    results.(roi).hm = hm;
    results.(roi).hl = hl;
    results.(roi).R = R;
    results.(roi).G = G;
    results.(roi).B = B;
    results.(roi).chat = chat;
    results.(roi).Xs = Xs;
    results.(roi).Ys = Ys;
    results.(roi).alpha = alpha;
    results.(roi).S = S;
    results.(roi).Sp = Sp;
    results.(roi).Sm = Sm;
    results.(roi).Sl = Sl;
    results.(roi).gain_p = gain_p;
    results.(roi).motion_ratio = motion_ratio;
    results.(roi).light_ratio = light_ratio;
    results.(roi).usableNorm = usableNorm;
    results.(roi).contamPenalty = contamPenalty;
    results.(roi).quality = quality;
end

%% fusion weights
q = zeros(1,nROI);
for k = 1:nROI
    q(k) = results.(roiNames{k}).quality;
end
wFusion = q / sum(q);

Sfused = zeros(1,N);
for k = 1:nROI
    Sfused = Sfused + wFusion(k) * results.(roiNames{k}).S;
end
Sfused = Sfused - mean(Sfused);

%% summary table
roiCol = strings(nROI,1);
usableCol = zeros(nROI,1);
contamCol = zeros(nROI,1);
alphaCol = zeros(nROI,1);
gainCol = zeros(nROI,1);
motionCol = zeros(nROI,1);
lightCol = zeros(nROI,1);
qualityCol = zeros(nROI,1);
weightCol = zeros(nROI,1);

for k = 1:nROI
    roi = roiNames{k};
    roiCol(k) = roi;
    usableCol(k) = usablePix.(roi);
    contamCol(k) = contamRatio.(roi);
    alphaCol(k) = results.(roi).alpha;
    gainCol(k) = results.(roi).gain_p;
    motionCol(k) = results.(roi).motion_ratio;
    lightCol(k) = results.(roi).light_ratio;
    qualityCol(k) = results.(roi).quality;
    weightCol(k) = wFusion(k);
end

summaryTable = table(roiCol, usableCol, contamCol, alphaCol, gainCol, motionCol, ...
    lightCol, qualityCol, weightCol, ...
    'VariableNames', {'roi','usablePixels','contamRatio','alpha','gain_p', ...
    'motion_ratio','light_ratio','quality','fusionWeight'});

disp(summaryTable);

%% plots
figure('Name','Step 1D: multi-ROI optical plant');

subplot(3,2,1);
plot(t,x_p,'LineWidth',1.2); grid on;
xlabel('t [s]'); ylabel('x_p(t)');
title('Shared latent physiological state');

subplot(3,2,2);
bar(categorical(roiNames), wFusion);
ylabel('weight'); title('Fusion weights'); grid on;

subplot(3,2,3);
plot(t, results.forehead.S,'LineWidth',1.0); hold on;
plot(t, results.left_cheek.S,'LineWidth',1.0);
plot(t, results.right_cheek.S,'LineWidth',1.0);
plot(t, results.nose.S,'LineWidth',1.0);
grid on;
xlabel('t [s]'); ylabel('S_i(t)');
title('ROI-wise CHROM outputs');
legend('forehead','left cheek','right cheek','nose','Location','best');

subplot(3,2,4);
plot(t, Sfused,'k','LineWidth',1.3); grid on;
xlabel('t [s]'); ylabel('S_{fused}(t)');
title('Fused CHROM output');

subplot(3,2,5);
bar(categorical(roiNames), [gainCol motionCol lightCol], 'grouped');
ylabel('RMS ratio');
legend('gain_p','motion','light','Location','best');
title('ROI projection metrics'); grid on;

subplot(3,2,6);
imagesc(L); axis image; colorbar;
xlabel('RGB channel'); ylabel('Chrominance channel');
set(gca,'XTick',1:3,'XTickLabel',{'R','G','B'});
title('Shared CHROM matrix L');

%% save
save('step1D_multi_roi_results.mat','results','summaryTable','wFusion','Sfused','t','x_p','L');

fprintf('Saved: step1D_multi_roi_results.mat\n');

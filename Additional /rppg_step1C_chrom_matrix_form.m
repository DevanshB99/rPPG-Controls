%% rppg_step1C_chrom_matrix_form.m
clear; clc; close all;

%% parameters
fHR_bpm = 72;
fHR_hz  = fHR_bpm/60;
omega_p = 2*pi*fHR_hz;
zeta_p  = 0.12;
Kw      = 1.0;

fs = 30; Ts = 1/fs; T = 20;
t = 0:Ts:T-Ts;

% operating point
c0 = [0.4730; 0.3837; 0.3444];

%% latent plant
Gp = tf(Kw,[1 2*zeta_p*omega_p omega_p^2]);
up = sin(2*pi*fHR_hz*t);
xp = lsim(Gp,up,t).';

%% disturbance states
xm = 0.25*sin(2*pi*0.35*t + 0.7);      % motion / geometry disturbance
xl = 0.35*sin(2*pi*0.08*t);            % illumination drift

rng(1);
eta = 0.002*randn(3,numel(t));

%% measurement model
% c(t) = c0 + h_p*x_p(t) + h_m*x_m(t) + h_l*x_l(t) + n(t)
hp = [ 0.012; -0.020; -0.006] .* c0;
hm = [ 0.015;  0.015;  0.015] .* c0;   % brightness-like direction
hl = [ 0.010;  0.008;  0.006] .* c0;

C = c0 + hp*xp + hm*xm + hl*xl + eta.*c0;

R = C(1,:); G = C(2,:); B = C(3,:);
chat = [R/mean(R); G/mean(G); B/mean(B)];

%% CHROM in matrix form
L = [3 -2 0;
     1.5 1 -1.5];

Z = L * chat;
Xs = Z(1,:);
Ys = Z(2,:);
alpha = std(Xs)/std(Ys);

w = [1 -alpha];
S = w * Z;
S = S - mean(S);

%% separate projections for interpretation
Cp = hp*xp;
Cm = hm*xm;
Cl = hl*xl;

Cp_hat = [Cp(1,:)/mean(R); Cp(2,:)/mean(G); Cp(3,:)/mean(B)];
Cm_hat = [Cm(1,:)/mean(R); Cm(2,:)/mean(G); Cm(3,:)/mean(B)];
Cl_hat = [Cl(1,:)/mean(R); Cl(2,:)/mean(G); Cl(3,:)/mean(B)];

Sp = w * (L * Cp_hat);
Sm = w * (L * Cm_hat);
Sl = w * (L * Cl_hat);

%% metrics
gain_p = rms(Sp) / rms(xp);
supp_motion = rms(Sm) / rms(xm);
supp_light  = rms(Sl) / rms(xl);

%% plots
figure('Name','Step 1C: CHROM matrix form');

subplot(3,2,1);
plot(t,xp,'LineWidth',1.2); grid on;
xlabel('t [s]'); ylabel('x_p(t)');
title('Latent physiological state');

subplot(3,2,2);
plot(t,R,'LineWidth',1.0); hold on;
plot(t,G,'LineWidth',1.0);
plot(t,B,'LineWidth',1.0);
grid on; xlabel('t [s]'); ylabel('RGB');
legend('R','G','B','Location','best');
title('Measured RGB');

subplot(3,2,3);
plot(t,Xs,'LineWidth',1.0); hold on;
plot(t,Ys,'LineWidth',1.0);
grid on; xlabel('t [s]'); ylabel('projection');
legend('X_s','Y_s','Location','best');
title('CHROM chrominance channels');

subplot(3,2,4);
plot(t,S,'LineWidth',1.2); grid on;
xlabel('t [s]'); ylabel('S(t)');
title('CHROM scalar output');

subplot(3,2,5);
bar([rms(Sp) rms(Sm) rms(Sl)]);
set(gca,'XTickLabel',{'phys','motion','light'});
ylabel('RMS after CHROM');
title('Projected component strengths');

subplot(3,2,6);
imagesc(L); colorbar;
axis image;
title('Projection matrix L');
xlabel('RGB channel'); ylabel('chrominance channel');
set(gca,'XTick',1:3,'XTickLabel',{'R','G','B'});

%% save compact outputs
results = struct();
results.c0 = c0;
results.hp = hp;
results.hm = hm;
results.hl = hl;
results.L = L;
results.alpha = alpha;
results.w = w;
results.gain_p = gain_p;
results.supp_motion = supp_motion;
results.supp_light = supp_light;
results.time = t;
results.xp = xp;
results.S = S;
save('step1C_results.mat','results');

fprintf('c(t) = c0 + h_p x_p(t) + h_m x_m(t) + h_l x_l(t) + n(t)\n');
fprintf('L = [3 -2 0; 1.5 1 -1.5]\n');
fprintf('alpha = %.6f\n', alpha);
fprintf('gain_p       = %.6e\n', gain_p);
fprintf('motion_ratio = %.6e\n', supp_motion);
fprintf('light_ratio  = %.6e\n', supp_light);

7%% rppg_minimal_plant.m
% Minimal latent physiological plant for rPPG
%
% This script builds the FIRST plant only:
%   latent cardiovascular excitation -> latent pulsatile skin state
%
% Model:
%   x_p''(t) + 2*zeta_p*omega_p*x_p'(t) + omega_p^2*x_p(t) = K_w*w_p(t)
%
% Output:
%   - symbolic differential equation
%   - symbolic transfer function G_p(s) = X_p(s)/W_p(s)
%   - state-space model
%   - numerical example
%   - discrete-time model at camera frame rate
%
% Next step later:
%   connect x_p(t) to optical measurement and RGB/CHROM pipeline

clear; clc;

fprintf('===============================================================\n');
fprintf('rPPG Minimal Latent Physiological Plant\n');
fprintf('===============================================================\n\n');

%% 1) Symbolic model
fprintf('1) SYMBOLIC MODEL\n');
fprintf('-----------------\n');

syms t s zeta_p omega_p K_w real positive
syms x_p(t) w_p(t)

ode = diff(x_p,t,2) + 2*zeta_p*omega_p*diff(x_p,t) + omega_p^2*x_p == K_w*w_p;
disp('Differential equation:')
pretty(ode)
fprintf('\n');

Gp_sym = K_w / (s^2 + 2*zeta_p*omega_p*s + omega_p^2);
disp('Transfer function G_p(s) = X_p(s)/W_p(s):')
pretty(Gp_sym)
fprintf('\n');

A_sym = [0 1; -omega_p^2 -2*zeta_p*omega_p];
B_sym = [0; K_w];
C_sym = [1 0];
D_sym = 0;

disp('State-space form:')
disp('A = '); disp(A_sym);
disp('B = '); disp(B_sym);
disp('C = '); disp(C_sym);
disp('D = '); disp(D_sym);
fprintf('\n');

%% 2) Numerical hypothesis parameters
fprintf('2) NUMERICAL HYPOTHESIS PARAMETERS\n');
fprintf('----------------------------------\n');

f_HR_bpm = 72;              % nominal HR hypothesis [bpm]
f_HR_hz  = f_HR_bpm/60;     % [Hz]
omega_p_num = 2*pi*f_HR_hz; % [rad/s]
zeta_p_num = 0.12;          % damping ratio hypothesis
K_w_num    = 1.0;           % input gain hypothesis

fs = 30;                    % camera frame rate [Hz]
Ts = 1/fs;                  % sampling period [s]

fprintf('Nominal heart rate   f_HR   = %.3f bpm\n', f_HR_bpm);
fprintf('Nominal frequency    f_HR   = %.3f Hz\n', f_HR_hz);
fprintf('Natural frequency    omega_p= %.6f rad/s\n', omega_p_num);
fprintf('Damping ratio        zeta_p = %.6f\n', zeta_p_num);
fprintf('Input gain           K_w    = %.6f\n', K_w_num);
fprintf('Sampling frequency   fs     = %.3f Hz\n', fs);
fprintf('Sampling period      Ts     = %.6f s\n\n', Ts);

%% 3) Numerical transfer function
fprintf('3) NUMERICAL TRANSFER FUNCTION\n');
fprintf('------------------------------\n');

num = K_w_num;
den = [1, 2*zeta_p_num*omega_p_num, omega_p_num^2];
Gp = tf(num, den);

disp('Continuous-time plant G_p(s) = X_p(s)/W_p(s):');
Gp

fprintf('Equivalent scalar equation:\n');
fprintf('x_p''''(t) + %.6f x_p''(t) + %.6f x_p(t) = %.6f w_p(t)\n\n', ...
    2*zeta_p_num*omega_p_num, omega_p_num^2, K_w_num);

%% 4) Numerical state-space model
fprintf('4) NUMERICAL STATE-SPACE MODEL\n');
fprintf('------------------------------\n');

A = [0 1; -omega_p_num^2 -2*zeta_p_num*omega_p_num];
B = [0; K_w_num];
C = [1 0];
D = 0;

Gp_ss = ss(A,B,C,D);

disp('A ='); disp(A);
disp('B ='); disp(B);
disp('C ='); disp(C);
disp('D ='); disp(D);
fprintf('\n');

%% 5) Dynamic characteristics
fprintf('5) DYNAMIC CHARACTERISTICS\n');
fprintf('--------------------------\n');

wn = omega_p_num;
fn = wn/(2*pi);
wd = wn*sqrt(max(0,1-zeta_p_num^2));
fd = wd/(2*pi);

fprintf('Natural frequency     wn = %.6f rad/s = %.6f Hz\n', wn, fn);
fprintf('Damped frequency      wd = %.6f rad/s = %.6f Hz\n', wd, fd);

fprintf('\nPoles of G_p(s):\n');
disp(pole(Gp));

fprintf('Damping data:\n');
try
    damp(Gp)
catch
    fprintf('(damp() unavailable in this environment)\n');
end
fprintf('\n');

%% 6) Discrete-time model
fprintf('6) DISCRETE-TIME MODEL AT CAMERA FRAME RATE\n');
fprintf('-------------------------------------------\n');

Gp_zoh = c2d(Gp, Ts, 'zoh');
disp('Discrete-time plant G_p(z) using zero-order hold:')
Gp_zoh

fprintf('\nThis discrete model is useful because the camera samples once per frame.\n\n');

%% 7) Sanity-check plots
fprintf('7) GENERATING SANITY-CHECK PLOTS\n');
fprintf('--------------------------------\n');

figure('Name','Minimal rPPG Plant - Continuous-time responses');
subplot(2,1,1);
step(Gp, 10);
grid on;
title('Step response of latent physiological plant G_p(s)');

subplot(2,1,2);
impulse(Gp, 10);
grid on;
title('Impulse response of latent physiological plant G_p(s)');

figure('Name','Minimal rPPG Plant - Frequency response');
bode(Gp);
grid on;
title('Bode plot of G_p(s)');

%% 8) Driven response near nominal HR frequency
fprintf('8) SIMULATED DRIVEN RESPONSE\n');
fprintf('----------------------------\n');

t_sim = 0:Ts:20;
f_drive = f_HR_hz;
w_drive = sin(2*pi*f_drive*t_sim);
x_resp = lsim(Gp, w_drive, t_sim);

figure('Name','Driven response near nominal heart-rate frequency');
plot(t_sim, w_drive, 'LineWidth', 1.1); hold on;
plot(t_sim, x_resp, 'LineWidth', 1.4);
grid on;
xlabel('Time [s]');
ylabel('Amplitude');
legend('Input w_p(t)', 'Output x_p(t)', 'Location', 'best');
title('Driven latent physiological plant');

%% 9) Summary
fprintf('9) COMPACT SUMMARY\n');
fprintf('------------------\n');
fprintf('Chosen first plant:\n');
fprintf('   x_p'''' + 2*zeta_p*omega_p*x_p'' + omega_p^2*x_p = K_w*w_p\n');
fprintf('with transfer function:\n');
fprintf('   G_p(s) = K_w / (s^2 + 2*zeta_p*omega_p*s + omega_p^2)\n\n');

fprintf('Interpretation:\n');
fprintf(' - Input  w_p(t): latent cardiovascular excitation\n');
fprintf(' - Output x_p(t): observable pulsatile physiological state at skin\n');
fprintf(' - This is NOT yet the camera or CHROM model\n');
fprintf(' - It is the first building block for the complete rPPG plant\n\n');

fprintf('Next recommended step:\n');
fprintf(' Build the optical measurement equation\n');
fprintf('   c(t) = c0 + H_p*x_p(t) + H_m*x_m(t) + H_l*x_l(t) + n(t)\n');
fprintf(' and then derive how CHROM acts as a disturbance-rejecting measurement transform.\n');

fprintf('\nDone.\n');

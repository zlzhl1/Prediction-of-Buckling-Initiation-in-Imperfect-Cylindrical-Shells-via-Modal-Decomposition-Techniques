clear; clc; close all;

load('VORT.mat');       
load('CCcool.mat');     
X = VORT;
[Nxy, Nt] = size(X);
Nx = 199; Ny = 449;     
dt = 1;                 

X1 = X(:, 1:end-1);
X2 = X(:, 2:end);

%% SVD 
[U, S, V] = svd(X1, 'econ');
r = 21;                        
Ur = U(:,1:r);
Sr = S(1:r,1:r);
Vr = V(:,1:r);

%% Calculating Atilde and DMD modes
Atilde = Ur' * X2 * Vr / Sr;
[W, D] = eig(Atilde);
Phi = X2 * Vr / Sr * W;         % DMD modal
lambda = diag(D);
omega = log(lambda) / dt;

x1 = X(:,1);
b = Phi \ x1;

%% Constructing temporal evolution
t = (0:Nt-1) * dt;
time_dynamics = zeros(r, Nt);
for i = 1:Nt
    time_dynamics(:,i) = b .* exp(omega * t(i));
end
X_dmd = real(Phi * time_dynamics);     

%% reconstruction error
rel_error = norm(X - X_dmd, 'fro') / norm(X, 'fro');
fprintf('Reconstruction relative error with r = %d: %.4f\n', r, rel_error);

%% Visualisation of the frequency spectrum
frequencies = abs(imag(omega)) / (2*pi);
amplitudes = abs(b);

figure;
stem(frequencies, amplitudes, 'filled');
xlabel('Frequency (Hz)');
ylabel('|b| (Initial amplitude)');
title('DMD Frequency Spectrum');
grid on;

%% Visualisation of DMD time evolution coefficients (first 4 modes)
figure;
for i = 1:4
    subplot(2,2,i);
    plot(t, real(time_dynamics(i,:)), 'b-', 'LineWidth', 1.5); hold on;
    plot(t, abs(time_dynamics(i,:)), 'r--', 'LineWidth', 1.2);
    xlabel('Time'); ylabel(['Mode ', num2str(i)]);
    legend('Real part', 'Magnitude');
    title(['Time Dynamics of Mode ', num2str(i)]);
    grid on;
end

%% Visualisation of DMD modal space structures (first 4)
figure;
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:4
    nexttile;
    mode_i = reshape(real(Phi(:,i)), Nx, Ny);
    imagesc(mode_i);
    colormap(CC); colorbar;
    title(['DMD Mode ', num2str(i)]);
    axis equal tight;
end

title(t, 'Top 4 DMD Modes', 'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(gcf, 'dmd_modes_top4.png', 'Resolution', 300);

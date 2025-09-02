clear; clc; close all;

load('W.mat');        
load('t.mat'); 
[nSpace, Nt] = size(W);
nrC = 200; nrL = 51;    
Nx = nrC; Ny = nrL;     
R = 0.1;      
L = 0.1609;   

t = t(:);                  
dt = mean(diff(t));       
fprintf('Length of time vector: %d, average dt = %.6f s\n', length(t), dt);

X1 = W(:, 1:end-1);       
X2 = W(:, 2:end);       

%% SVD Decomposition
[U, S, V] = svd(X1, 'econ');
r = 20;                                % Retained modal number
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);

%% DMD Mode
Atilde = Ur' * X2 * Vr / Sr;
[W_dmd, D] = eig(Atilde);
Phi = X2 * Vr / Sr * W_dmd;
lambda = diag(D);
omega = log(lambda) / dt;

%% time evolution
x1 = W(:,1);
b = Phi \ x1;

time_dynamics = zeros(r, Nt);
for i = 1:Nt
    time_dynamics(:,i) = b .* exp(omega * t(i));
end

X_dmd = real(Phi * time_dynamics);    % DMD reconstructed field

rel_error = norm(W - X_dmd, 'fro') / norm(W, 'fro');
fprintf('Reconstruction relative error (r=%d): %.6f\n', r, rel_error);

%% frequency spectrum
frequencies = abs(imag(omega)) / (2*pi);
amplitudes = abs(b);

figure;
stem(frequencies, amplitudes, 'filled');
xlabel('Frequency (Hz)');
ylabel('|b| (Initial Amplitude)');
title('DMD Frequency Spectrum');
grid on;

%% Select Top-4 Modes by Energy

phi_norm2 = vecnorm(Phi, 2, 1)'.^2;        % (r×1)
energy    = phi_norm2 .* (abs(b).^2);      % (r×1)

[energy_sorted, idx_sorted] = sort(energy, 'descend');
K = min(4, numel(idx_sorted));
idx_top = idx_sorted(1:K);

% Top-K 
Phi_top   = Phi(:, idx_top);             
omega_top = omega(idx_top);              
b_top     = b(idx_top);                   
freq_top  = frequencies(idx_top);

fprintf('按能量降序的前 %d 个模态索引：', K); fprintf('%d ', idx_top); fprintf('\n');
for ii = 1:K
    fprintf('  #%d: f=%.4f Hz, Energy=%.4e\n', idx_top(ii), freq_top(ii), energy_sorted(ii));
end

% Time evolution of Top-K modes only
time_dyn_top = zeros(K, Nt);
for i = 1:Nt
    time_dyn_top(:,i) = b_top .* exp(omega_top * t(i));
end
%% Time evolution
figure;
for i = 1:K
    subplot(2,2,i);
    plot(t, real(time_dyn_top(i,:)), 'b-', 'LineWidth', 1.5); hold on;
    plot(t, abs(time_dyn_top(i,:)), 'r--', 'LineWidth', 1.2);
    xlabel('Time (s)'); ylabel(sprintf('Mode %d (by energy)', i));
    legend('Real part', 'Magnitude', 'Location','best');
    title(sprintf('Top-%d by Energy: f=%.3f Hz', i, freq_top(i)));
    grid on;
end

%% Spatial visualization
figure;
tl = tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:K
    nexttile;
    phi_i = real(Phi_top(:,i));
    phi2d = reshape(phi_i, nrC, nrL);
    phi2d(nrC+1,:) = phi2d(1,:);   % 闭合
    phi2d = phi2d';

    imagesc(phi2d);
    colormap(jet); colorbar;
    title(sprintf('Energy-Top %d', i));
    axis equal tight;
end

title(tl, 'Energy-Top 4 DMD Modes (Displacement Field)', 'FontSize', 14, 'FontWeight', 'bold');
exportgraphics(gcf, 'dmd_modes_W_energy_top4.png', 'Resolution', 300);
%% GIF animation parameters
dtGIF = 0.05;
fileGIF_orig = 'dmd_original.gif';
fileGIF_recon = 'dmd_reconstructed.gif';

theta = linspace(0, 2*pi, nrC+1);
x_axis = linspace(0, L, nrL);
[T1, X1grid] = meshgrid(theta, x_axis);
Ysurf = R * cos(T1);
Zsurf = R * sin(T1);

% Original data GIF
figure('Color', [1 1 1]);
scrsz = get(groot,'ScreenSize');
set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);
for k = 1:Nt
    Wk = reshape(W(:,k), nrC, nrL);
    Wk(nrC+1,:) = Wk(1,:);
    Wk = Wk';

    surf(Ysurf, Zsurf, X1grid, Wk, 'EdgeColor', 'none');
    shading interp; colormap jet; axis equal off;
    view(-90,30);
    title(sprintf('Original W, Frame %d / %d', k, Nt));

    frame = getframe(gcf);
    [im, cm] = rgb2ind(frame2im(frame), 256);
    if t == 1
        imwrite(im, cm, fileGIF_orig, 'gif', 'Loopcount', inf, 'DelayTime', dtGIF);
    else
        imwrite(im, cm, fileGIF_orig, 'gif', 'WriteMode', 'append', 'DelayTime', dtGIF);
    end
end
disp(['Original data GIF generated: ', fileGIF_orig]);

%  DMD reconstructed data GIF
figure('Color', [1 1 1]);
scrsz = get(groot,'ScreenSize');
set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);
for k = 1:Nt
    Wk_dmd = reshape(X_dmd(:,k), nrC, nrL);
    Wk_dmd(nrC+1,:) = Wk_dmd(1,:);
    Wk_dmd = Wk_dmd';

    surf(Ysurf, Zsurf, X1grid, Wk_dmd, 'EdgeColor', 'none');
    shading interp; colormap jet; axis equal off;
    view(-90,30);
    
    title(sprintf('DMD Reconstruction, Frame %d / %d', k, Nt));

    frame = getframe(gcf);
    [im, cm] = rgb2ind(frame2im(frame), 256);
    if k == 1
        imwrite(im, cm, fileGIF_recon, 'gif', 'Loopcount', inf, 'DelayTime', dtGIF);
    else
        imwrite(im, cm, fileGIF_recon, 'gif', 'WriteMode', 'append', 'DelayTime', dtGIF);
    end
end
disp(['DMD reconstructed GIF generated: ', fileGIF_recon]);

%% 
% X_dmd_top4 = real(Phi_top * time_dyn_top);
% figure; imagesc(X_dmd_top4(:,1)); title('Top-4 DMD reconstruction at t_1');

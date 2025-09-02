clear; clc; close all;

load('W.mat');         
[nSpace, Nt] = size(W);
nrC = 200; nrL = 51;   
Nx = nrC; Ny = nrL;  
R = 0.1;    
L = 0.1609; 


dt = 1;  
Wmean = mean(W, 2);
W = W - Wmean;  

X1 = W(:, 1:end-1);   
X2 = W(:, 2:end);      

%% SVD decomposition
[U, S, V] = svd(X1, 'econ');
r = 20;                                % Number of retained modes
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);

%% DMD modes
Atilde = Ur' * X2 * Vr / Sr;
[W_dmd, D] = eig(Atilde);
Phi = X2 * Vr / Sr * W_dmd;
lambda = diag(D);
omega = log(lambda) / dt;

%% Time evolution
x1 = W(:,1);
b = Phi \ x1;
t = (0:Nt-1) * dt;
time_dynamics = zeros(r, Nt);
for i = 1:Nt
    time_dynamics(:,i) = b .* exp(omega * t(i));
end

X_dmd = real(Phi * time_dynamics);   % DMD reconstructed field

rel_error = norm(W - X_dmd, 'fro') / norm(W, 'fro');
fprintf('Reconstruction relative error (r=%d): %.6f\n', r, rel_error);

%% Frequency spectrum
frequencies = abs(imag(omega)) / (2*pi);
amplitudes  = abs(b);

% sort by frequency so the stem looks cleaner
[frequencies_s, ord_f] = sort(frequencies, 'ascend');
amplitudes_s = amplitudes(ord_f);

figure('Color',[1 1 1]);
h = stem(frequencies_s, amplitudes_s, 'o', 'LineWidth', 1.2);
set(h, 'Marker', 'o', 'MarkerSize', 4);  % stem does not support 'filled'
xlabel('Frequency (Hz)');
ylabel('|b| (initial amplitude)');
title('DMD Frequency Spectrum');
grid on; box on;

%% Growth-rate map: frequency vs sigma, color = energy
sigma = real(omega);
freq  = abs(imag(omega)) / (2*pi);
phi_norm2 = vecnorm(Phi, 2, 1)'.^2;     % ||phi_i||_2^2
energy    = phi_norm2 .* (abs(b).^2);   % mode energy
energy_n  = energy / max(energy + eps); % normalize to [0,1]

figure('Color',[1 1 1]);
sz = 36;
scatter(freq, sigma, sz, energy_n, 'filled'); hold on;
yline(0,'--','Color',[0.4 0.4 0.4]);
xlabel('Frequency (Hz)');
ylabel('\sigma (growth rate)');
title('DMD Modes: Growth Rate vs Frequency (color = normalized energy)');
colormap(turbo); cb = colorbar; cb.Label.String = 'Normalized energy';
grid on; box on;

[~, eorder] = sort(energy, 'descend');
for k = 1:min(4, numel(eorder))
    i = eorder(k);
    text(freq(i), sigma(i), sprintf('  #%d', k), ...
        'VerticalAlignment','bottom','FontSize',9);
end

%% Low-Frequency + High-Growth, then sort-by-Energy (Top-4)
K_desired = min(4, r);


sigma = real(omega);                      
freq  = abs(imag(omega)) / (2*pi);        
phi_norm2 = vecnorm(Phi, 2, 1)'.^2;       
energy    = phi_norm2 .* (abs(b).^2);     

% 1) Prefer “high-growth” sigma > 0; if not enough, add “near-neutral” (-eps<=sigma<=0)
sigma_eps = 1e-3;
idx_pos   = find(sigma > 0);
idx_near0 = find(sigma <= 0 & sigma >= -sigma_eps);
cand_idx  = unique([idx_pos; idx_near0], 'stable');

% 2) If fewer than 4 candidates, supplement from the rest using low frequency → high growth → high energy
if numel(cand_idx) < K_desired
    rest = setdiff((1:numel(omega))', cand_idx, 'stable');
    Mrest = [freq(rest), -sigma(rest), -energy(rest)];
    [~, ord_rest] = sortrows(Mrest, [1 2 3]);
    need = K_desired - numel(cand_idx);
    take = min(need, numel(rest));
    cand_idx = [cand_idx; rest(ord_rest(1:take))];
end

% 3) Rank within candidates: low frequency → high growth → high energy
M = [freq(cand_idx), -sigma(cand_idx), -energy(cand_idx)];
[~, ord_cand] = sortrows(M, [1 2 3]);
idx_pref = cand_idx(ord_cand(1:min(K_desired, numel(cand_idx))));

% 4) Final ordering: sort by energy descending
[~, ord_energy] = sort(energy(idx_pref), 'descend');
idx_sel = idx_pref(ord_energy);

K_eff = numel(idx_sel);

Phi_top    = Phi(:, idx_sel);
omega_top  = omega(idx_sel);
b_top      = b(idx_sel);
freq_top   = freq(idx_sel);
sigma_top  = sigma(idx_sel);
energy_top = energy(idx_sel);


fprintf('Selected (Low-Freq + High-Growth → energy-ranked), count=%d:\n', K_eff);
for ii = 1:K_eff
    fprintf('  #%d: idx=%d | f=%.6g Hz | sigma=%.3e | Energy=%.3e\n', ...
        ii, idx_sel(ii), freq_top(ii), sigma_top(ii), energy_top(ii));
end



time_dyn_top = zeros(K_eff, Nt);
for i = 1:Nt
    time_dyn_top(:,i) = b_top .* exp(omega_top * t(i));
end

%% Time evolution
figure('Color',[1 1 1]);
nplot = max(4, K_eff);          
for i = 1:min(4, K_eff)
    subplot(2,2,i);
    plot(t, real(time_dyn_top(i,:)), '-', 'LineWidth', 1.4); hold on;
    plot(t, abs(time_dyn_top(i,:)), '--', 'LineWidth', 1.2);
    xlabel('Time (s)'); 
    ylabel('Amplitude');
    title(sprintf('Mode idx=%d | f=%.3g Hz | \\sigma=%.2e | E=%.2e', ...
        idx_sel(i), freq_top(i), sigma_top(i), energy_top(i)));
    legend('Real part','Magnitude','Location','best'); grid on; box on;
end
sgtitle('Time Evolution of Selected DMD Modes (Top-4 by energy)');


%% Spatial modes
figure;
tl = tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:K_eff
    nexttile;
    phi_i = real(Phi_top(:,i));
    phi2d = reshape(phi_i, nrC, nrL);
    phi2d(nrC+1,:) = phi2d(1,:);   % 闭合
    phi2d = phi2d';
    imagesc(phi2d);
    colormap(jet); colorbar;
    axis equal tight;
    title(sprintf('idx=%d | f=%.3g Hz | \\sigma=%.2e', ...
        idx_sel(i), freq_top(i), sigma_top(i)));
end
title(tl, 'Selected DMD Modes', ...
    'FontSize', 14, 'FontWeight', 'bold');
exportgraphics(gcf, 'dmd_modes_W_lowfreq_highgrowth_energy_top4.png', 'Resolution', 300);

%% GIF animation parameters
dtGIF = 0.05;
fileGIF_orig = 'dmd_original.gif';
fileGIF_recon = 'dmd_reconstructed.gif';


theta = linspace(0, 2*pi, nrC+1);
x_axis = linspace(0, L, nrL);
[T1, X1grid] = meshgrid(theta, x_axis);
Ysurf = R * cos(T1);
Zsurf = R * sin(T1);

% % Original data GIF
% figure('Color', [1 1 1]);
% scrsz = get(groot,'ScreenSize');
% set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);
% for k = 1:Nt
%     Wk = reshape(W(:,k), nrC, nrL);
%     Wk(nrC+1,:) = Wk(1,:);
%     Wk = Wk';
% 
%     surf(Ysurf, Zsurf, X1grid, Wk, 'EdgeColor', 'none');
%     shading interp; colormap jet; axis equal off;
%     view(-90,30);
%     title(sprintf('Original W, Frame %d / %d', k, Nt));
% 
%     frame = getframe(gcf);
%     [im, cm] = rgb2ind(frame2im(frame), 256);
%     if k == 1
%         imwrite(im, cm, fileGIF_orig, 'gif', 'Loopcount', inf, 'DelayTime', dtGIF);
%     else
%         imwrite(im, cm, fileGIF_orig, 'gif', 'WriteMode', 'append', 'DelayTime', dtGIF);
%     end
% end
% disp(['Original data GIF generated: ', fileGIF_orig]);
% 
% % DMD reconstructed data GIF
% figure('Color', [1 1 1]);
% scrsz = get(groot,'ScreenSize');
% set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);
% for k = 1:Nt
%     Wk_dmd = reshape(X_dmd(:,k), nrC, nrL);
%     Wk_dmd(nrC+1,:) = Wk_dmd(1,:);
%     Wk_dmd = Wk_dmd';
% 
%     surf(Ysurf, Zsurf, X1grid, Wk_dmd, 'EdgeColor', 'none');
%     shading interp; colormap jet; axis equal off;
%     view(-90,30);
% 
%     title(sprintf('DMD Reconstruction, Frame %d / %d', k, Nt));
% 
%     frame = getframe(gcf);
%     [im, cm] = rgb2ind(frame2im(frame), 256);
%     if k == 1
%         imwrite(im, cm, fileGIF_recon, 'gif', 'Loopcount', inf, 'DelayTime', dtGIF);
%     else
%         imwrite(im, cm, fileGIF_recon, 'gif', 'WriteMode', 'append', 'DelayTime', dtGIF);
%     end
% end
% disp(['DMD reconstructed GIF generated: ', fileGIF_recon]);

% X_dmd_top4 = real(Phi_top * time_dyn_top);
% figure; imagesc(X_dmd_top4(:,1)); title('Top-4 DMD reconstruction at t_1');

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
fprintf('Length of time vector：%d，average dt = %.4f s\n', length(t), dt);

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

time_dynamics = zeros(r, Nt);
for i = 1:Nt
    time_dynamics(:,i) = b .* exp(omega * t(i));
end

X_dmd = real(Phi * time_dynamics);    %  DMD reconstructed field

rel_error = norm(W - X_dmd, 'fro') / norm(W, 'fro');
fprintf('Reconstruction relative error (r=%d): %.4f\n', r, rel_error);

%% Frequency spectrum
frequencies = abs(imag(omega)) / (2*pi);
amplitudes = abs(b);

figure;
stem(frequencies, amplitudes, 'filled');
xlabel('Frequency (Hz)');
ylabel('|b| (Initial Amplitude)');
title('DMD Frequency Spectrum');
grid on;

%% Time evolution (first 4 modes)
figure;
for i = 1:4
    subplot(2,2,i);
    plot(t, real(time_dynamics(i,:)), 'b-', 'LineWidth', 1.5); hold on;
    plot(t, abs(time_dynamics(i,:)), 'r--', 'LineWidth', 1.2);
    xlabel('Frame'); ylabel(['Mode ', num2str(i)]);
    legend('Real part', 'Magnitude');
    title(['Time Dynamics of DMD Mode ', num2str(i)]);
    grid on;
end

%% Spatial mode visualization (first 4)
figure;
tl = tiledlayout(2,2, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:4
    nexttile;
    phi_i = real(Phi(:,i));
    phi2d = reshape(phi_i, nrC, nrL);
    phi2d(nrC+1,:) = phi2d(1,:);   % 闭合
    phi2d = phi2d';

    imagesc(phi2d);
    colormap(jet); colorbar;
    title(['DMD Mode ', num2str(i)]);
    axis equal tight;
end

title(tl, 'Top 4 DMD Modes (Displacement Field)', 'FontSize', 14, 'FontWeight', 'bold');
exportgraphics(gcf, 'dmd_modes_W_top4.png', 'Resolution', 300);
%% GIF Animation Parameters
dtGIF = 0.05;
fileGIF_orig = 'dmd_original.gif';
fileGIF_recon = 'dmd_reconstructed.gif';

theta = 0:2*pi/nrC:2*pi;
x = linspace(0, L, nrL);
[T1, X1grid] = meshgrid(theta, x);
Ysurf = R * cos(T1);
Zsurf = R * sin(T1);

% Original data GIF
figure('Color', [1 1 1]);
for t = 1:Nt
    Wt = reshape(W(:,t), nrC, nrL);
    Wt(nrC+1,:) = Wt(1,:);
    Wt = Wt';
    scrsz = get(groot,'ScreenSize');
    set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);
    surf(Ysurf, Zsurf, X1grid, Wt, 'EdgeColor', 'none');
    shading interp; colormap jet; axis equal off;
    view(-90,30);
    title(sprintf('Original W, Frame %d / %d', t, Nt));

    frame = getframe(gcf);
    [im, cm] = rgb2ind(frame2im(frame), 256);
    if t == 1
        imwrite(im, cm, fileGIF_orig, 'gif', 'Loopcount', inf, 'DelayTime', dtGIF);
    else
        imwrite(im, cm, fileGIF_orig, 'gif', 'WriteMode', 'append', 'DelayTime', dtGIF);
    end
end
disp([' Original data GIF generated: ', fileGIF_orig]);

% DMD reconstructed data GIF
figure('Color', [1 1 1]);
for t = 1:Nt
    Wt_dmd = reshape(X_dmd(:,t), nrC, nrL);
    Wt_dmd(nrC+1,:) = Wt_dmd(1,:);
    Wt_dmd = Wt_dmd';

    surf(Ysurf, Zsurf, X1grid, Wt_dmd, 'EdgeColor', 'none');
    shading interp; colormap jet; axis equal off;
    view(-90,30);
    scrsz = get(groot,'ScreenSize');
    set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);
    title(sprintf('DMD Reconstruction, Frame %d / %d', t, Nt));

    frame = getframe(gcf);
    [im, cm] = rgb2ind(frame2im(frame), 256);
    if t == 1
        imwrite(im, cm, fileGIF_recon, 'gif', 'Loopcount', inf, 'DelayTime', dtGIF);
    else
        imwrite(im, cm, fileGIF_recon, 'gif', 'WriteMode', 'append', 'DelayTime', dtGIF);
    end
end
disp(['DMD reconstructed GIF generated: ', fileGIF_recon]);

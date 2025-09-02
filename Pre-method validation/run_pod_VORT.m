clear; clc; close all;

load('VORT.mat');       
load('CCcool.mat');     
X = VORT;

Nx = 199; Ny = 449; Nt = size(X, 2); 

%% POD decomposition
X_mean = mean(X, 2);
X_fluct = X - X_mean;

C = X_fluct' * X_fluct;
[W, D] = eig(C);
[lambda, idx] = sort(diag(D), 'descend');
W = W(:, idx);
Phi = X_fluct * W;

for i = 1:Nt
    Phi(:,i) = Phi(:,i) / norm(Phi(:,i));
end

A = Phi' * X_fluct;

% Reconstruction with the first r modes
r = 10;
X_rec = Phi(:,1:r) * A(1:r,:) + X_mean;


%% Generate GIF - original Data
vortmin = -5; vortmax = 5;

filename_orig = 'vorticity_original.gif';
% Save the 1st frame
VORT = reshape(X(:,1), Nx, Ny);
VORT(VORT > vortmax) = vortmax;
VORT(VORT < vortmin) = vortmin;

figure;
imagesc(VORT); colormap(turbo); colorbar;
title('Original Vorticity: t = 1');
axis equal tight;

saveas(gcf, 'vorticity_original_t1.png'); 

figure('Visible','off');

for t = 1:Nt
    VORT = reshape(X(:,t), Nx, Ny);
    VORT(VORT > vortmax) = vortmax;
    VORT(VORT < vortmin) = vortmin;

    imagesc(VORT); colormap(CC); colorbar;
    title(['Original Vorticity: t = ', num2str(t)]);
    axis equal tight;

    frame = getframe(gcf);
    [im, cm] = rgb2ind(frame2im(frame), 256);
    if t == 1
        imwrite(im, cm, filename_orig, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
    else
        imwrite(im, cm, filename_orig, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
    end
end

%% Generate GIF - Reconstruction data
filename_rec = ['vorticity_reconstructed_r', num2str(r), '.gif'];
% Save the 1st frame
VORT = reshape(X_rec(:,1), Nx, Ny);
VORT(VORT > vortmax) = vortmax;
VORT(VORT < vortmin) = vortmin;

figure;
imagesc(VORT); colormap(CC); colorbar;
title(['Reconstructed Vorticity (r=', num2str(r), '): t = 1']);
axis equal tight;

saveas(gcf, ['vorticity_reconstructed_r', num2str(r), '_t1.png']);

figure('Visible','off');

for t = 1:Nt
    VORT = reshape(X_rec(:,t), Nx, Ny);
    VORT(VORT > vortmax) = vortmax;
    VORT(VORT < vortmin) = vortmin;

    imagesc(VORT); colormap(CC); colorbar;
    title(['Reconstructed Vorticity (r=', num2str(r), '): t = ', num2str(t)]);
    axis equal tight;

    frame = getframe(gcf);
    [im, cm] = rgb2ind(frame2im(frame), 256);
    if t == 1
        imwrite(im, cm, filename_rec, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
    else
        imwrite(im, cm, filename_rec, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
    end
end

disp('GIFs saved:');
disp([' - ', filename_orig]);
disp([' - ', filename_rec]);


%% POD modes energy spectra

energy = lambda / sum(lambda);            
cumulative_energy = cumsum(energy);       

figure;
subplot(2,1,1);
plot(1:length(energy), energy, 'b.-');
xlabel('Mode number'); ylabel('Energy fraction');
title('POD Energy Spectrum');
grid on;

subplot(2,1,2);
plot(1:length(cumulative_energy), cumulative_energy, 'r.-');
xlabel('Mode number'); ylabel('Cumulative energy');
title('Cumulative POD Energy');
grid on;

r95 = find(cumulative_energy >= 0.95, 1);
fprintf('%d modes capture at least 95%% of total energy.\n', r95);

%% Visualising the spatial structure of POD modes（first 4）
figure;
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:4
    nexttile;
    mode_i = reshape(Phi(:,i), Nx, Ny);
    imagesc(mode_i);
    colormap(CC); colorbar;
    title(['POD Mode ', num2str(i)]);
    axis equal tight;
end

title(t, 'Top 4 POD Modes', 'FontWeight', 'bold');

exportgraphics(gcf, 'pod_modes_top4.png', 'Resolution', 300);

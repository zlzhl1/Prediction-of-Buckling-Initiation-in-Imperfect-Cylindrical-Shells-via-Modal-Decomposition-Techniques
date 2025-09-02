clear; clc; close all;

load('DATA.mat');     
whos               
X = real(XX);               
Nx = 400; Nt = size(X, 2);
whos
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

%% Reconstruction 
% Reconstruction with the first r modes
r = 10;
X_rec = Phi(:,1:r) * A(1:r,:) + X_mean;

% Reconstruction error assessment
rel_error = norm(X - X_rec, 'fro') / norm(X, 'fro');
fprintf('Reconstruction relative error with r = %d: %.4f\n', r, rel_error);
disp(['X is complex: ', num2str(~isreal(X))]);
disp(['X_rec is complex: ', num2str(~isreal(X_rec))]);

%% Generate GIF - original data
filename_orig = 'data_original.gif';
xgrid = linspace(-10, 10, Nx);
figure('Visible','off');

for t = 1:Nt
    plot(xgrid, X(:,t), 'b-', 'LineWidth', 1.2);
    ylim([-3, 3]);
    xlabel('x'); ylabel('q(x)');
    title(['Original Data: t = ', num2str(t)]);
    grid on;

    frame = getframe(gcf);
    [im, cm] = rgb2ind(frame2im(frame), 256);
    if t == 1
        imwrite(im, cm, filename_orig, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
    else
        imwrite(im, cm, filename_orig, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
    end
end

%% Generate GIF - Reconstruction data
filename_rec = ['data_reconstructed_r', num2str(r), '.gif'];
figure('Visible','off');

for t = 1:Nt
    plot(xgrid, X_rec(:,t), 'r--', 'LineWidth', 1.2);
    ylim([-3, 3]);
    xlabel('x'); ylabel('q(x)');
    title(['Reconstructed Data (r = ', num2str(r), '): t = ', num2str(t)]);
    grid on;

    frame = getframe(gcf);
    [im, cm] = rgb2ind(frame2im(frame), 256);
    if t == 1
        imwrite(im, cm, filename_rec, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
    else
        imwrite(im, cm, filename_rec, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
    end
end

disp(' GIFs saved:');
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

clear; clc; close all;

load('DATA.mat');       % DATA: size 400 x 200
X = XX;
Nx = size(X,1);
Nt = size(X,2);
xgrid = linspace(-10, 10, Nx);

%% Constructing the DMD Input Matrix
X1 = X(:, 1:end-1);   
X2 = X(:, 2:end);      
dt = 1;                

%% SVD
[U, S, V] = svd(X1, 'econ');

r = 10;
Ur = U(:,1:r);
Sr = S(1:r,1:r);
Vr = V(:,1:r);

%% Build the approximate linear model Atilde and find its eigen-decomposition
Atilde = Ur' * X2 * Vr / Sr;
[W, D] = eig(Atilde);
Phi = X2 * Vr / Sr * W;         
lambda = diag(D);               
omega = log(lambda)/dt;         

%% Calculation of time coefficients (initial conditions)
x1 = X(:,1);
b = Phi \ x1;

%% Reconstruction 
% Reconstruction time evolution
t = (0:Nt-1)*dt;
time_dynamics = zeros(r, Nt);
for i = 1:Nt
    time_dynamics(:,i) = b .* exp(omega * t(i));
end
X_dmd = real(Phi * time_dynamics);

% Reconstruction error assessment
rel_error = norm(X - X_dmd, 'fro') / norm(X, 'fro');
fprintf('Reconstruction relative error with r = %d: %.4f\n', r, rel_error);

%% Visualisation of mode frequency spectrum
figure;
plot(real(omega), imag(omega), 'ro');
xlabel('Re(\omega)'); ylabel('Im(\omega)');
title('DMD Eigenvalues in Continuous-Time Domain');
grid on;

%% Visualisation of DMD modes
figure;
for i = 1:4
    subplot(2,2,i);
    plot(xgrid, real(Phi(:,i)), 'b-', 'LineWidth', 1.5);
    xlabel('x'); ylabel(['\phi_', num2str(i), '(x)']);
    title(['Mode ', num2str(i)]);
    grid on;
end

%% Optional GIF export
export_gif = false;   
if export_gif
    filename = 'data_dmd_reconstruction.gif';
    figure('Visible','off');
    for i = 1:Nt
        plot(xgrid, X(:,i), 'b-', 'LineWidth', 1.2); hold on;
        plot(xgrid, X_dmd(:,i), 'r--', 'LineWidth', 1.2); hold off;
        ylim([-3 3]);
        xlabel('x'); ylabel('q(x)');
        title(['DMD Reconstruct t = ', num2str(i)]);
        legend('Original','DMD');
        drawnow;

        frame = getframe(gcf);
        [im, cm] = rgb2ind(frame2im(frame), 256);
        if i == 1
            imwrite(im, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
        else
            imwrite(im, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
        end
    end
    disp(['GIF saved: ', filename]);
end
%% DMD Frequency Spectrum Analysis
frequencies = abs(imag(omega)) / (2*pi); 
amplitudes = abs(b);                      

figure;
stem(frequencies, amplitudes, 'filled');
xlabel('Frequency (Hz)');
ylabel('|b| (initial amplitude)');
title('DMD Mode Frequency Spectrum');
grid on;
%% Visualisation DMD Modal time evolution coefficients
figure;
for i = 1:4
    subplot(2,2,i);
    plot(t, real(time_dynamics(i,:)), 'b-', 'LineWidth', 1.5); hold on;
    plot(t, abs(time_dynamics(i,:)), 'r--', 'LineWidth', 1.2);
    xlabel('Time'); ylabel(['Mode ', num2str(i), ' amplitude']);
    legend('Real part', 'Magnitude');
    title(['DMD Time Dynamics - Mode ', num2str(i)]);
    grid on;
end

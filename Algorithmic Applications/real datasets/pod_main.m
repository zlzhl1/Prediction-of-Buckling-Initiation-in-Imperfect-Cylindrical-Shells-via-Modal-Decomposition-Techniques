clear; clc; close all;

load W.mat         
[nSpace, Nt] = size(W);
nrC   = 180;     
nrL   = 76;    
R     = 0.1;      
L     = 0.1609;  
rPOD  = 4;      
dtGIF = 0.05;      

%% Generate cylindrical static meshes
theta = 0:2*pi/nrC:2*pi; 
x     = linspace(0, L, nrL);

[T1,X1] = meshgrid(theta, x);    
Ysurf   = R * cos(T1);
Zsurf   = R * sin(T1);

%% POD Decomposition & Reconstruction
X        = W;                 
X_mean   = mean(X,2);
X_fluct  = X - X_mean;

C        = X_fluct' * X_fluct;
[Vec, D] = eig(C);
[lambda, idx] = sort(diag(D),'descend');
Vec      = Vec(:,idx);
tol = 1e-12;
keep = lambda > tol;
lambda = lambda(keep);
Vec    = Vec(:,keep);


Phi = X_fluct * (Vec ./ sqrt(lambda'));
A        = Phi' * X_fluct;       
X_rec    = Phi(:,1:rPOD)*A(1:rPOD,:) + X_mean;
%% Reconfiguration error assessment


rel_err_full = norm(X - X_rec, 'fro') / norm(X, 'fro');

% Only look at the volume of fluctuations
X_rec_fluct = X_rec - mean(X_rec, 2);
rel_err_fluct = norm(X_fluct - X_rec_fluct, 'fro') / norm(X_fluct, 'fro');

fprintf('Relative error (full field):   %.6f\n', rel_err_full);
fprintf('Relative error (fluctuation): %.6f\n', rel_err_fluct);

% Relative error per frame
den = vecnorm(X, 2, 1);                 % L2 norms per frame
den(den==0) = 1;
e_t = vecnorm(X - X_rec, 2, 1) ./ den; 
figure; plot(1:Nt, e_t, 'LineWidth', 1.5); grid on;
xlabel('Frame'); ylabel('Relative error per frame');
title(sprintf('Per-frame relative error (rPOD=%d)', rPOD));

% 4) RMSE at each spatial point
rmse_space = sqrt(mean((X - X_rec).^2, 2));  % nSpace×1
drawFrame(rmse_space, Ysurf, Zsurf, X1, 1, 1, sprintf('rmse_space_r%d.png', rPOD));
disp(' Saved: RMSE visualization at each spatial point');

% Energy retention
energy_kept = sum(lambda(1:rPOD)) / sum(lambda);
fprintf('Energy kept by r=%d: %.4f (%.2f%%)\n', rPOD, energy_kept, 100*energy_kept);

% 6) Global R^2
SS_tot = norm(X - mean(X,2), 'fro')^2;
SS_res = norm(X - X_rec,       'fro')^2;
R2 = 1 - SS_res / SS_tot;
fprintf('Global R^2: %.4f\n', R2);

% %% Create GIF : Original Data
% fileGIF1 = 'cylinder_3d_original.gif';
% figure('Color',[1 1 1]);
% for t = 1:Nt
%     ww = reshape(W(:,t), nrC, nrL); 
%     ww(nrC+1,:) = ww(1,:);       
%     ww = ww';                   
%     scrsz = get(groot,'ScreenSize');
%     set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);
%     surf(Ysurf, Zsurf, X1, ww, 'EdgeColor','none');
%     colormap jet; shading interp; axis equal off;
% 
%     view(-90,30);                 
%     title(sprintf('Original, frame %d / %d', t, Nt));
% 
%     frame = getframe(gcf);
%     [im,cm] = rgb2ind(frame2im(frame),256);
%     if t==1
%         imwrite(im,cm,fileGIF1,'gif','Loopcount',inf,'DelayTime',dtGIF);
%     else
%         imwrite(im,cm,fileGIF1,'gif','WriteMode','append','DelayTime',dtGIF);
%     end
% end
% disp(['GIF generated: ' fileGIF1]);
% 
% %% Create GIF : POD Reconstruction
% fileGIF2 = sprintf('cylinder_3d_reconstructed_r%d.gif', rPOD);
% figure('Color',[1 1 1]);
% for t = 1:Nt
%     ww = reshape(W(:,t), nrC, nrL);  
%     ww(nrC+1,:) = ww(1,:);       
%     ww = ww';                
%     scrsz = get(groot,'ScreenSize');
%     set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);
%     surf(Ysurf, Zsurf, X1, ww, 'EdgeColor','none');
%     colormap jet; shading interp; axis equal off;
% 
%     view(-90,30);
%     title(sprintf('Reconstructed (r=%d), frame %d / %d', rPOD, t, Nt));
% 
%     frame = getframe(gcf);
%     [im,cm] = rgb2ind(frame2im(frame),256);
%     if t==1
%         imwrite(im,cm,fileGIF2,'gif','Loopcount',inf,'DelayTime',dtGIF);
%     else
%         imwrite(im,cm,fileGIF2,'gif','WriteMode','append','DelayTime',dtGIF);
%     end
% end
% disp(['GIF generated: ' fileGIF2]);
% 

%% Visualisation of frames 1 and last
plotFrame = @(frameIdx, filename) ...
    drawFrame(W(:,frameIdx), Ysurf, Zsurf, X1, frameIdx, Nt, filename);

plotFrame(1, 'frame1_surface.png');

plotFrame(Nt, 'frame534_surface.png');

%% Select the top four in descending order of total modal energy
modeEnergy = sum(A(1:rPOD,:).^2, 2);                 % Total energy per mode E_k
[modeEnergySorted, modeOrder] = sort(modeEnergy, 'descend');

energyPct = 100 * modeEnergySorted / sum(modeEnergy);
fprintf('POD modal energy ranking (Top %d/%d):\n', min(4,rPOD), rPOD);
for k = 1:min(4, rPOD)
    fprintf('   Rank %d -> Mode #%d, Energy share = %.2f%%\n', ...
        k, modeOrder(k), energyPct(k));
end

totalEnergy_t = sum(A(1:rPOD,:).^2, 1);             
[~, bucklingFrame] = max(diff(totalEnergy_t));       % energy transition point
fprintf('Predicted buckling occurs at frame %d (based on total energy jump)\n', bucklingFrame);

% Visualisation of the top 4 POD spatial modes
topK = min(4, rPOD);
for k = 1:topK
    idx_mode = modeOrder(k);             
    phi_k    = Phi(:, idx_mode);           
    fname    = sprintf('pod_mode_energyRank%d_id%d.png', k, idx_mode);

    drawFrame(phi_k, Ysurf, Zsurf, X1, idx_mode, rPOD, fname);
end

E_lambda_all = lambda(:);                       
E_lambda_pct = 100 * E_lambda_all / sum(E_lambda_all);
%% POD energy spectrum（Contains cumulative energy）

% The first rPOD modes
E_lambda_r   = lambda(1:rPOD);
E_lambda_r_pct = 100 * E_lambda_r / sum(lambda);

% Energy based on time factor A
E_A_r = sum(A(1:rPOD,:).^2, 2);                   % rPOD×1
E_A_r_pct = 100 * E_A_r / sum(E_A_r);


figure('Color',[1 1 1]); 
yyaxis left
bar(1:length(E_lambda_all), E_lambda_pct, 'LineWidth', 1);
xlabel('POD mode index'); ylabel('Energy fraction (%)');
title(sprintf('POD Energy Spectrum (Eigenvalue-based), Cumulative r=%d: %.2f%%', ...
    rPOD, 100*sum(lambda(1:rPOD))/sum(lambda)));
grid on; box on;

% Cumulative energy curve
cumE = 100 * cumsum(E_lambda_all) / sum(E_lambda_all);
yyaxis right
plot(1:length(E_lambda_all), cumE, '-o', 'LineWidth', 1.5); hold on;
yline(100*sum(lambda(1:rPOD))/sum(lambda), '--', ...
    sprintf(' r=%d cumulative=%.2f%%', rPOD, 100*sum(lambda(1:rPOD))/sum(lambda)));
ylabel('Cumulative energy（%）');

exportgraphics(gcf, 'pod_energy_spectrum.png', 'BackgroundColor','white');
disp('Saved: pod_energy_spectrum.png');

%%
cumE = cumsum(lambda)/sum(lambda);
r95 = find(cumE>=0.95,1);
d2 = diff(log(lambda+eps),2); [~,r_elbow] = max(-d2); r_elbow=max(r_elbow,1);
r_candidates = unique([r_elbow, r95, 4:10]);  
r_candidates


function drawFrame(W_col, Ysurf, Zsurf, X1, idx, Nt, fname)
    nrC = 180; nrL = 76;

    ww = reshape(W_col, nrC, nrL);  
    ww(nrC+1,:) = ww(1,:);         
    ww = ww';                     

    figure('Color',[1 1 1]);
    scrsz = get(groot,'ScreenSize');
    set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);

    surf(Ysurf, Zsurf, X1, ww, 'EdgeColor','None');

    colormap(jet);  
    shading interp;
    axis equal;
    axis off;
    view(45,30);

    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('z (mm)');
    title(sprintf('Frame %d / %d', idx, Nt));
    set(gca,'FontName','Arial','FontSize',16); box on;

    exportgraphics(gcf, fname, 'BackgroundColor','white');
    disp(['Saved: ' fname]);
end

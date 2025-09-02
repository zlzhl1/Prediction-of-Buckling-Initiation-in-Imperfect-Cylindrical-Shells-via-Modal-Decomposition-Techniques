clear; clc; close all;

load W.mat          
[nSpace, Nt] = size(W);
nrC   = 200;        
nrL   = 51;        
R     = 0.1;        
L     = 0.1609;     
rPOD  = 10;         % Number of retained POD modes
dtGIF = 0.05;       % GIF frame delay (s)

%% Generate static cylinder mesh
theta = 0:2*pi/nrC:2*pi;  
x     = linspace(0, L, nrL);

[T1,X1] = meshgrid(theta, x);    
Ysurf   = R * cos(T1);
Zsurf   = R * sin(T1);

%% POD decomposition & reconstruction
X        = W;                     
X_mean   = mean(X,2);
X_fluct  = X - X_mean;

C        = X_fluct' * X_fluct;
[Vec, D] = eig(C);
[lambda, idx] = sort(diag(D),'descend');
Vec      = Vec(:,idx);
Phi      = X_fluct * Vec;          % Spatial modes
Phi      = Phi ./ vecnorm(Phi);  
A        = Phi' * X_fluct;         % Temporal coefficients
X_rec    = Phi(:,1:rPOD)*A(1:rPOD,:) + X_mean;


%% ---------- 初始化 GIF ----------
fileGIF3 = 'cylinder_3d_with_buckling_marker.gif';
figure('Color',[1 1 1]);

%% ---------- 主循环 ----------
for t = 1:Nt
    %% 1. 当前帧位移场
    Wt = reshape(W(:,t), nrC, nrL);  % 200 × 51
    Wt(nrC+1,:) = Wt(1,:);           % 闭合：201 × 51
    Wt = Wt';                        % 51 × 201

    %% 2. 计算梯度和梯度模
    [dX, dTheta] = gradient(Wt);
    gradMag = sqrt(dX.^2 + dTheta.^2);

    %% 3. 找到最大梯度（预测屈曲点）
    [~, maxIdx] = max(gradMag(:));
    [rowIdx, colIdx] = ind2sub(size(gradMag), maxIdx);

    %% 4. 获取预测点空间坐标
    x_buck = X1(rowIdx, colIdx);
    y_buck = Ysurf(rowIdx, colIdx);
    z_buck = Zsurf(rowIdx, colIdx);

    %% 5. 绘图
    scrsz = get(groot,'ScreenSize');
    set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);

    surf(Ysurf, Zsurf, X1, Wt, 'EdgeColor','none');
    shading interp; colormap jet; axis equal off;
    view(-90,30);
    title(sprintf('Buckling Prediction - Frame %d / %d', t, Nt));

    % 标记 buckling 点（红色大圆点）
    hold on;
    plot3(y_buck, z_buck, x_buck, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    hold off;

    %% 6. 保存帧到 GIF
    frame = getframe(gcf);
    [im, cm] = rgb2ind(frame2im(frame), 256);
    if t == 1
        imwrite(im, cm, fileGIF3, 'gif', 'Loopcount', inf, 'DelayTime', dtGIF);
    else
        imwrite(im, cm, fileGIF3, 'gif', 'WriteMode', 'append', 'DelayTime', dtGIF);
    end
end

disp(['✓ 已生成包含屈曲预测点的 GIF：' fileGIF3]);
%%
% maxGrad = zeros(1, Nt);
for t = 1:Nt
    Wt = reshape(W(:,t), nrC, nrL);
    Wt(nrC+1,:) = Wt(1,:);
    Wt = Wt';

    [dX, dTheta] = gradient(Wt);
    gradMag = sqrt(dX.^2 + dTheta.^2);
    maxGrad(t) = max(gradMag(:));
end
%%
% 绘制最大梯度随时间变化
figure;
plot(1:Nt, maxGrad, 'b-', 'LineWidth', 2);
xlabel('Frame'); ylabel('Max Gradient');
title('Maximum Displacement Gradient over Time'); grid on;

% 找出变化最大的位置（也可用diff分析）
[~, bucklingFrame] = max(diff(maxGrad));
fprintf('✓ 预测 Buckling 起始帧：第 %d 帧\n', bucklingFrame);

modalEnergy = sum(A(1:rPOD,:).^2, 1);  % 时间方向总能量
figure;
plot(1:Nt, modalEnergy, 'LineWidth', 2);
xlabel('Frame'); ylabel('Total Modal Energy');
title('Modal Energy over Time');

[~, bucklingFrame] = max(diff(modalEnergy));
fprintf('✓ 预测 POD 屈曲起始帧：第 %d 帧\n', bucklingFrame);
%%
diffNorm = zeros(1, Nt-1);
for t = 2:Nt
    diffNorm(t-1) = norm(W(:,t) - W(:,t-1));
end
figure;
plot(2:Nt, diffNorm, 'LineWidth', 2);
xlabel('Frame'); ylabel('||ΔW||');
title('Frame-to-Frame Displacement Change');

[~, bucklingFrame] = max(diffNorm);
fprintf('✓ 预测位移突变点（屈曲）：第 %d 帧\n', bucklingFrame + 1);
%%

% POD 时间系数已存在变量 A 中 (rPOD × Nt)
modalEnergy = sum(A(1:rPOD,:).^2, 1);

% 找时间系数能量突变位置（差分最大处）
[~, bucklingFrame] = max(diff(modalEnergy));
fprintf('✓ 预测 Buckling 发生在第 %d 帧\n', bucklingFrame);

% 获取该帧的 POD 系数向量
a_buck = A(1:rPOD, bucklingFrame);

% 按模态能量大小排序
[~, modeIdx] = sort(abs(a_buck), 'descend');

topK = 3;  % 绘制前几个模态
for k = 1:topK
    idx = modeIdx(k);
    phi_k = reshape(Phi(:,idx), nrC, nrL);  % 200×51
    phi_k(nrC+1,:) = phi_k(1,:);
    phi_k = phi_k';

    figure('Color',[1 1 1]);
    scrsz = get(groot,'ScreenSize');
    set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);

    surf(Ysurf, Zsurf, X1, phi_k, 'EdgeColor', 'none');
    colormap(jet); shading interp;
    view(-90,30); axis equal off;

    title(sprintf('POD Mode %d (Rank %d in Frame %d)', idx, k, bucklingFrame));
    exportgraphics(gcf, sprintf('pod_mode_%d_frame_%d.png', idx, bucklingFrame), 'BackgroundColor','white');

    disp(['✓ 已保存 POD 模态图像: pod_mode_' num2str(idx) '_frame_' num2str(bucklingFrame) '.png']);
end

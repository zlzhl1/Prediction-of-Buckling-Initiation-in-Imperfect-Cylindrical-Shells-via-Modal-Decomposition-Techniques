clear; clc; close all;

load W2.mat         
[nSpace, Nt] = size(W2);
nrC   = 200;     
nrL   = 51;        
R     = 0.1;       
L     = 0.1609;   
rPOD  = 10;        
dtGIF = 0.05;      

%% Generate cylindrical static meshes
theta = 0:2*pi/nrC:2*pi;  
x     = linspace(0, L, nrL);

[T1,X1] = meshgrid(theta, x);     
Ysurf   = R * cos(T1);
Zsurf   = R * sin(T1);

%% POD Decomposition & Reconstruction
X        = W2;                     
X_mean   = mean(X,2);
X_fluct  = X - X_mean;

C        = X_fluct' * X_fluct;
[Vec, D] = eig(C);
[lambda, idx] = sort(diag(D),'descend');
Vec      = Vec(:,idx);
Phi      = X_fluct * Vec;          % spatial modes
Phi      = Phi ./ vecnorm(Phi);    
A        = Phi' * X_fluct;         % time factor
X_rec    = Phi(:,1:rPOD)*A(1:rPOD,:) + X_mean;

%% Create GIF original data
fileGIF1 = 'cylinder_3d_original.gif';
figure('Color',[1 1 1]);
for t = 1:Nt
    ww = reshape(W2(:,t), nrC, nrL); 
    ww(nrC+1,:) = ww(1,:);         
    ww = ww';                       
    scrsz = get(groot,'ScreenSize');
    set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);
    surf(Ysurf, Zsurf, X1, ww, 'EdgeColor','none');
    colormap jet; shading interp; axis equal off;

    view(-90,30);                 
    title(sprintf('Original, frame %d / %d', t, Nt));
    
    frame = getframe(gcf);
    [im,cm] = rgb2ind(frame2im(frame),256);
    if t==1
        imwrite(im,cm,fileGIF1,'gif','Loopcount',inf,'DelayTime',dtGIF);
    else
        imwrite(im,cm,fileGIF1,'gif','WriteMode','append','DelayTime',dtGIF);
    end
end
disp(['Generate GIF: ' fileGIF1]);

%% Create GIF : POD reconstruction
fileGIF2 = sprintf('cylinder_3d_reconstructed_r%d.gif', rPOD);
figure('Color',[1 1 1]);
for t = 1:Nt
    ww = reshape(W2(:,t), nrC, nrL);  
    ww(nrC+1,:) = ww(1,:);         
    ww = ww';                      
    scrsz = get(groot,'ScreenSize');
    set(gcf,'Position',[scrsz(3)/20 scrsz(4)/5 700 600]);
    surf(Ysurf, Zsurf, X1, ww, 'EdgeColor','none');
    colormap jet; shading interp; axis equal off;

    view(-90,30);
    title(sprintf('Reconstructed (r=%d), frame %d / %d', rPOD, t, Nt));
    
    frame = getframe(gcf);
    [im,cm] = rgb2ind(frame2im(frame),256);
    if t==1
        imwrite(im,cm,fileGIF2,'gif','Loopcount',inf,'DelayTime',dtGIF);
    else
        imwrite(im,cm,fileGIF2,'gif','WriteMode','append','DelayTime',dtGIF);
    end
end
disp(['Generate GIF: ' fileGIF2]);


%% Visualisation Frame 1 and last

plotFrame = @(frameIdx, filename) ...
    drawFrame(W2(:,frameIdx), Ysurf, Zsurf, X1, frameIdx, Nt, filename);

plotFrame(1, 'frame1_surface.png');

plotFrame(148, 'frame148_surface.png');

%%

modalEnergy = sum(A(1:rPOD,:).^2, 1);

% Find the location of the time coefficient energy mutation 
[~, bucklingFrame] = max(diff(modalEnergy));
fprintf('✓ Buckling is predicted to occur at frame %d\n', bucklingFrame);


a_buck = A(1:rPOD, bucklingFrame);

% Sort by mode energy
[~, modeIdx] = sort(abs(a_buck), 'descend');

topK = 4;  % Plotting the first few modes
for k = 1:topK
    idx_mode = modeIdx(k);               
    phi_k = Phi(:, idx_mode);              
    fname = sprintf('pod_mode_%d_frame_%d.png', idx_mode, bucklingFrame);

    drawFrame(phi_k, Ysurf, Zsurf, X1, idx_mode, rPOD, fname);
end

function drawFrame(W_col, Ysurf, Zsurf, X1, idx, Nt, fname)
    nrC = 200; nrL = 51;

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
    view(-90,30);

    xlabel('x (mm)');
    ylabel('y (mm)');
    zlabel('z (mm)');
    title(sprintf('Frame %d / %d', idx, Nt));
    set(gca,'FontName','Arial','FontSize',16); box on;

    exportgraphics(gcf, fname, 'BackgroundColor','white');
    disp(['saved: ' fname]);
end

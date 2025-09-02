clear; clc; close all;

load W.mat         
load t.mat          
[nSpace, Nt] = size(W);
nrC   = 200;       
nrL   = 51;         
R     = 0.1;     
L     = 0.1609;    
rPOD  = 10;         
dtGIF = 0.05;      

t = t(:);                                
frames_to_plot = find(t >= (t(end) - 1));
Nt_plot = length(frames_to_plot);        

%% Generate cylindrical static meshes
theta = linspace(0, 2*pi, nrC+1);  
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
Phi      = X_fluct * Vec;
Phi      = Phi ./ vecnorm(Phi);
A        = Phi' * X_fluct;
X_rec    = Phi(:,1:rPOD)*A(1:rPOD,:) + X_mean;
%% Create GIF
vmax = max(abs(X(:)));
vmin = -vmax;

toGrid = @(v) reshape(v, nrC, nrL)';  

%% original data
fileGIF1 = 'cylinder_3d_original_last1s.gif';
figure('Visible','off');
for idx = 1:Nt_plot
    t_idx = frames_to_plot(idx);
    wGrid = toGrid(X(:,t_idx));
    wGrid(:,end+1) = wGrid(:,1);  

    surf(Ysurf, Zsurf, X1, wGrid, 'EdgeColor','none');
    colormap jet; shading interp; axis equal off;
    caxis([vmin vmax]);
    view(-90,30);
    title(sprintf('Original, frame %d / %d', t_idx, Nt));

    frame = getframe(gcf);
    [im,cm] = rgb2ind(frame2im(frame),256);
    if idx == 1
        imwrite(im,cm,fileGIF1,'gif','Loopcount',inf,'DelayTime',dtGIF);
    else
        imwrite(im,cm,fileGIF1,'gif','WriteMode','append','DelayTime',dtGIF);
    end
end
disp(['Generate GIF: ' fileGIF1]);

%% Create GIF : POD reconstruction
fileGIF2 = sprintf('cylinder_3d_reconstructed_r%d_last1s.gif', rPOD);
figure('Visible','off');
for idx = 1:Nt_plot
    t_idx = frames_to_plot(idx);
    wGrid = toGrid(X_rec(:,t_idx));
    wGrid(:,end+1) = wGrid(:,1);

    surf(Ysurf, Zsurf, X1, wGrid, 'EdgeColor','none');
    colormap jet; shading interp; axis equal off;
    caxis([vmin vmax]);
    view(-90,30);
    title(sprintf('Reconstructed (r=%d), frame %d / %d', rPOD, t_idx, Nt));

    frame = getframe(gcf);
    [im,cm] = rgb2ind(frame2im(frame),256);
    if idx == 1
        imwrite(im,cm,fileGIF2,'gif','Loopcount',inf,'DelayTime',dtGIF);
    else
        imwrite(im,cm,fileGIF2,'gif','WriteMode','append','DelayTime',dtGIF);
    end
end
disp(['Generate GIF: ' fileGIF2]);

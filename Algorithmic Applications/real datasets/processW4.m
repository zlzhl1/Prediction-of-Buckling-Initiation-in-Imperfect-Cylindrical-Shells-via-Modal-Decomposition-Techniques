
load('W.mat'); 

W_temp = W(:, 1:450);        
save('W_temp.mat', 'W_temp');

fprintf('W original size:[%d × %d]\n', size(W,1), size(W,2));
fprintf('W_temp Build completed，size:[%d × %d]\n', size(W_temp,1), size(W_temp,2));

W_block = [W_temp, fliplr(W_temp)];

nRepeat = 2;

W4 = repmat(W_block, 1, nRepeat);  

save('W4.mat', 'W4');

fprintf('W original size:[%d × %d]\n', size(W,1), size(W,2));
fprintf('W4 Build completed，size:[%d × %d]\n', size(W4,1), size(W4,2));


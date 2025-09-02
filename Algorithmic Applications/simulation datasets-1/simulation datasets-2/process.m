load('W.mat');  

W_block = [W, fliplr(W)];

nRepeat = 2;

W3 = repmat(W_block, 1, nRepeat); 

save('W3.mat', 'W3');

fprintf('W original size:[%d × %d]\n', size(W,1), size(W,2));
fprintf('W3 Build completed，size:[%d × %d]\n', size(W3,1), size(W3,2));


function niqes = computequality_batch(matcachepath,blocksizerow,blocksizecol,...
    blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam)
%COMPUTEQUALITY_DISK Summary of this function goes here
%   Detailed explanation goes here

load(matcachepath);

n = size(imgs,1);
niqes = [n];

for i = 1:n
    img = imgs(i,:,:,:);
    img = squeeze(img);
    niqes(i) = computequality(img,blocksizerow,blocksizecol,...
        blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
end
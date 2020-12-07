load("niqe_release/modelparameters.mat")
blocksizerow = 96;
blocksizecol = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

tic;
%imgs = zeros(2,210,373,3);
%imgs(1,:,:,:) = imread('test1.png');
%imgs(2,:,:,:) = imread('test2.png');
%img1 = imread('test1.png');
%img2 = imread('test2.png');
%imgs = cat(4, img1, img2);
%size(imgs)
%imgs = reshape(imgs,[2,210,373,3]);
%size(imgs)
load("../imgs.mat");
niqes = computequality_batch(imgs, mu_prisparam, cov_prisparam);
toc
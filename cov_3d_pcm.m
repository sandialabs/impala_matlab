function out = cov_3d_pcm(arr, mean)
% Covariance array from 3d Array (with pre-computed mean):
% arr = 3d Array (nSamp x nTemp x nCol)
% mean = 2d Array (nTemp x nCol)
% out = 3d Array (nTemp x nCol x nCol)

N = size(arr,1);
if ndims(arr) == 3
    meantmp = repmat(mean,1,1,N);
    meantmp = permute(meantmp,[3,1,2]);
elseif ismatrix(arr)
    meantmp = repmat(mean,N,1);
end
out = tensorproduct('ijl' , arr-meantmp , 'kij' , arr-meantmp , 'kil') ./ (N-1);

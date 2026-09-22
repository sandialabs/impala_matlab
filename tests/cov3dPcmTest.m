function tests = cov3dPcmTest
% Tests for cov_3d_pcm: batched covariance across the first (sample)
% dimension of a 3D array, using a precomputed mean.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testMatchesBuiltinCovPerPage(tc)
% For each temperature slice, the result must match MATLAB's cov().
rng(1);
nSamp = 200; nTemp = 3; nCol = 4;
arr = randn(nSamp, nTemp, nCol);
% mean over samples -> nTemp x nCol
m = squeeze(mean(arr, 1));
out = cov_3d_pcm(arr, m);           % nTemp x nCol x nCol
tc.verifySize(out, [nTemp, nCol, nCol]);
for t = 1:nTemp
    slice = squeeze(arr(:, t, :));  % nSamp x nCol
    expected = cov(slice);          % normalized by (N-1)
    got = squeeze(out(t, :, :));
    tc.verifyEqual(got, expected, 'AbsTol', 1e-8);
end
end

function testSymmetricOutput(tc)
rng(2);
arr = randn(100, 2, 3);
m = squeeze(mean(arr, 1));
out = cov_3d_pcm(arr, m);
for t = 1:2
    slice = squeeze(out(t, :, :));
    tc.verifyEqual(slice, slice', 'AbsTol', 1e-9);
end
end

function testDiagonalIsVariance(tc)
rng(3);
nSamp = 300;
arr = randn(nSamp, 1, 3);
m = squeeze(mean(arr, 1))';   % 1 x 3
out = cov_3d_pcm(arr, m);
got = squeeze(out(1, :, :));
expected = var(squeeze(arr));
tc.verifyEqual(diag(got)', expected, 'AbsTol', 1e-8);
end

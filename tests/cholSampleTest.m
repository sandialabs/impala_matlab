function tests = cholSampleTest
% Tests for chol_sample: draw a multivariate-normal sample via the lower
% Cholesky factor. Stochastic, so we test structure and statistics.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testOutputSizeMatchesMean(tc)
mu = [1; 2; 3];
C = eye(3);
out = chol_sample(mu, C);
tc.verifySize(out, size(mu));
end

function testNearZeroCovarianceReturnsApproxMean(tc)
% A tiny (but positive-definite) covariance yields a draw essentially
% equal to the mean. Note: chol() requires positive definiteness, so an
% exactly-zero covariance is not a supported input.
mu = [5; -2; 0.5];
C = 1e-12 * eye(3);
out = chol_sample(mu, C);
tc.verifyEqual(out, mu, 'AbsTol', 1e-4);
end

function testEmpiricalMeanApproachesMu(tc)
rng(101);
mu = [10; -4];
C = [2 0.5; 0.5 1];
N = 20000;
draws = zeros(2, N);
for i = 1:N
    draws(:, i) = chol_sample(mu, C);
end
tc.verifyEqual(mean(draws, 2), mu, 'AbsTol', 0.1);
end

function testEmpiricalCovarianceApproachesC(tc)
rng(202);
mu = [0; 0];
C = [4 -1; -1 3];
N = 50000;
draws = zeros(2, N);
for i = 1:N
    draws(:, i) = chol_sample(mu, C);
end
tc.verifyEqual(cov(draws'), C, 'AbsTol', 0.15);
end

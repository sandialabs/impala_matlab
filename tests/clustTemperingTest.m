function tests = clustTemperingTest
% Tests that tempering leaves the cold-chain posterior unchanged: the
% Escobar & West eta update, the s2 swap kernel, pooled s2 Gibbs with
% tempering, and clustered calibration with a flat likelihood.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testSampleEtaPreservesPrior(tc)
% Geweke-style check of the Escobar & West (1995) update: if
% eta ~ Gamma(a, b) and k | eta is the number of clusters of a CRP(eta)
% partition of n items, one update must return eta ~ Gamma(a, b) again.
rng(0);
a = 2; b = 2; n = 8; N = 400000;
eta = gamrnd(a, 1/b, N, 1);
i = 1:(n-1);
k = 1 + sum(rand(N, n-1) < eta ./ (eta + i), 2);
eta_new = sample_eta(eta, k, n, a, b);
tc.verifyLessThan(abs(mean(eta_new) - a/b), 0.01);
tc.verifyLessThan(abs(var(eta_new) - a/b^2), 0.02);
[~, pval] = kstest(eta_new(1:20000), 'CDF', makedist('Gamma', 'a', a, 'b', 1/b));
tc.verifyGreaterThan(pval, 1e-3);
end

function testS2SwapKernMatchesUpdate(tc)
[f, x, bounds] = calibToyProblem();
modes = {'gibbs', 'gibbs_trunc', 'MH'};
expect = {@ldig_kern, @ldig_kern, @ldhc_kern};
for k = 1:3
    s = CalibSetup(bounds);
    s.addVecExperiments(f([0.5 1]), ModelF(f, {'a','b'}, NaN, modes{k}), 0.1, 0, ...
        ones(size(x)), NaN, NaN, NaN, 1, 0.01, 1);
    tc.verifyEqual(func2str(s2_swap_kern(s, 1)), func2str(expect{k}), modes{k});
end
% with s2_df = 0 the Gibbs update targets p(s2) ~ 1/s2
xx = [0.5 2];
tc.verifyEqual(ldig_kern(xx, 0, 0), -log(xx), 'AbsTol', 1e-14);
end

function testPooledGibbsS2WithTempering(tc)
% s2 Gibbs with s2_df = 0 and a model that does not depend on theta:
% s2 | y ~ InvGamma(n/2, SS/2) exactly, with or without tempering.
rng(123);
n = 20;
y = 0.5*randn(1, n);
a = n/2; b = sum(y.^2)/2;
rng(1);
bounds = struct('a', [0 1]);
s = CalibSetup(bounds);
s.addVecExperiments(y, ModelF(@(X) zeros(size(X,1), n), {'a'}, NaN, 'gibbs'), 1, 0, ones(1, n));
s.setTemperatureLadder(2.^(0:3), 100);
s.setMCMC(12000, 0, 1, 100);
evalc('o = calibPool(s);');
s2 = o.s2{1}(2001:end, 1, 1);
tc.verifyLessThan(abs(mean(s2) / (b/(a-1)) - 1), 0.015);
% median of InvGamma(a, b) is b / gammaincinv(0.5, a, 'upper')
tc.verifyLessThan(abs(median(s2) / (b / gammaincinv(0.5, a, 'upper')) - 1), 0.015);
end

function testPooledMHS2WithTempering(tc)
% s2 by Metropolis-Hastings on log(s2) with s2_df = 0 (kernel 1/(1 + s2))
% and a model that does not depend on theta. The Jacobian of the log
% transform must not be tempered, otherwise hot chains target a different
% family than the swap ratio assumes and the cold chain is biased.
rng(123);
n = 20;
y = 0.5*randn(1, n);
SS = sum(y.^2);
unnorm = @(s2) exp(-n/2*log(s2) - SS/2./s2 - log1p(s2) + 20);
Z = integral(unnorm, 1e-4, 50);
exact = integral(@(s) s .* unnorm(s), 1e-4, 50) / Z;
rng(2);
s = CalibSetup(struct('a', [0 1]));
s.addVecExperiments(y, ModelF(@(X) zeros(size(X,1), n), {'a'}, NaN, 'MH'), 0.5, 0, ones(1, n));
s.setTemperatureLadder(2.^(0:3), 200);
s.setMCMC(20000, 0, 1, 100, 1e-8, 0, 0.1);
evalc('o = calibPool(s);');
s2 = o.s2{1}(3001:end, 1, 1);
tc.verifyLessThan(abs(mean(s2) / exact - 1), 0.025);
end

function testClustTemperingFlatLikelihood(tc)
% With a flat likelihood every chain's target is the clustering prior, so
% the cold chain must reproduce the Gamma(2, 2) prior on eta (mean 1).
rng(1);
p = 2; nobs = 4; ntheta = 8;
L = nobs * ntheta;
bounds = struct('a', [0 1], 'b', [0 1]);
s = CalibSetup(bounds);
ind = repelem(1:ntheta, nobs);
s.addVecExperiments(zeros(1, L), ModelF(@(X) zeros(size(X,1), L), {'a','b'}, NaN, 'fix'), ...
    ones(1, ntheta), zeros(1, ntheta), ind, NaN, ind);
s.setTemperatureLadder(1.5.^(0:2), 100);
s.setMCMC(5000, 0, 1, 100);
s.setHierPriors(repmat(0.5, 1, p), eye(p)*0.2^2, p + 4, eye(p)*0.3^2);
s.setClusterPriors(8, 2, 2);
evalc('o = calibClust(s);');
eta = o.eta(1001:end, 1);
tc.verifyLessThan(abs(mean(eta) - 1), 0.12);
end

function tests = essTest
% Tests for ESS (Effective Sample Size) and mcmcgr (Gelman-Rubin PSRF).
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

% ---- ESS -----------------------------------------------------------------

function testEssIidSampleNearN(tc)
% For (near) independent samples, tau ~ 1 and ESS ~ number of samples.
rng(5);
nr = 4000;
sample = randn(nr, 1);
[ess, tau] = ESS(sample);
tc.verifyGreaterThan(tau, 0);
% Independent chain: ESS should be a large fraction of nr.
tc.verifyGreaterThan(ess, 0.5 * nr);
end

function testEssCorrelatedLessThanN(tc)
% Strongly autocorrelated series -> tau > 1 and ESS < nr.
rng(9);
nr = 4000;
e = randn(nr, 1);
x = filter(1, [1 -0.9], e);   % AR(1), phi = 0.9
[ess, tau] = ESS(x);
tc.verifyGreaterThan(tau, 1);
tc.verifyLessThan(ess, nr);
end

function testEssMultipleColumns(tc)
% One tau/ess per parameter column.
rng(2);
sample = randn(2000, 3);
[ess, tau] = ESS(sample);
tc.verifySize(ess, [1 3]);
tc.verifySize(tau, [1 3]);
end

function testEssAggregateOutputs(tc)
rng(4);
sample = randn(2000, 2);
[~, ~, ess_all, tau_all] = ESS(sample);
tc.verifyGreaterThan(ess_all, 0);
tc.verifyGreaterThan(tau_all, 0);
end

% ---- mcmcgr --------------------------------------------------------------

function testMcmcgrConvergedNearOne(tc)
% Well-mixed independent chains from the same target -> R close to 1.
rng(21);
nsimu = 5000; npar = 2; nchain = 4;
chain = randn(nsimu, npar, nchain);
R = mcmcgr(chain);
tc.verifySize(R, [1 npar]);
tc.verifyEqual(R, ones(1, npar), 'AbsTol', 0.15);
end

function testMcmcgrDivergentChainsLargeR(tc)
% Chains centered at very different means -> R noticeably above 1.
rng(22);
nsimu = 3000; nchain = 3;
chain = zeros(nsimu, 1, nchain);
for i = 1:nchain
    chain(:, 1, i) = randn(nsimu, 1) + (i - 1) * 20;  % big offsets
end
R = mcmcgr(chain);
tc.verifyGreaterThan(R, 1.5);
end

function testMcmcgrCellArrayInput(tc)
% Accepts a cell array of chains.
rng(23);
chain = {randn(2000, 2), randn(2000, 2)};
R = mcmcgr(chain);
tc.verifySize(R, [1 2]);
tc.verifyTrue(all(isfinite(R)));
end

function testMcmcgrRequiresTwoChains(tc)
% A single chain must error.
tc.verifyError(@() mcmcgr(randn(100, 2, 1)), ?MException);
end

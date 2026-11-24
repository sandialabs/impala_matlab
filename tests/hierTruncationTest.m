function tests = hierTruncationTest
% Tests for the hierarchical truncation diagnostic
% (hier_truncation_weights, log_box_prob, weighted_quantile).
% Port of the python package's tests/test_hier_truncation.py.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function s = make_setup(p, nexp, constraint_func)
% A hierarchical setup; no MCMC is run, only the setup fields are used.
arguments
    p = 2;
    nexp = 3;
    constraint_func = @cf_bounds;
end
nobs = 4;
names = arrayfun(@(j) sprintf('x%d', j), 1:p, 'UniformOutput', false);
bounds = struct();
for j = 1:p
    bounds.(names{j}) = [0 1];
end
s = CalibSetup(bounds, constraint_func);
ind = repelem(1:nexp, nobs);
s.addVecExperiments(zeros(1, nobs*nexp), ...
    ModelF(@(X) zeros(size(X,1), nobs*nexp), names, NaN, 'fix'), ...
    ones(1, nexp), zeros(1, nexp), ind, NaN, ind);
end

function out = make_out(th0, s0diag, ntemps)
% calibHier-shaped output holding the given cold-chain draws. The hot
% chain is filled with the identity so that picking the wrong temperature
% would change the answer.
arguments
    th0
    s0diag
    ntemps = 2;
end
[nd, p] = size(th0);
out.theta0 = zeros(nd, ntemps, p);
out.theta0(:,1,:) = th0;
out.Sigma0 = zeros(nd, ntemps, p, p);
for j = 1:nd
    out.Sigma0(j,1,:,:) = diag(s0diag(j,:));
    out.Sigma0(j,2,:,:) = eye(p);
end
end

function testWeightsMatchClosedForm(tc)
% For a diagonal Sigma0 the box probability is a product of normal CDFs,
% so the weights can be checked against the formula directly.
s = make_setup(2, 3);
n = sum(s.ntheta);
tc.verifyEqual(n, 3);
rng(0);
nd = 50;
th0 = 0.1 + 0.8*rand(nd, 2);
s0 = (0.02 + 0.48*rand(nd, 2)).^2;
res = hier_truncation_weights(s, make_out(th0, s0), 1:nd);

sd = sqrt(s0);
lp = sum(log(normcdf((1 - th0)./sd) - normcdf(-th0./sd)), 2);
lw = -n * lp;
expect = exp(lw - max(lw));
expect = expect / sum(expect);
tc.verifyEqual(res.weights, expect, 'AbsTol', 1e-12);
tc.verifyEqual(res.log_box_prob, lp, 'AbsTol', 1e-12);
tc.verifyEqual(res.n_theta, n);
end

function testFullCovariancePathAgreesWithDiagonal(tc)
% The mvncdf path and the diagonal shortcut must agree when the covariance
% is diagonal but stored with explicit (near-zero) off-diagonal entries.
mu = [0.3 0.6];
cov = diag([0.04 0.09]);
diag_val = log_box_prob(mu, cov);
nudged = cov + [0 1e-9; 1e-9 0];
tc.verifyEqual(log_box_prob(mu, nudged), diag_val, 'AbsTol', 1e-6);
end

function testColdChainIsUsedByDefault(tc)
% make_out puts the identity in the hot chain, so reading the wrong
% temperature would give a different box probability.
s = make_setup(2, 3);
nd = 20;
th0 = repmat(0.5, nd, 2);
s0 = repmat(0.01^2, nd, 2);
res = hier_truncation_weights(s, make_out(th0, s0), 1:nd);
hot = hier_truncation_weights(s, make_out(th0, s0), 1:nd, 2);
tc.verifyEqual(res.log_box_prob, zeros(nd,1), 'AbsTol', 1e-12);
tc.verifyLessThan(max(hot.log_box_prob), -0.1);
end

function testNoTiltWhenBoundsAreSlack(tc)
% theta0 mid-box with a tiny Sigma0 keeps essentially all of the normal
% mass inside the bounds, so the weights are flat and nothing is corrected.
s = make_setup(2, 5);
rng(1);
nd = 200;
th0 = 0.5 + 0.01*randn(nd, 2);
s0 = repmat(0.01^2, nd, 2);
res = hier_truncation_weights(s, make_out(th0, s0), 1:nd);
tc.verifyGreaterThan(res.ess_frac, 0.99);
tc.verifyLessThan(res.tilt_nats, 1e-6);
tc.verifyEqual(res.theta0_sd_reweighted, res.theta0_sd_raw, 'RelTol', 1e-3);
end

function testReweightingWidensSigma0WhenBoundsBind(tc)
% With the bounds binding, the correction must push Sigma0 UP -- that is
% the contraction it exists to undo.
s = make_setup(1, 6);
rng(2);
nd = 4000;
th0 = 0.2 + 0.6*rand(nd, 1);
s0 = (0.05 + 0.55*rand(nd, 1)).^2;
res = hier_truncation_weights(s, make_out(th0, s0), 1:nd);
tc.verifyGreaterThan(res.tilt_nats, 0.1);
tc.verifyGreaterThan(res.sigma0_diag_median_reweighted(1), ...
                     res.sigma0_diag_median_raw(1));
end

function testDefaultsToSecondHalfOfChain(tc)
s = make_setup(1, 2);
nd = 100;
th0 = repmat(0.5, nd, 1);
s0 = repmat(0.01, nd, 1);
res = hier_truncation_weights(s, make_out(th0, s0));
tc.verifyLength(res.weights, nd/2);
end

function testCustomConstraintFuncWarns(tc)
s = make_setup(1, 1, @(x, bounds) true);
out = make_out(repmat(0.5, 10, 1), repmat(0.04, 10, 1));
tc.verifyWarning(@() hier_truncation_weights(s, out, 1:10), ...
                 'hier_truncation_weights:customConstraint');
end

function testUnusableDrawsAreDroppedWithWarning(tc)
% A non-positive-definite Sigma0 draw cannot give a box probability; it is
% dropped rather than poisoning the whole diagnostic.
s = make_setup(2, 2);
nd = 10;
th0 = repmat(0.5, nd, 2);
out = make_out(th0, repmat(0.04, nd, 2));
out.Sigma0(3,1,:,:) = [1 2; 2 1];  % indefinite, and not diagonal
res = tc.verifyWarning(@() hier_truncation_weights(s, out, 1:nd), ...
                       'hier_truncation_weights:unusableDraws');
tc.verifyLength(res.weights, nd - 1);
tc.verifyTrue(all(isfinite(res.log_box_prob)));
end

function testReweightingRecoversHyperpriorOnAFlatRun(tc)
% End-to-end: with a flat likelihood the cold chain should sample the
% hierarchical prior, but the bounds truncation contracts it -- sd(theta0)
% comes out ~20% below its 0.2 prior. Reweighting must put it back.
p = 2; nexp = 6; nobs = 5;
bounds = struct('a', [0 1], 'b', [0 1]);
s = CalibSetup(bounds);
ind = repelem(1:nexp, nobs);
s.addVecExperiments(zeros(1, nobs*nexp), ...
    ModelF(@(X) zeros(size(X,1), nobs*nexp), {'a','b'}, NaN, 'fix'), ...
    ones(1, nexp), zeros(1, nexp), ind, NaN, ind);
s.setMCMC(12000, 0, 1, 100);
s.setHierPriors(repmat(0.5, 1, p), eye(p)*0.2^2, p + 4, eye(p)*0.3^2);
rng(11);
evalc('o = calibHier(s);');
res = hier_truncation_weights(s, o, 2001:12000);

tc.verifyEqual(res.n_theta, nexp);
tc.verifyGreaterThan(res.tilt_nats, 0.05);
% raw is contracted, reweighted is close to the 0.2 prior
tc.verifyLessThan(res.theta0_sd_raw, 0.18);
tc.verifyLessThan(abs(res.theta0_sd_reweighted - 0.2), 0.025);
tc.verifyGreaterThan(res.theta0_sd_reweighted, res.theta0_sd_raw);
% and Sigma0 widens toward its own prior median (~0.0206)
tc.verifyGreaterThan(res.sigma0_diag_median_reweighted, res.sigma0_diag_median_raw);
tc.verifyLessThan(abs(res.sigma0_diag_median_reweighted - 0.0206), 0.004);
end

function testWeightedQuantileMatchesUnweighted(tc)
x = [5 1 3 2 4];
w = repmat(0.2, 1, 5);
tc.verifyEqual(weighted_quantile(x, 0.5, w), median(x), 'AbsTol', 1e-12);
end

function testWeightedQuantileShiftsWithWeights(tc)
% Putting all the weight on one value returns that value.
x = [1 2 3 4 5];
w = [0 0 0 0 1];
tc.verifyEqual(weighted_quantile(x, 0.5, w), 5, 'AbsTol', 1e-12);
tc.verifyEqual(weighted_quantile(x, [0.1 0.9], [1 1 1 1 1]), ...
               weighted_quantile(x, [0.1 0.9], repmat(0.2, 1, 5)), 'AbsTol', 1e-12);
end

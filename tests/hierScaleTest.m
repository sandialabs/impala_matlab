function tests = hierScaleTest
% Tests for the probit-scale hierarchy of calibHier
% (setHierPriors(..., 'probit')) and CalibSetup.sample_hier_parent.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function s = flat_setup(scale, nmcmc, ntemps)
% Hierarchical setup whose likelihood does not depend on theta, so the
% cold chain should sample the prior of the hierarchical model exactly.
p = 2; nexp = 6; nobs = 5;
bounds = struct('a', [0 1], 'b', [0 1]);
s = CalibSetup(bounds);
ind = repelem(1:nexp, nobs);
s.addVecExperiments(zeros(1, nobs*nexp), ModelF(@(X) zeros(size(X,1), nobs*nexp), {'a','b'}, NaN, 'fix'), ...
    ones(1, nexp), zeros(1, nexp), ind, NaN, ind);
s.setTemperatureLadder(1.5.^(0:ntemps-1), 100);
s.setMCMC(nmcmc, 0, 1, 100);
if strcmp(scale, 'probit')
    mean0 = 0;
else
    mean0 = 0.5;
end
s.setHierPriors(repmat(mean0, 1, p), eye(p)*0.2^2, p + 4, eye(p)*0.3^2, scale);
end

function testBadScale(tc)
s = CalibSetup(struct('a', [0 1]));
tc.verifyError(@() s.setHierPriors(0, 1, 3, 1, 'logit'), 'CalibSetup:setHierPriors:badScale');
end

function testProbitHierarchyRecoversPrior(tc)
% On the probit scale theta_i ~ N(theta0, Sigma0) is not truncated, so
% with a flat likelihood theta0 must follow its N(0, 0.2^2 I) prior. (On
% the unit scale sd(theta0) comes out ~20% too small because the bounds
% tilt the effective hyperprior.)
for ntemps = [1 3]
    rng(11);
    s = flat_setup('probit', 12000, ntemps);
    evalc('o = calibHier(s);');
    th0 = reshape(o.theta0(2001:end, 1, :), [], 2);
    lbl = sprintf('ntemps=%d', ntemps);
    tc.verifyTrue(all(th0(:) > 0 & th0(:) < 1), lbl);  % returned on the 0-1 scale
    z0 = probit(clip_unit(th0));
    % batch-means Monte Carlo standard error of the mean
    nb = 20;
    nn = floor(size(z0, 1) / nb) * nb;
    batches = squeeze(mean(reshape(z0(1:nn,:), [], nb, 2), 1));
    mcse = std(batches, 0, 1) / sqrt(nb);
    tc.verifyLessThan(abs(mean(z0, 1)), 4*mcse + 0.01, lbl);
    tc.verifyLessThan(abs(std(z0, 0, 1) / 0.2 - 1), 0.12 * [1 1], lbl);
    th = o.theta{1}(2001:end, 1, :, :);
    tc.verifyTrue(all(th(:) > 0 & th(:) < 1), lbl);

    S0 = reshape(o.Sigma0(2001:end, 1, :, :), [], 2, 2);
    parent = s.sample_hier_parent(th0, S0);
    tc.verifySize(parent, size(th0), lbl);
    tc.verifyTrue(all(parent(:) > 0 & parent(:) < 1), lbl);
end
end

function testUnitScaleParentUnchanged(tc)
% sample_hier_parent on the unit scale is the old constrained sampler.
s = flat_setup('unit', 10, 1);
th0 = repmat(0.5, 50, 2);
S0 = repmat(reshape(0.01*eye(2), 1, 2, 2), 50, 1, 1);
rng(5);
a = s.sample_hier_parent(th0, S0);
rng(5);
b = sample_constrained(th0, S0, s);
tc.verifyEqual(a, b);
end

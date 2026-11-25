function tests = hierParentTest
% Tests for CalibSetup.sample_hier_parent, which draws parameters for a new
% experiment from the fitted hierarchical distribution.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function s = flat_setup()
% Hierarchical setup whose likelihood does not depend on theta.
p = 2; nexp = 6; nobs = 5;
bounds = struct('a', [0 1], 'b', [0 1]);
s = CalibSetup(bounds);
ind = repelem(1:nexp, nobs);
s.addVecExperiments(zeros(1, nobs*nexp), ModelF(@(X) zeros(size(X,1), nobs*nexp), {'a','b'}, NaN, 'fix'), ...
    ones(1, nexp), zeros(1, nexp), ind, NaN, ind);
s.setMCMC(10, 0, 1, 100);
s.setHierPriors(repmat(0.5, 1, p), eye(p)*0.2^2, p + 4, eye(p)*0.3^2);
end

function testParentIsConstrainedSampler(tc)
% sample_hier_parent is the constrained normal sampler at (theta0, Sigma0).
s = flat_setup();
th0 = repmat(0.5, 50, 2);
S0 = repmat(reshape(0.01*eye(2), 1, 2, 2), 50, 1, 1);
rng(5);
a = s.sample_hier_parent(th0, S0);
rng(5);
b = sample_constrained(th0, S0, s);
tc.verifyEqual(a, b);
end

function testParentRespectsBounds(tc)
% Draws from a wide Sigma0 are redrawn until they lie inside the bounds.
s = flat_setup();
th0 = repmat(0.5, 200, 2);
S0 = repmat(reshape(0.25*eye(2), 1, 2, 2), 200, 1, 1);
rng(7);
parent = s.sample_hier_parent(th0, S0);
tc.verifySize(parent, size(th0));
tc.verifyTrue(all(parent(:) > 0 & parent(:) < 1));
end

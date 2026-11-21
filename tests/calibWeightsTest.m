function tests = calibWeightsTest
% Tests for observation weights (wt in CalibSetup.addVecExperiments).
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testIntegerWeightEqualsReplication(tc)
% A weight of 2 on an observation gives the same log likelihood as
% including that observation twice.
mdl = ModelF(@(X) X, {'a'});
y = [0.3 1.2 -0.5]; pred = [0.1 1.0 0.0]; s2 = [0.2 0.2 0.5];
wt = [2 1 3];
cw = mdl.lik_cov_inv(s2, 1:3, wt);
lw = mdl.llik(y, pred, cw, wt);
rep = [1 1 2 3 3 3];
cr = mdl.lik_cov_inv(s2(rep));
lr = mdl.llik(y(rep), pred(rep), cr);
tc.verifyEqual(lw, lr, 'AbsTol', 1e-12);
end

function testUnitWeightsMatchDefault(tc)
mdl = ModelF(@(X) X, {'a'});
y = randn(1,5); pred = randn(1,5); s2 = rand(1,5) + 0.1;
c1 = mdl.lik_cov_inv(s2);
c2 = mdl.lik_cov_inv(s2, 1:5, ones(1,5));
tc.verifyEqual(mdl.llik(y, pred, c1), mdl.llik(y, pred, c2, ones(1,5)), 'AbsTol', 1e-12);
end

function testWeightedErrorGroupCounts(tc)
[f, x, bounds] = calibToyProblem();
n = numel(x);
s = CalibSetup(bounds);
s2_ind = [ones(1,10) 2*ones(1,n-10)];
wt = [0.5*ones(1,10) 2*ones(1,n-10)];
s.addVecExperiments(f([0.5 1]), ModelF(f, {'a','b'}), [0.1 0.1], 5, s2_ind, ...
    NaN, NaN, NaN, 1, [], [], wt);
tc.verifyEqual(s.ny_s2{1}, [5 2*(n-10)]);
tc.verifyEqual(s.wt{1}, wt);
end

function testDefaultWeightsAreOnes(tc)
[f, x, bounds] = calibToyProblem();
s = CalibSetup(bounds);
s.addVecExperiments(f([0.5 1]), ModelF(f, {'a','b'}), 0.1, 5, ones(size(x)));
tc.verifyEqual(s.wt{1}, ones(size(x)));
end

function testBadWeightsError(tc)
[f, x, bounds] = calibToyProblem();
s = CalibSetup(bounds);
add = @(w) s.addVecExperiments(f([0.5 1]), ModelF(f, {'a','b'}), 0.1, 5, ones(size(x)), ...
    NaN, NaN, NaN, 1, [], [], w);
tc.verifyError(@() add([1 2]), 'CalibSetup:addVecExperiments:badWt');
tc.verifyError(@() add(-ones(size(x))), 'CalibSetup:addVecExperiments:badWt');
end

function testDownweightingCorruptedDataRecoversTruth(tc)
% The second half of the data comes from different parameters; giving it
% a tiny weight lets pooled calibration recover the first half's truth.
rng(50);
[f, x, bounds] = calibToyProblem();
n = numel(x);
half = 1:floor(n/2);
y = f([0.6 1.5]);
yb = f([0.2 2.5]);
y(half(end)+1:end) = yb(half(end)+1:end);
y = y + 0.03*randn(1,n);
wt = [ones(1,numel(half)) 1e-3*ones(1,n-numel(half))];
s2_ind = [ones(1,numel(half)) 2*ones(1,n-numel(half))];
est = zeros(2,2);
for k = 1:2
    s = CalibSetup(bounds);
    if k == 1
        w = [];
    else
        w = wt;
    end
    s.addVecExperiments(y, ModelF(f, {'a','b'}, NaN, 'fix'), [0.03 0.03], 5, s2_ind, ...
        NaN, NaN, NaN, 1, [], [], w);
    s.setTemperatureLadder(1.05.^(0:2), 300);
    s.setMCMC(1500, 500, 1, 100);
    evalc('o = calibPool(s);');
    est(k,:) = [mean(o.theta_native.a(501:end)) mean(o.theta_native.b(501:end))];
end
tc.verifyEqual(est(2,:), [0.6 1.5], 'AbsTol', [0.05 0.15]);
tc.verifyGreaterThan(norm(est(1,:) - [0.6 1.5]), norm(est(2,:) - [0.6 1.5]));
end

function testWeightsInHierarchicalCalibration(tc)
rng(51);
[f, x, bounds] = calibToyProblem();
n = numel(x);
f2 = @(X) [f(X) f(X)];
ind = [ones(1,n) 2*ones(1,n)];
y = [f([0.3 1]) f([0.7 2])] + 0.05*randn(1,2*n);
s = CalibSetup(bounds);
s.addVecExperiments(y, ModelF(f2, {'a','b'}), [0.05 0.05], 5, ind, NaN, ind, NaN, 1, [], [], ...
    [2*ones(1,n) 0.5*ones(1,n)]);
s.setMCMC(1500, 500, 1, 100);
s.setHierPriors([0.5 0.5], eye(2), 5, 0.1*eye(2));
evalc('o = calibHier(s);');
tn = o.theta_native{1};
tc.verifyEqual(mean(tn.a(501:end,:), 1), [0.3 0.7], 'AbsTol', 0.05);
end

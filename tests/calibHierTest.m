function tests = calibHierTest
% End-to-end tests for hierarchical calibration on a small toy model.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testRecoversExperimentSpecificParameters(tc)
rng(30);
[f, x, bounds] = calibToyProblem();
n = numel(x);
truth = [0.3 1.0; 0.5 1.5; 0.7 2.0];
f2 = @(X) [f(X) f(X)];
s = CalibSetup(bounds);
% experiment 1 holds two thetas in one vector, experiment 2 holds one
y1 = [f(truth(1,:)) f(truth(2,:))] + 0.05*randn(1, 2*n);
ind = [ones(1,n) 2*ones(1,n)];
s.addVecExperiments(y1, ModelF(f2, {'a','b'}), [0.05 0.05], 5, ind, NaN, ind);
s.addVecExperiments(f(truth(3,:)) + 0.05*randn(1,n), ModelF(f, {'a','b'}, NaN, 'MH'), 0.05, 5, ones(1,n));
s.setTemperatureLadder(1.05.^(0:2), 300);
s.setMCMC(2000, 1000, 1, 100);
s.setHierPriors([0.5 0.5], eye(2), 5, 0.1*eye(2));
evalc('o = calibHier(s);');
b = 1001:2000;
est = [mean(o.theta_native{1}.a(b,:), 1)' mean(o.theta_native{1}.b(b,:), 1)'; ...
       mean(o.theta_native{2}.a(b)) mean(o.theta_native{2}.b(b))];
tc.verifyEqual(est(:,1), truth(:,1), 'AbsTol', 0.05);
tc.verifyEqual(est(:,2), truth(:,2), 'AbsTol', 0.1);
tc.verifySize(o.theta{1}, [2000 3 2 2]);
tc.verifySize(o.Sigma0, [2000 3 2 2]);
tc.verifyEqual(o.count_temper, o.count_temper');
tc.verifyGreaterThan(sum(o.count_s2(2,:)), 0);
end

function testRequiresHierPriors(tc)
[f, x, bounds] = calibToyProblem();
s = CalibSetup(bounds);
s.addVecExperiments(f([0.5 1]), ModelF(f, {'a','b'}), 0.1, 5, ones(size(x)));
s.setMCMC(10);
tc.verifyError(@() calibHier(s), 'calibHier:noPriors');
end

function testSetHierPriorsValidatesDf(tc)
[~, ~, bounds] = calibToyProblem();
s = CalibSetup(bounds);
tc.verifyError(@() s.setHierPriors([0.5 0.5], eye(2), 3, eye(2)), 'CalibSetup:setHierPriors:badDf');
end

function tests = calibClustTest
% End-to-end tests for clustered calibration on a small toy model.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testFindsTwoClusters(tc)
rng(40);
[f, x, bounds] = calibToyProblem();
truth = [0.3 1.0; 0.3 1.0; 0.7 2.0; 0.7 2.0; 0.7 2.0];
s = CalibSetup(bounds);
for e = 1:size(truth,1)
    s.addVecExperiments(f(truth(e,:)) + 0.05*randn(size(x)), ModelF(f, {'a','b'}), 0.05, 5, ones(size(x)));
end
s.setTemperatureLadder(1.05.^(0:2), 300);
s.setMCMC(1500, 500, 1, 100);
s.setHierPriors([0.5 0.5], eye(2), 5, 0.1*eye(2));
s.setClusterPriors(5);
evalc('o = calibClust(s);');
b = 501:1500;
d = zeros(numel(b), 5);
for i = 1:5
    d(:,i) = o.delta{i}(b,1);
    tc.verifyEqual(mean(o.theta_hist_native{i}.a(b)), truth(i,1), 'AbsTol', 0.05);
    tc.verifyEqual(mean(o.theta_hist_native{i}.b(b)), truth(i,2), 'AbsTol', 0.1);
end
tc.verifyGreaterThan(mean(d(:,1) == d(:,2)), 0.6);
tc.verifyGreaterThan(mean(d(:,3) == d(:,5)), 0.6);
tc.verifyLessThan(mean(d(:,1) == d(:,3)), 0.05);
tc.verifySize(o.theta, [1500 3 5 2]);
tc.verifyTrue(all(o.eta(:) > 0));
end

function testScaleProposalRuns(tc)
rng(41);
[f, x, bounds] = calibToyProblem();
s = CalibSetup(bounds);
for e = 1:3
    s.addVecExperiments(f([0.5 1.5]) + 0.05*randn(size(x)), ModelF(f, {'a','b'}, NaN, 'MH'), 0.05, 5, ones(size(x)));
end
s.setMCMC(400, 0, 1, 100, 1e-6, 0, 1e-5, 0, 200);
s.setHierPriors([0.5 0.5], eye(2), 5, 0.1*eye(2));
s.setClusterPriors(4, 2, 0.1, true);
evalc('o = calibClust(s);');
tc.verifySize(o.delta{1}, [400 1]);
tc.verifyTrue(all(o.delta{1}(:) >= 1 & o.delta{1}(:) <= 4));
end

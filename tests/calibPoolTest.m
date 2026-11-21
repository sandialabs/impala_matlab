function tests = calibPoolTest
% End-to-end tests for pooled calibration on a small toy model.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testRecoversParametersForEachS2Mode(tc)
[f, x, bounds] = calibToyProblem();
modes = {'gibbs', 'gibbs_trunc', 'MH', 'fix'};
for k = 1:numel(modes)
    rng(10 + k);
    y = f([0.6 1.5]) + 0.05*randn(size(x));
    s = CalibSetup(bounds);
    s.addVecExperiments(y, ModelF(f, {'a','b'}, NaN, modes{k}), 0.05, 5, ...
        ones(size(x)), NaN, NaN, NaN, 1, 0.01, 1);
    s.setTemperatureLadder(1.05.^(0:2), 300);
    s.setMCMC(1500, 500, 1, 100);
    evalc('o = calibPool(s);');
    b = 501:1500;
    tc.verifyEqual(mean(o.theta_native.a(b)), 0.6, 'AbsTol', 0.05, modes{k});
    tc.verifyEqual(mean(o.theta_native.b(b)), 1.5, 'AbsTol', 0.1, modes{k});
    tc.verifyEqual(o.count, o.count', modes{k});
end
end

function testFixedS2StaysAtEstimate(tc)
rng(20);
[f, x, bounds] = calibToyProblem();
y = f([0.4 1]) + 0.05*randn(size(x));
s = CalibSetup(bounds);
s.addVecExperiments(y, ModelF(f, {'a','b'}, NaN, 'fix'), 0.07, 5, ones(size(x)));
s.setMCMC(300);
evalc('o = calibPool(s);');
tc.verifyEqual(o.s2{1}(:), repmat(0.07^2, 300, 1), 'AbsTol', 1e-12);
end

function testSingleTemperatureAndTwoErrorGroups(tc)
rng(21);
[f, x, bounds] = calibToyProblem();
y = f([0.4 1]);
s2_ind = [ones(1,15) 2*ones(1,15)];
y = y + [0.02*randn(1,15) 0.2*randn(1,15)];
s = CalibSetup(bounds);
s.addVecExperiments(y, ModelF(f, {'a','b'}), [0.1 0.1], 5, s2_ind);
s.setMCMC(1500);
evalc('o = calibPool(s);');
tc.verifySize(o.theta, [1500 1 2]);
tc.verifySize(o.s2{1}, [1500 1 2]);
sd = sqrt(mean(reshape(o.s2{1}(501:end,1,:), [], 2), 1));
tc.verifyLessThan(sd(1), sd(2));
end

function testZeroBasedS2IndexAccepted(tc)
[f, x, bounds] = calibToyProblem();
s = CalibSetup(bounds);
s.addVecExperiments(f([0.5 1]), ModelF(f, {'a','b'}), 0.1, 5, zeros(size(x)));
tc.verifyEqual(s.s2_ind{1}, ones(size(x)));
end

function testBadS2ModeErrors(tc)
[f, x, bounds] = calibToyProblem();
s = CalibSetup(bounds);
tc.verifyError(@() s.addVecExperiments(f([0.5 1]), ModelF(f, {'a','b'}, NaN, 'bogus'), 0.1, 5, ones(size(x))), ...
    'CalibSetup:addVecExperiments:badS2');
end

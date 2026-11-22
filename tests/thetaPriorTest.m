function tests = thetaPriorTest
% Priors on the calibration parameters (port of the python package's
% test_theta_prior.py).
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function s = makeSetup(bounds)
if nargin < 1
    bounds = struct('t_0', [0 1], 't_1', [0 1]);
end
s = CalibSetup(bounds, @cf_bounds);
end

function lp = thetaLogPrior(s, theta)
% log-prior of unit-scaled draws, as evaluated inside calibPool
lp = eval_theta_priors(tran_unif(theta, s.bounds_mat, fieldnames(s.bounds)), ...
                       s.theta_prior);
end

function lp = normLogpdf(x, mu, sd)
lp = -log(sd) - 0.5*log(2*pi) - 0.5*((x - mu)/sd).^2;
end

function testNoPriorIsZeroAndDefaultEmpty(tc)
s = makeSetup();
tc.verifyEmpty(s.theta_prior);
lp = thetaLogPrior(s, rand(5, 2));
tc.verifySize(lp, [1 5]);
tc.verifyEqual(lp, zeros(1, 5), 'AbsTol', 1e-15);
end

function testIndependentPriorsMatchReference(tc)
s = makeSetup(struct('t_0', [0 4], 't_1', [0 1]));
s.addThetaPrior('normal', struct('mean', 2, 'sd', 0.5), 't_0');
s.addThetaPrior('beta', struct('shape1', 2, 'shape2', 5), 't_1');
tc.verifyNumElements(s.theta_prior, 2);
theta = [0.25 0.10; 0.50 0.90; 0.75 0.50];
t0 = theta(:, 1)'*4;
t1 = theta(:, 2)';
expected = normLogpdf(t0, 2, 0.5) + log(30*t1.*(1 - t1).^4);
tc.verifyEqual(thetaLogPrior(s, theta), expected, 'RelTol', 1e-10);
end

function testAllSupportedDistsAreFiniteAndNormalized(tc)
dists = {'normal', struct('mean', 0.5, 'sd', 1); ...
         'lognormal', struct('meanlog', 0, 'sdlog', 1); ...
         'beta', struct('shape1', 2, 'shape2', 2); ...
         'uniform', struct('min', 0, 'max', 1); ...
         'gamma', struct('shape', 2, 'rate', 3); ...
         'cauchy', struct('location', 0.5, 'scale', 1)};
x = linspace(0.01, 0.99, 25)';
theta = [x, 0.5*ones(25, 1)];
for k = 1:size(dists, 1)
    s = makeSetup();
    s.addThetaPrior(dists{k, 1}, dists{k, 2}, 't_0');
    lp = thetaLogPrior(s, theta);
    tc.verifySize(lp, [1 25], dists{k, 1});
    tc.verifyTrue(all(isfinite(lp)), dists{k, 1});
    % crude check that the density integrates to <= 1 over [0, 1]
    tc.verifyLessThanOrEqual(trapz(x, exp(lp)), 1 + 1e-8, dists{k, 1});
end
end

function testJointPrior(tc)
s = makeSetup();
C = [1 0.5; 0.5 1];
mvn = @(v) -log(2*pi) - 0.5*log(det(C)) - 0.5*((v - 0.5)/C)*(v - 0.5)';
s.addJointThetaPrior({'t_0', 't_1'}, @(pp) mvn([pp.t_0, pp.t_1]));
theta = [0.2 0.3; 0.5 0.5; 0.9 0.1];
expected = arrayfun(@(i) mvn(theta(i, :)), 1:3);
tc.verifyEqual(thetaLogPrior(s, theta), expected, 'RelTol', 1e-12);
end

function testJointAndIndependentPriorsAdd(tc)
s = makeSetup();
s.addThetaPrior('normal', struct('mean', 0.5, 'sd', 0.2), 't_0');
s.addJointThetaPrior({'t_0', 't_1'}, @(pp) -1);
theta = [0.4 0.6; 0.5 0.5];
expected = normLogpdf(theta(:, 1)', 0.5, 0.2) - 1;
tc.verifyEqual(thetaLogPrior(s, theta), expected, 'RelTol', 1e-12);
end

function testScalarJointPriorBroadcasts(tc)
s = makeSetup();
s.addJointThetaPrior({'t_0', 't_1'}, @(pp) -2.5);
tc.verifyEqual(thetaLogPrior(s, rand(4, 2)), repmat(-2.5, 1, 4));
end

function testOutOfSupportIsNegInf(tc)
s = makeSetup();
s.addThetaPrior('gamma', struct('shape', 2, 'rate', 1), 't_0');
tc.verifyEqual(thetaLogPrior(s, [0 0.5]), -Inf);
end

function testEvalThetaPriorsUsesNames(tc)
priors = {struct('name', 't_0', 'dist', 'normal', ...
                 'params', struct('mean', 0, 'sd', 1))};
theta = struct('t_0', [0 1], 't_1', [9 9]);
tc.verifyEqual(eval_theta_priors(theta, priors), normLogpdf([0 1], 0, 1), ...
               'RelTol', 1e-12);
end

function testBadInputRaises(tc)
s = makeSetup();
n01 = struct('mean', 0, 'sd', 1);
tc.verifyError(@() s.addThetaPrior('normal', n01), ...
               'CalibSetup:addThetaPrior:badName');
tc.verifyError(@() s.addThetaPrior('normal', n01, 'nope'), ...
               'CalibSetup:addThetaPrior:badName');
tc.verifyError(@() s.addThetaPrior('weibull', struct(), 't_0'), ...
               'CalibSetup:addThetaPrior:badDist');
tc.verifyError(@() s.addThetaPrior('normal', struct('mean', 0), 't_0'), ...
               'CalibSetup:addThetaPrior:missingParams');
tc.verifyError(@() s.addJointThetaPrior({'t_0', 'nope'}, @(pp) 0), ...
               'CalibSetup:addJointThetaPrior:badNames');
tc.verifyError(@() s.addJointThetaPrior({'t_0'}, 'not a function'), ...
               'CalibSetup:addJointThetaPrior:badFunction');
tc.verifyError(@() eval_theta_priors(struct('t_0', zeros(1, 3)), ...
               {struct('name', 't_0', 'dist', 'weibull', 'params', struct())}), ...
               'eval_theta_priors:unsupportedDist');
end

function s = buildLineSetup(yobs, model, nmcmc)
s = CalibSetup(struct('t_0', [0 1], 't_1', [0 1]), @cf_bounds);
s.addVecExperiments(yobs, model, 0.1, 0, zeros(size(yobs)));
s.setTemperatureLadder(1.05.^(0:7));
s.setMCMC(nmcmc, 0, 1, 100);
end

function testCalibPoolPriorShiftsPosterior(tc)
% a tight prior away from the MLE pulls the posterior toward it
rng(0);
grid = linspace(0, 1, 40);
line = @(X) X(:, 1).*grid + X(:, 2);
model = ModelF(line, {'t_0', 't_1'});
yobs = line([0.8 0.2]) + 0.1*randn(size(grid));

rng(1);
evalc('flat = calibPool(buildLineSetup(yobs, model, 6000));');
rng(1);
s = buildLineSetup(yobs, model, 6000);
s.addThetaPrior('normal', struct('mean', 0.2, 'sd', 0.02), 't_0');
evalc('withPrior = calibPool(s);');

flatMean = mean(flat.theta(3001:end, 1, 1));
priorMean = mean(withPrior.theta(3001:end, 1, 1));
% without a prior the posterior sits at the least-squares fit (the python
% test compares with the truth, 0.8, but that depends on its noise draw)
ls = [grid' ones(numel(grid), 1)]\yobs';
tc.verifyLessThan(abs(flatMean - ls(1)), 0.05);
tc.verifyLessThan(abs(flatMean - 0.8), 0.15);
tc.verifyLessThan(priorMean, flatMean - 0.2);
tc.verifyLessThan(abs(priorMean - 0.2), 0.05);
end

function testCalibPoolRecoversPriorWithNoInformation(tc)
% with a flat likelihood the posterior looks like the prior
rng(0);
grid = linspace(0, 1, 20);
model = ModelF(@(X) zeros(size(X, 1), numel(grid)) + 0*X(:, 1), {'t_0', 't_1'});
yobs = 0.1*randn(1, 20);
s = buildLineSetup(yobs, model, 8000);
s.addThetaPrior('beta', struct('shape1', 3, 'shape2', 6), 't_0');
rng(2);
evalc('out = calibPool(s);');

draws = out.theta(4001:2:end, 1, 1);
a = 3; b = 6;
tc.verifyLessThan(abs(mean(draws) - a/(a + b)), 0.05);
tc.verifyLessThan(abs(std(draws, 1) - sqrt(a*b/((a + b)^2*(a + b + 1)))), 0.05);
% t_1 has no prior, so it stays uniform
tc.verifyLessThan(abs(mean(out.theta(4001:2:end, 1, 2)) - 0.5), 0.05);
end

function testCalibPoolUnchangedWithoutPrior(tc)
% an empty prior list gives bitwise-identical results
rng(0);
grid = linspace(0, 1, 30);
line = @(X) X(:, 1).*grid + X(:, 2);
model = ModelF(line, {'t_0', 't_1'});
yobs = line([0.8 0.2]) + 0.1*randn(size(grid));

rng(3);
evalc('a = calibPool(buildLineSetup(yobs, model, 2000));');
s = buildLineSetup(yobs, model, 2000);
s.theta_prior = {};
rng(3);
evalc('b = calibPool(s);');
tc.verifyEqual(a.theta, b.theta);
end

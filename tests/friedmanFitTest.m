function tests = friedmanFitTest
% Pooled calibration of the Friedman function (port of the python package's
% test_friedman_fit.py). Six parameters are calibrated but the function only
% uses the first four, so the posteriors of the last two must stay uniform.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testFriedmanFit(tc)
rng(0);
nfeat = 6;
grid = linspace(0, 1, 50);
friedman = @(X) 10*sin(pi*grid.*X(:, 1)) + 20*(X(:, 2) - 0.5).^2 ...
    + 10*X(:, 3) + 5*X(:, 4);
truth = 0.1 + 0.8*rand(1, nfeat);
sigma = 0.1;
yobs = friedman(truth) + sigma*randn(size(grid));

names = arrayfun(@(i) sprintf('t%d', i), 0:nfeat-1, 'UniformOutput', false);
bounds = struct();
for i = 1:nfeat
    bounds.(names{i}) = [0 1];
end
s = CalibSetup(bounds, @cf_bounds);
s.addVecExperiments(yobs, ModelF(friedman, names), 1, 0, zeros(size(yobs)));
s.setTemperatureLadder(1.05.^(0:19));
s.setMCMC(15000, 0, 1, 100);
evalc('out = calibPool(s);');

post = reshape(out.theta(6001:5:end, 1, :), [], nfeat);
% superfluous parameters are uniform, utilized ones are not
for i = 1:nfeat
    pval = ksUniformPValue(post(:, i));
    if i <= 4
        tc.verifyLessThan(pval, 0.01, names{i});
    else
        tc.verifyGreaterThan(pval, 0.01, names{i});
    end
end
% the true values are in the 95% credible intervals
lo = quantile(post, 0.025, 1);
hi = quantile(post, 0.975, 1);
tc.verifyTrue(all(lo < truth & truth < hi));
end

function p = ksUniformPValue(x)
% one-sample Kolmogorov-Smirnov test against Uniform(0,1), asymptotic p-value
x = sort(x(:));
n = numel(x);
F = (1:n)'/n;
D = max(max(F - x), max(x - (F - 1/n)));
lam = (sqrt(n) + 0.12 + 0.11/sqrt(n))*D;
k = (1:100)';
p = min(max(2*sum((-1).^(k - 1).*exp(-2*k.^2*lam^2)), 0), 1);
end

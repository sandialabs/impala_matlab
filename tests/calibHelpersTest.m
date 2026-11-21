function tests = calibHelpersTest
% Tests for the helper functions used by pooled, hierarchical and
% clustered calibration.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testUnpoolPredPicksBlocks(tc)
% 2 replicates x 3 thetas stacked by theta block
pred = reshape(1:6*4, 6, 4);
exp_ind = [1 2 3 3];
out = unpool_pred(pred, exp_ind);
tc.verifyEqual(out, [pred(1:2,1) pred(3:4,2) pred(5:6,3:4)]);
end

function testUnpoolPredScalarIndexIsIdentity(tc)
pred = rand(3, 5);
tc.verifyEqual(unpool_pred(pred, 1), pred);
end

function testMvnLogpdfBatchMatchesMvnpdf(tc)
rng(1);
p = 3; nt = 2; n = 4;
X = rand(nt, n, p); mu = rand(nt, p);
prec = zeros(nt, p, p); ldet = zeros(nt, 1); C = cell(1, nt);
for t = 1:nt
    A = randn(p); C{t} = A*A' + eye(p);
    prec(t,:,:) = inv(C{t}); ldet(t) = log(det(C{t}));
end
ld = mvn_logpdf_batch(X, mu, prec, ldet);
for t = 1:nt
    ref = log(mvnpdf(reshape(X(t,:,:), n, p), mu(t,:), C{t}))';
    tc.verifyEqual(ld(t,:), ref, 'AbsTol', 1e-10);
end
end

function testInvWishartLogpdfOneDimIsInverseGamma(tc)
% for p = 1 the inverse Wishart(df, s) is inverse gamma(df/2, s/2)
df = 7; s = 2.5; w = [0.3; 1.1; 4];
a = df/2; b = s/2;
ref = a*log(b) - gammaln(a) - (a+1)*log(w) - b./w;
tc.verifyEqual(invwishart_logpdf(reshape(w, 3, 1, 1), df, s), ref, 'AbsTol', 1e-10);
end

function testGammaLogpdfMatchesGampdf(tc)
x = [0.5 2 7];
tc.verifyEqual(gamma_logpdf(x, 2, 0.1), log(gampdf(x, 2, 10)), 'AbsTol', 1e-12);
end

function testSampleConstrainedRespectsConstraints(tc)
rng(2);
bounds = struct('a', [0 1], 'b', [0 1]);
s = CalibSetup(bounds, @(x, b) x.a + x.b < 1);
out = sample_constrained(repmat([0.45 0.45], 200, 1), 0.05*eye(2), s);
tc.verifyTrue(all(sum(out, 2) < 1));
tc.verifySize(out, [200 2]);
end

function testDrawS2GibbsTruncatedStaysInBounds(tc)
rng(3);
bounds = struct('a', [0 1]);
s = CalibSetup(bounds);
f = @(X) X(:,1) * ones(1, 10);
s.addVecExperiments(zeros(1,10), ModelF(f, {'a'}, NaN, 'gibbs_trunc'), 0.1, 5, ones(1,10), ...
    NaN, NaN, NaN, 1, 0.09, 0.11);
for k = 1:50
    ls2 = draw_s2_gibbs(100, s, 1, 1, true);
    tc.verifyGreaterThanOrEqual(ls2, log(0.09^2) - 1e-12);
    tc.verifyLessThanOrEqual(ls2, log(0.11^2) + 1e-12);
end
end

function testAMcovPoolSingleTemperature(tc)
rng(4);
am = AMcov_pool(1, 3, 1e-4, 10, 0);
x = randn(20, 1, 3);
for m = 2:20
    am.update(x, m);
end
cand = am.gen_cand(x, 20);
tc.verifySize(cand, [1 3]);
tc.verifySize(am.S, [1 3 3]);
end

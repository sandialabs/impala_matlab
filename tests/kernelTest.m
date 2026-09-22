function tests = kernelTest
% Tests for log-density kernels: ldig_kern (inverse-gamma) and ldhc_kern
% (half-Cauchy). These return log-density up to an additive constant.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testLdigMatchesFormula(tc)
% ldig_kern(x,a,b) = (-a-1)*log(x) - b/x
x = [0.5 1 2 5];
a = 2; b = 3;
out = ldig_kern(x, a, b);
expected = (-a - 1) .* log(x) - b ./ x;
tc.verifyEqual(out, expected, 'AbsTol', 1e-12);
end

function testLdigProportionalToInvGammaLogPdf(tc)
% The kernel differs from the true inverse-gamma log-pdf only by a
% constant (a*log(b) - gammaln(a)); differences across points must match.
x = [0.5 1 2 5];
a = 2.5; b = 1.7;
kern = ldig_kern(x, a, b);
truelog = a*log(b) - gammaln(a) + (-a-1).*log(x) - b./x;
tc.verifyEqual(diff(kern), diff(truelog), 'AbsTol', 1e-12);
end

function testLdigVectorized(tc)
x = rand(1, 10) + 0.1;
out = ldig_kern(x, 1, 1);
tc.verifySize(out, size(x));
end

function testLdhcMatchesFormula(tc)
% ldhc_kern(x) = -log(x+1)
x = [0 1 3 9];
tc.verifyEqual(ldhc_kern(x), -log(x + 1), 'AbsTol', 1e-12);
end

function testLdhcMonotoneDecreasing(tc)
x = linspace(0, 10, 50);
out = ldhc_kern(x);
tc.verifyTrue(all(diff(out) < 0));
end

function testLdhcAtZeroIsZero(tc)
tc.verifyEqual(ldhc_kern(0), 0, 'AbsTol', 1e-12);
end

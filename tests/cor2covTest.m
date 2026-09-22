function tests = cor2covTest
% Tests for cor2cov. NOTE: the implementation computes out = (s*s') * R
% using matrix multiplication (not element-wise Hadamard scaling).
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testMatchesMatrixProductFormula(tc)
% Golden reference: exactly the documented operation.
rng(7);
n = 4;
s = rand(n, 1) + 0.5;
A = randn(n);
R = corrcov(A * A' + eye(n));   % a valid correlation matrix
out = cor2cov(R, s);
expected = (s * s') * R;
tc.verifyEqual(out, expected, 'AbsTol', 1e-10);
end

function testIdentityCorrelationGivesOuterProduct(tc)
% With R = I, (s*s')*I = s*s'.
s = [2; 3; 4];
out = cor2cov(eye(3), s);
tc.verifyEqual(out, s * s', 'AbsTol', 1e-12);
end

function testKnownTwoByTwo(tc)
s = [2; 3];
R = [1 0.5; 0.5 1];
out = cor2cov(R, s);
expected = (s * s') * R;
tc.verifyEqual(out, expected, 'AbsTol', 1e-12);
end

function testOutputSize(tc)
s = ones(5, 1);
R = eye(5);
out = cor2cov(R, s);
tc.verifySize(out, [5 5]);
end

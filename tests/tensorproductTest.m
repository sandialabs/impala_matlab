function tests = tensorproductTest
% Tests for tensorproduct: Einstein-summation over multidimensional arrays.
% Behavior is checked against equivalent explicit MATLAB operations.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testMatrixMultiplyAsInnerProduct(tc)
% 'ik','kj' -> 'ij' is ordinary matrix multiplication.
rng(1);
A = randn(4, 3);
B = randn(3, 5);
R = tensorproduct('ij', A, 'ik', B, 'kj');
tc.verifyEqual(R, A * B, 'AbsTol', 1e-10);
end

function testTransposeMatrixMultiply(tc)
% A' * B expressed via a shared inner index on the first dimension.
rng(2);
A = randn(3, 4);
B = randn(3, 5);
R = tensorproduct('ij', A, 'ki', B, 'kj');
tc.verifyEqual(R, A' * B, 'AbsTol', 1e-10);
end

function testInnerProductOfVectors(tc)
% Full contraction of a shared index -> scalar dot product.
rng(3);
a = randn(6, 1);
b = randn(6, 1);
R = tensorproduct('', a, 'i', b, 'i');
tc.verifyEqual(R, sum(a .* b), 'AbsTol', 1e-10);
end

function testOuterProduct(tc)
% No shared index -> outer product.
a = (1:3)';
b = (1:4)';
R = tensorproduct('ij', a, 'i', b, 'j');
tc.verifyEqual(R, a * b', 'AbsTol', 1e-12);
end

function testPagewiseMatrixMultiply(tc)
% Shared page index 'g' present in output -> batched (paged) matmul.
rng(4);
A = randn(3, 2, 5);   % i j g
B = randn(2, 4, 5);   % j k g
R = tensorproduct('ikg', A, 'ijg', B, 'jkg');
expected = zeros(3, 4, 5);
for g = 1:5
    expected(:, :, g) = A(:, :, g) * B(:, :, g);
end
tc.verifyEqual(R, expected, 'AbsTol', 1e-10);
end

function testEmptyInputErrors(tc)
tc.verifyError(@() tensorproduct('ij', [], 'ik', randn(2, 2), 'kj'), ?MException);
end

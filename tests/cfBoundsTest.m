function tests = cfBoundsTest
% Tests for cf_bounds: returns a logical mask of samples that fall strictly
% inside the (lower, upper) bounds for every named field.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testAllInsideReturnsTrue(tc)
x.a = [1 2 3];
x.b = [5 6 7];
bounds.a = [0 10];
bounds.b = [0 10];
good = cf_bounds(x, bounds);
tc.verifyEqual(good, logical([1 1 1]));
end

function testValuesOutsideAreFalse(tc)
x.a = [-1 5 11];   % first below, last above bound [0 10]
bounds.a = [0 10];
good = cf_bounds(x, bounds);
tc.verifyEqual(good, logical([0 1 0]));
end

function testStrictBoundariesAreExcluded(tc)
% Comparison is strict (< upper and > lower); endpoints are rejected.
x.a = [0 10 5];
bounds.a = [0 10];
good = cf_bounds(x, bounds);
tc.verifyEqual(good, logical([0 0 1]));
end

function testMultipleFieldsAreAnded(tc)
% A sample is good only if in-bounds for every field.
x.a = [1 1 20];
x.b = [1 20 1];
bounds.a = [0 10];
bounds.b = [0 10];
good = cf_bounds(x, bounds);
tc.verifyEqual(good, logical([1 0 0]));
end

function testReturnsLogicalType(tc)
x.a = [1 2];
bounds.a = [0 10];
good = cf_bounds(x, bounds);
tc.verifyClass(good, 'logical');
end

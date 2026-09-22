function tests = tranUnifTest
% Tests for tran_unif: unnormalize a unit-scale parameter matrix and pack
% it into a struct keyed by parameter names.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testFieldNamesMatch(tc)
th = [0.5 0.5];
bounds = [0 10; -1 1];
names = {'alpha', 'beta'};
out = tran_unif(th, bounds, names);
tc.verifyEqual(sort(fieldnames(out)), sort(names(:)));
end

function testValuesAreUnnormalized(tc)
% Row of 0.5 -> midpoint of each bound.
th = [0.5 0.5];
bounds = [0 10; -1 1];
names = {'a', 'b'};
out = tran_unif(th, bounds, names);
tc.verifyEqual(out.a, 5, 'AbsTol', 1e-12);
tc.verifyEqual(out.b, 0, 'AbsTol', 1e-12);
end

function testMultipleSamplesPerField(tc)
% Several unit-scale rows -> row vector per named parameter.
th = [0 0; 1 1; 0.5 0.5];
bounds = [0 2; 10 20];
names = {'x', 'y'};
out = tran_unif(th, bounds, names);
tc.verifyEqual(out.x, [0 2 1], 'AbsTol', 1e-12);
tc.verifyEqual(out.y, [10 20 15], 'AbsTol', 1e-12);
end

function testConsistentWithUnnormalize(tc)
rng(3);
th = rand(5, 3);
bounds = [0 1; -5 5; 100 200];
names = {'p1', 'p2', 'p3'};
out = tran_unif(th, bounds, names);
u = unnormalize(th, bounds);   % 5x3
tc.verifyEqual(out.p1, u(:,1)', 'AbsTol', 1e-10);
tc.verifyEqual(out.p2, u(:,2)', 'AbsTol', 1e-10);
tc.verifyEqual(out.p3, u(:,3)', 'AbsTol', 1e-10);
end

function tests = normalizeTest
% Tests for normalize / unnormalize (round-trip and boundary behavior).
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testNormalizeBoundsToUnitInterval(tc)
% Lower bound maps to 0, upper bound maps to 1.
bounds = [0 10; -5 5];        % 2 parameters
x = [0 -5;                    % row of lower bounds
     10 5];                   % row of upper bounds
out = normalize(x, bounds);
tc.verifyEqual(out, [0 0; 1 1], 'AbsTol', 1e-12);
end

function testNormalizeMidpoint(tc)
bounds = [0 10; -5 5];
x = [5 0];                    % midpoints
out = normalize(x, bounds);
tc.verifyEqual(out, [0.5 0.5], 'AbsTol', 1e-12);
end

function testUnnormalizeIsInverseOfNormalize(tc)
% normalize then unnormalize recovers the original input.
bounds = [-2 3; 100 200; 0 1];
rng(1);
x = bounds(:,1)' + rand(20, 3) .* (bounds(:,2) - bounds(:,1))';
z = normalize(x, bounds);
recovered = unnormalize(z, bounds);
tc.verifyEqual(recovered, x, 'AbsTol', 1e-10);
end

function testNormalizeThenUnnormalizeUnitScale(tc)
% A value on the unit scale unnormalizes to within bounds.
bounds = [10 20];
z = [0; 0.25; 0.5; 1];
out = unnormalize(z, bounds);
tc.verifyEqual(out, [10; 12.5; 15; 20], 'AbsTol', 1e-12);
end

function testNormalizeSingleRow(tc)
bounds = [0 4];
out = normalize(1, bounds);
tc.verifyEqual(out, 0.25, 'AbsTol', 1e-12);
end

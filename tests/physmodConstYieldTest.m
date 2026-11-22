function tests = physmodConstYieldTest
% Constant yield stress model tests (port of the python package's
% test_physmod_constyield.py).
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
tc.TestData.h = physicsTestHelpers();
tc.TestData.params = struct();
tc.TestData.consts = struct('yield_stress', 0.1, 'rho0', 8.96, ...
    'Cv0', 0.383e-5, 'G0', 0.4578, 'Tmelt0', 1356, 'chi', 1);
tc.TestData.options = struct('edotcrit_chi', 1e-6);
tc.TestData.model = physics.MaterialModel();
end

function res = runIt(tc, edot, options)
res = tc.TestData.h.run(tc.TestData.model, tc.TestData.params, ...
    tc.TestData.consts, 1, edot, 1000, 298, options);
end

function testIsothermalLowRate(tc)
% rates below edotcrit_chi (~1/s) are isothermal
res = runIt(tc, 1e-7, tc.TestData.options);
tc.verifyEqual(res(1, 4), res(end, 4));
end

function testAdiabaticHighRate(tc)
res = runIt(tc, 1, struct());
tc.verifyNotEqual(res(1, 4), res(end, 4));
end

function testConstantStress(tc)
res = runIt(tc, 1, struct());
tc.verifyEqual(res(:, 3), repmat(tc.TestData.consts.yield_stress, 1000, 1));
end

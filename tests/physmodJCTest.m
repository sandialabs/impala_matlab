function tests = physmodJCTest
% Johnson-Cook yield stress tests, compared with FLAG simulations of OFHC
% copper (port of the python package's test_physmod_yjc.py).
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
tc.TestData.h = physicsTestHelpers();
tc.TestData.params = struct('A', 0.0009, 'B', 0.00292, 'C', 0.025, ...
    'n', 0.31, 'm', 1.09);
tc.TestData.consts = struct('rho0', 8.9375, 'Cv0', 0.383e-5, 'G0', 0.4578, ...
    'Tmelt0', 1356, 'Tref', 298, 'edot0', 1e-6, 'chi', 1);
tc.TestData.options = struct('edotcrit_chi', 1e-6);
tc.TestData.model = physics.MaterialModel(flow_stress_model="JC_Yield_Stress");
end

function res = runIt(tc, emax, edot, options)
res = tc.TestData.h.run(tc.TestData.model, tc.TestData.params, ...
    tc.TestData.consts, emax, edot, 1000, 298, options);
end

function checkFLAG(tc, emax, edot, options, stressFLAG, tol)
% flow stress at emax within tol (relative) of the FLAG result
res = runIt(tc, emax, edot, options);
tc.verifyLessThan(tc.TestData.h.relDiff(res(end, 3), stressFLAG), tol);
end

function testIsothermalLowRate(tc)
res = runIt(tc, 1, 1e-7, tc.TestData.options);
tc.verifyEqual(res(1, 4), res(end, 4));
end

function testAdiabaticHighRate(tc)
res = runIt(tc, 1, 1e-2, struct());
tc.verifyNotEqual(res(1, 4), res(end, 4));
end

function testStressHighRate(tc)
checkFLAG(tc, 0.28179, 1e-2, struct(), 343.37e-5, 0.02);
end

function testStressMidRate(tc)
checkFLAG(tc, 0.28895, 1e-4, struct(), 317.06e-5, 0.01);
end

function testStressLowRate(tc)
checkFLAG(tc, 0.28928, 1e-7, tc.TestData.options, 284.89e-5, 0.05);
end

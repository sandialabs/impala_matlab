function tests = physmodPTWTest
% PTW yield stress tests with constant, simple (PW) and Steinberg-Guinan
% shear moduli, compared with FLAG simulations of OFHC copper (port of the
% python package's test_physmod_ptw.py).
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
tc.TestData.h = physicsTestHelpers();
tc.TestData.params = struct('theta', 0.025, 'p', 2, 's0', 0.0085, ...
    'sInf', 0.00055, 'kappa', 0.11, 'lgamma', log(1e-5), 'y0', 0.0001, ...
    'yInf', 0.00009999, 'y1', 0.094, 'y2', 0.575);
base = struct('beta', 0.25, 'matomic', 63.546, 'chi', 1, 'Cv0', 0.383e-5, ...
    'rho0', 8.9375, 'Tmelt0', 1625);
c = base;
c.G0 = 0.4578;
tc.TestData.consts.Constant_Shear_Modulus = c;
c = base;
c.G0 = 0.50889;
c.alpha = 0.21;
tc.TestData.consts.Simple_Shear_Modulus = c;
c = base;
c.G0 = 0.4578;
c.sgB = 3.8e-4;
tc.TestData.consts.Stein_Shear_Modulus = c;
tc.TestData.options = struct('edotcrit_chi', 1e-6);
end

function res = runIt(tc, shear, emax, edot, useOptions, T)
if nargin < 6
    T = 298;
end
m = physics.MaterialModel(flow_stress_model="PTW_Yield_Stress", ...
                          shear_modulus_model=shear);
opts = struct();
if useOptions
    opts = tc.TestData.options;
end
res = tc.TestData.h.run(m, tc.TestData.params, tc.TestData.consts.(shear), ...
    emax, edot, 1000, T, opts);
end

function checkFLAG(tc, shear, emax, edot, useOptions, stressFLAG, tol, T)
if nargin < 8
    T = 298;
end
res = runIt(tc, shear, emax, edot, useOptions, T);
tc.verifyLessThan(tc.TestData.h.relDiff(res(end, 3), stressFLAG), tol, shear);
end

function testIsothermalLowRate(tc)
for shear = ["Constant_Shear_Modulus" "Simple_Shear_Modulus" "Stein_Shear_Modulus"]
    res = runIt(tc, shear, 1, 1e-7, true);
    tc.verifyEqual(res(1, 4), res(end, 4), shear);
end
end

function testAdiabaticHighRate(tc)
for shear = ["Constant_Shear_Modulus" "Simple_Shear_Modulus" "Stein_Shear_Modulus"]
    res = runIt(tc, shear, 1, 1e-2, false);
    tc.verifyNotEqual(res(1, 4), res(end, 4), shear);
end
end

% constant shear modulus and melt temperature
function testConstShearStressHighRate(tc)
checkFLAG(tc, 'Constant_Shear_Modulus', 0.244376, 1, false, 521.416e-5, 0.01);
end

function testConstShearStressMidHighRate(tc)
checkFLAG(tc, 'Constant_Shear_Modulus', 0.382749, 1e-2, false, 458.198e-5, 0.09);
end

function testConstShearStressMidLowRate(tc)
checkFLAG(tc, 'Constant_Shear_Modulus', 0.388339, 1e-4, false, 418.512e-5, 0.11);
end

function testConstShearStressLowRate(tc)
checkFLAG(tc, 'Constant_Shear_Modulus', 0.371143, 1e-6, true, 385.751e-5, 0.09);
end

% simple (PW) shear modulus, constant melt temperature
function testSimpleShearStressHighRate(tc)
% starts near the ~620 K FLAG reaches at this rate, as in the python test
checkFLAG(tc, 'Simple_Shear_Modulus', 0.244463, 1, false, 485.301e-5, 0.06, 620);
end

function testSimpleShearStressMidHighRate(tc)
checkFLAG(tc, 'Simple_Shear_Modulus', 0.383542, 1e-2, false, 439.790e-5, 0.02);
end

function testSimpleShearStressMidLowRate(tc)
checkFLAG(tc, 'Simple_Shear_Modulus', 0.388302, 1e-4, false, 401.108e-5, 0.01);
end

function testSimpleShearStressLowRate(tc)
checkFLAG(tc, 'Simple_Shear_Modulus', 0.385416, 1e-6, false, 375.227e-5, 0.02);
end

% Steinberg-Guinan shear modulus, constant melt temperature
function testSteinShearStressHighRate(tc)
% starts near the FLAG temperature at this rate, as in the python test
checkFLAG(tc, 'Stein_Shear_Modulus', 0.244632, 1, false, 427.011e-5, 0.04, 617);
end

function testSteinShearStressMidHighRate(tc)
checkFLAG(tc, 'Stein_Shear_Modulus', 0.384834, 1e-2, false, 410.483e-5, 0.02);
end

function testSteinShearStressMidLowRate(tc)
checkFLAG(tc, 'Stein_Shear_Modulus', 0.388306, 1e-4, false, 373.629e-5, 0.01);
end

function testSteinShearStressLowRate(tc)
checkFLAG(tc, 'Stein_Shear_Modulus', 0.388966, 1e-6, false, 351.103e-5, 0.02);
end

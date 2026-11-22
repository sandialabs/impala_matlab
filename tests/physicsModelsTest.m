function tests = physicsModelsTest
% PTW regression and pooled-calibration tests (port of the python
% package's test_physical_models_vec.py).
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
tc.TestData.h = physicsTestHelpers();
tc.TestData.consts = struct('alpha', 0.2, 'beta', 0.33, 'matomic', 45.9, ...
    'Tmelt0', 2110, 'rho0', 4.419, 'Cv0', 0.525e-5, 'G0', 0.4, 'chi', 1, ...
    'sgB', 6.44e-4);
tc.TestData.models = {'PTW_Yield_Stress', 'Constant_Melt_Temperature', ...
    'Stein_Shear_Modulus', 'Constant_Specific_Heat', 'Constant_Density'};
end

function ptw = newPTW()
ptw = physics.MaterialModel(flow_stress_model="PTW_Yield_Stress", ...
                            shear_modulus_model="Stein_Shear_Modulus");
end

function testPhysicsBaseline(tc)
h = tc.TestData.h;
params = struct('theta', 0.1, 'p', 2, 's0', 0.02, 'sInf', 0.01, ...
    'kappa', 0.3, 'lgamma', -12, 'y0', 0.01, 'yInf', 0.003, 'y1', 0.09, 'y2', 0.7);
res = h.run(newPTW(), params, tc.TestData.consts, 0.6, 2500e-6, 100, 1000);
tc.verifyEqual([res(:, 2) res(:, 3)], h.baseline('physics_strainstress_baseline.csv'), ...
    'AbsTol', 1e-8, 'RelTol', 1e-5);
end

function testConstParams(tc)
% calibrate PTW with some of its parameters kept constant; the best draws
% should be within 5 permille of the values that generated the data
h = tc.TestData.h;
consts = tc.TestData.consts;
consts.y1 = 0.09;
consts.y2 = 0.7;
consts.kappa = 0.3;
consts.lgamma = -12;
params = struct('theta', 0.1, 'p', 2, 's0', 0.02, 'sInf', 0.01, ...
    'y0', 0.01, 'yInf', 0.003);
res = h.run(newPTW(), params, consts, 0.6, 2500e-6, 100, 1000);
strains = res(:, 2);
stresses = res(:, 3);
tc.verifyEqual([strains stresses], h.baseline('physics_strainstress_baseline.csv'), ...
    'AbsTol', 1e-8, 'RelTol', 1e-5);

rng(1);
out = h.calibrate(params, consts, strains, stresses, tc.TestData.models, @constraints_ptw);
uu = 1001:5:2000;
fn = fieldnames(out.theta_native);
tc.verifyEqual(sort(fn), sort(fieldnames(params)));
for i = 1:numel(fn)
    v = out.theta_native.(fn{i})(uu);
    tc.verifyLessThan(min(abs(v/params.(fn{i}) - 1)), 5e-3, fn{i});
end

% pairs plot of the posterior draws, the best draw and the posterior mean
llik = out.llik(uu);
[~, best] = max(llik);
th = reshape(out.theta(uu, 1, :), numel(uu), []);
mat = [th; th(best, :); mean(th, 1)];
f = figure('Visible', 'off');
tc.addTeardown(@() close(f));
tc.verifyWarningFree(@() pairs(mat, [], fn));
end

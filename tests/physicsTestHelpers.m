function h = physicsTestHelpers()
%PHYSICSTESTHELPERS Shared helpers for the physics model tests.
h.baseline = @baseline;
h.run = @run_history;
h.bounds = @bounds_around;
h.calibrate = @calibrate;
h.relDiff = @(a, b) abs(a - b)/mean([a, b]);
end

function out = baseline(name)
% strain/stress baseline written by the python package's tests
d = fullfile(fileparts(mfilename('fullpath')), 'data');
out = readmatrix(fullfile(d, name), 'NumHeaderLines', 1);
end

function res = run_history(model, params, consts, emax, edot, nhist, T, options)
% integrate the model from zero stress/strain at temperature T
if nargin < 8
    options = struct();
end
model.set_history_variables(emax, edot, nhist);
model.initialize(params, consts, options);
model.initialize_state(T);
res = model.compute_state_history();
end

function b = bounds_around(params)
% +/- 2% bounds around each parameter value
fn = fieldnames(params);
b = struct();
for i = 1:numel(fn)
    b.(fn{i}) = sort([0.98 1.02]*params.(fn{i}));
end
end

function out = calibrate(params, consts, strains, stresses, models, constraint)
% pooled calibration of a ModelMaterialStrength to one strain history, as
% in the python package's tests
m = ModelMaterialStrength(1000, 2500e-6, consts, {strains}, models{:}, true, 'gibbs');
s = CalibSetup(bounds_around(params), constraint);
s.addVecExperiments(stresses(:)', m, 1, 0, zeros(1, numel(stresses)));
s.setTemperatureLadder(1.05.^(0:19));
s.setMCMC(2000, 0, 1, 100);
evalc('out = calibPool(s);');
end

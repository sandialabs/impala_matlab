M = 100;
t = linspace(0, 2800, M);
params = [24, -1e-3, 1e-9, .005, 2000, 2];

f_exp = (params(1) + (params(2) + params(3) * t).* t) ./ ...
                    (1.0 + exp(params(4) * (t-params(5)))) - params(6);

plot(t,f_exp)

input_names = {'a0', 'a1', 'a2', 'b', 't0', 'k'};
bounds.a0 = [1 40];
bounds.a1 = [-1e-4 1e-4];
bounds.a2 = [-1e-8 1e-8];
bounds.b = [0 1];
bounds.t0 = [1000 3000];
bounds.k = [1 10];

setup = CalibSetup(bounds, @cf_bounds);
model_ftilde = ModelE(t, input_names, f_exp, s2='gibbs', elastic=true);
yobs = [f_exp, zeros(1, M)];
setup.addVecExperiments(yobs, model_ftilde, 0.001, 20, ones(1,2*M));
setup.setTemperatureLadder(1.05.^(0:19));
setup.setMCMC(3000,1000,2,10);
out = calibPool(setup);

uu = 2000:2:3000;

theta = tran_unif(squeeze(out.theta(uu,1,:)), setup.bounds_mat, fieldnames(setup.bounds));
ftilde_pred_obs = setup.models{1}.eval(theta);

figure()
theta_native = out.theta_native;
nplots = length(input_names);
ns1 = ceil(sqrt(nplots));
ns2 = round(sqrt(nplots));
for i = 1:nplots
    subplot(ns1, ns2, i)
    plot(theta_native.(input_names{i})(uu))
    theta_native.(input_names{i}) = theta_native.(input_names{i})(uu)';
    title(input_names{i})
end

figure()
parmat_array = zeros(length(uu),numel(input_names));
for i = 1:numel(input_names)
    parmat_array(:,i) = theta_native.(input_names{i});
end
pair_plot(parmat_array, input_names, params);

figure()
plot(t, ftilde_pred_obs(:,1:M)', 'Color', [0.58,0.70,0.75])  %light blue
hold on
plot(t, f_exp, 'k') 

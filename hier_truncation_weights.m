function res = hier_truncation_weights(setup, calib_out, mcmc_use, temperature)
% Diagnose (and, where feasible, undo) the contraction that the hierarchy's
% truncation to the bounds induces on theta0 and Sigma0.
%
% calibHier is a correct sampler for the joint density
%
%   p(theta0) p(Sigma0) prod_i N(theta_i; theta0, Sigma0) 1{theta_i in B}
%
% but integrating out theta leaves the effective hyperprior equal to the
% stated one times prod_i P(theta_i in B | theta0, Sigma0). That factor
% rewards a central theta0 and a small Sigma0, so the hierarchy looks
% tighter than the prior that was written down. The intended model --
% theta_i truncated normal with its normalising constant -- is recovered by
% reweighting the saved draws with
%
%   w proportional to P(B | theta0, Sigma0)^(-n),
%
% where n = sum(setup.ntheta) is the total number of experiment-level
% theta_i.
%
% Read ess_frac first:
%   near 1:   the bounds are not binding for this run and the uncorrected
%             output needs no adjustment;
%   moderate: the reweighted summaries are usable;
%   small:    the run is materially contracted AND reweighting cannot
%             repair it, because the chain never visited the region the
%             corrected posterior wants. Treat the reported hyperprior as
%             an approximation, or reduce the number of parameters sitting
%             against a bound.
%
% The size of the tilt is n*|log P(B)| in nats; it grows with both the
% number of experiments and the dimension, and is only large when
% parameters sit near the bounds.
%
% setup: the CalibSetup passed to calibHier
% calib_out: the struct calibHier returned
% mcmc_use: iterations to use (e.g. setup.nburn+1:setup.nmcmc); defaults to
%           the second half of the chain
% temperature: temperature index; 1 (the default) is the cold chain
%
% returns a struct with fields weights (normalised, one per used draw),
% ess, ess_frac, log_box_prob, n_theta, tilt_nats, and theta0_mean,
% theta0_sd and sigma0_diag_median each given both _raw and _reweighted
%
% The weights account for the bounds only. With a constraint_func other
% than cf_bounds the true normalising constant is smaller still, so the
% correction is partial and a warning is issued.
arguments
    setup CalibSetup
    calib_out
    mcmc_use = [];
    temperature = 1;
end
theta0 = calib_out.theta0;
Sigma0 = calib_out.Sigma0;
nd = size(theta0, 1);
p = size(theta0, 3);
if isempty(mcmc_use)
    mcmc_use = floor(nd/2)+1 : nd;
end
mcmc_use = reshape(mcmc_use, 1, []);

th0 = reshape(theta0(mcmc_use, temperature, :), [], p);
S0 = reshape(Sigma0(mcmc_use, temperature, :, :), [], p, p);
n_theta = sum(setup.ntheta);

if ~isequal(setup.constraint_func, @cf_bounds)
    warning('hier_truncation_weights:customConstraint', ...
            ['hier_truncation_weights accounts for the bounds only; this ' ...
             'setup uses a custom constraint_func, so the correction is partial.']);
end

nuse = numel(mcmc_use);
lp = zeros(nuse, 1);
for j = 1:nuse
    try
        lp(j) = log_box_prob(th0(j,:), reshape(S0(j,:,:), p, p));
    catch
        lp(j) = NaN;
    end
end
ok = isfinite(lp);
if ~all(ok)
    warning('hier_truncation_weights:unusableDraws', ...
            ['%d of %d draws gave an unusable box probability and were ' ...
             'dropped from the diagnostic.'], sum(~ok), numel(ok));
end
th0 = th0(ok,:);
S0 = S0(ok,:,:);
lp = lp(ok);

lw = -n_theta * lp;
w = exp(lw - max(lw));
w = w / sum(w);
ess = 1 / sum(w.^2);

s0diag = zeros(size(S0, 1), p);
for k = 1:p
    s0diag(:,k) = S0(:,k,k);
end
mean_rw = sum(w .* th0, 1);

res = struct();
res.weights = w;
res.ess = ess;
res.ess_frac = ess / numel(w);
res.log_box_prob = lp;
res.n_theta = n_theta;
res.tilt_nats = n_theta * abs(median(lp));
res.theta0_mean_raw = mean(th0, 1);
res.theta0_mean_reweighted = mean_rw;
res.theta0_sd_raw = std(th0, 1, 1);
res.theta0_sd_reweighted = sqrt(sum(w .* (th0 - mean_rw).^2, 1));
res.sigma0_diag_median_raw = median(s0diag, 1);
res.sigma0_diag_median_reweighted = arrayfun( ...
    @(k) weighted_quantile(s0diag(:,k), 0.5, w), 1:p);
end

function lp = eval_theta_priors(theta, priors)
% Evaluate the log-prior for calibration parameters.
%
% theta: struct of parameter values on the native scale, one field per
%        calibration parameter, each holding a 1 x ntemps row vector (this
%        is exactly what tran_unif returns).
% priors: setup.theta_prior, a cell array of prior specifications built by
%         CalibSetup.addThetaPrior / CalibSetup.addJointThetaPrior. Each
%         element is a struct with either
%           .name, .dist, .params           (independent, marginal prior)
%         or
%           .names, .log_density_fn         (joint prior)
%
% returns a 1 x ntemps row vector of log-prior values. When no priors have
% been added the result is all zeros, so calibration is unaffected.

fn = fieldnames(theta);
lp = zeros(1, numel(theta.(fn{1})));

if isempty(priors)
    return
end

for j = 1:numel(priors)
    p = priors{j};

    if isfield(p, 'names')
        % Joint prior: collect the named parameters into a struct and hand
        % it to the user supplied log-density function. The function is
        % called once per temperature so it only ever sees scalars.
        add = zeros(1, numel(lp));
        for t = 1:numel(lp)
            params_list = struct();
            for k = 1:numel(p.names)
                params_list.(p.names{k}) = theta.(p.names{k})(t);
            end
            add(t) = p.log_density_fn(params_list);
        end
    else
        % Independent prior on a single parameter.
        x = reshape(theta.(p.name), 1, []);
        add = log_dens(p.dist, x, p.params);
    end

    lp = lp + add;
end

end


function ld = log_dens(dist, x, pr)
% Log densities, written out directly so that no Statistics and Machine
% Learning Toolbox call is needed and so that tail values do not underflow.

switch lower(dist)
    case 'normal'
        ld = -log(pr.sd) - 0.5*log(2*pi) - 0.5*((x - pr.mean)./pr.sd).^2;

    case 'lognormal'
        ld = -Inf(size(x));
        ok = x > 0;
        ld(ok) = -log(x(ok)) - log(pr.sdlog) - 0.5*log(2*pi) ...
                 - 0.5*((log(x(ok)) - pr.meanlog)./pr.sdlog).^2;

    case 'beta'
        ld = -Inf(size(x));
        ok = (x > 0) & (x < 1);
        ld(ok) = (pr.shape1 - 1).*log(x(ok)) + (pr.shape2 - 1).*log1p(-x(ok)) ...
                 - betaln(pr.shape1, pr.shape2);

    case 'uniform'
        ld = -Inf(size(x));
        ok = (x >= pr.min) & (x <= pr.max);
        ld(ok) = -log(pr.max - pr.min);

    case 'gamma'
        ld = -Inf(size(x));
        ok = x > 0;
        ld(ok) = pr.shape.*log(pr.rate) + (pr.shape - 1).*log(x(ok)) ...
                 - pr.rate.*x(ok) - gammaln(pr.shape);

    case 'cauchy'
        ld = -log(pi*pr.scale) - log1p(((x - pr.location)./pr.scale).^2);

    otherwise
        error('eval_theta_priors:unsupportedDist', ...
              'Unsupported dist: %s', dist);
end

end

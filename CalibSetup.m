classdef CalibSetup < handle
    % Structure for storing calibration experimental data, likelihood, discrepancy, etc.

    properties
        nexp  % Number of independent emulators
        ys
        y_lens
        models
        tl
        itl
        bounds  % should be a struct so we can use parameter names
        bounds_mat
        p
        constraint_func  % constraint function, see checkConstraints
        constants  % constants of the first model (if it has any), passed to constraint_func
        nmcmc
        nburn
        thin
        decor
        ntemps
        sd_est
        s2_df
        ig_a
        ig_b
        s2_ind
        s2_exp_ind
        ns2
        ny_s2
        ntheta
        theta_ind
        nswap
        nswap_per
        swap_scheme  % 'deo' (default), 'seo' or 'random'; see SwapScheduler
        adapt_ladder  % adapt the temperature ladder during burn-in
        adapt_ladder_until  % last iteration at which the ladder may change
        start_temper
        s2_prior_kern
        start_var_theta
        start_tau_theta
        start_var_ls2
        start_tau_ls2
        start_adapt_iter
        theta0_prior_mean
        theta0_prior_cov
        Sigma0_prior_df
        Sigma0_prior_scale
        nclustmax
        eta_prior_shape
        eta_prior_rate
        scale_proposal
        wt  % cell array of observation weights, one vector per experiment
        sd_lower  % lower bounds on the error sd (only used with 'gibbs_trunc')
        sd_upper  % upper bounds on the error sd (only used with 'gibbs_trunc')
        theta0_start  % optional ntemps x p start for theta0 (0-1 scale)
        theta_start   % optional ntemps x nclustmax x p start for clustered theta
        theta_prior  % cell array of priors on the calibration parameters
    end

    methods
        function obj = CalibSetup(bounds, constraint_func)
            % **Pooled Bayesian Model Calibration**
            %
            % This function setups up calibration object
            %
            % bounds: a struct with fields of variable names with values of 
            %         two dimensional arrays where first element is lower bound
            %         and second element is upper bound.
            % constraint_func: function handle to constraint function on 
            %                  variables. Defaults to function on checking 
            %                  bounds. It is called as f(x, bounds), or as
            %                  f(x, bounds, constants) if it takes three
            %                  or more inputs (e.g. @constraints_ptw)
            % returns an object of class CalibSetup
            arguments
                bounds
                constraint_func = @cf_bounds
            end
            obj.nexp = 0;
            obj.ys = {};
            obj.y_lens = {};
            obj.models = {};
            obj.tl = 1;
            obj.itl = 1/obj.tl;
            obj.bounds = bounds;
            fn = fieldnames(bounds);
            obj.p = numel(fn);
            bmattmp = zeros(obj.p,2);

            for i = 1:obj.p
                bmattmp(i,:) = bounds.(fn{i});
            end
            obj.bounds_mat = bmattmp;
            obj.constraint_func = constraint_func;
            obj.constants = [];
            obj.nmcmc = 10000;
            obj.nburn = 5000;
            obj.thin = 5;
            obj.decor = 100;
            obj.ntemps = 1;
            obj.sd_est = {};
            obj.s2_df = {};
            obj.ig_a = {};
            obj.ig_b = {};
            obj.s2_ind = {};
            obj.s2_exp_ind = {};
            obj.ns2 = {};
            obj.ny_s2 = {};
            obj.ntheta = [];
            obj.theta_ind = {};
            obj.nswap = 5;
            obj.nswap_per = 0;
            obj.swap_scheme = 'deo';
            obj.adapt_ladder = false;
            obj.adapt_ladder_until = [];
            obj.start_temper = 1000;
            obj.s2_prior_kern = {};
            obj.sd_lower = {};
            obj.sd_upper = {};
            obj.wt = {};
            obj.theta_prior = {};
            obj.start_var_theta = 1e-8;
            obj.start_tau_theta = 0;
            obj.start_var_ls2 = 1e-5;
            obj.start_tau_ls2 = 0;
            obj.start_adapt_iter = 300;
            obj.nclustmax = 10;
            obj.eta_prior_shape = 2;
            obj.eta_prior_rate = 0.1;
            obj.scale_proposal = false;
        end

        function good = checkConstraints(obj, x, bounds, constants)
            % Calls the constraint function on the parameter struct x.
            %
            % bounds: overrides obj.bounds (optional)
            % constants: overrides obj.constants (optional); only passed to
            %            constraint functions taking three or more inputs
            arguments
                obj
                x
                bounds = obj.bounds;
                constants = obj.constants;
            end
            f = obj.constraint_func;
            n = nargin(f);
            if n < 0 || n >= 3
                good = f(x, bounds, constants);
            else
                good = f(x, bounds);
            end
        end

        function obj = addThetaPrior(obj, dist, params, pname)
            % This method adds a prior for the specified calibration
            % parameter. Priors are evaluated on the native (unnormalized)
            % scale of the parameter, i.e. the scale given by bounds.
            %
            % dist: string naming the distribution, one of 'normal',
            %       'lognormal', 'beta', 'uniform', 'gamma', 'cauchy'
            %       (default: 'uniform')
            % params: struct of distribution parameters, e.g. for
            %         dist='normal', struct('mean',0,'sd',1). Expected
            %         fields are
            %           normal    : mean, sd
            %           lognormal : meanlog, sdlog
            %           beta      : shape1, shape2
            %           uniform   : min, max
            %           gamma     : shape, rate
            %           cauchy    : location, scale
            % pname: name of the calibration parameter, must be one of the
            %        field names of bounds
            %
            % returns an object of class CalibSetup
            %
            % Example:
            %   bounds.t_1 = [0, 1];
            %   setup = CalibSetup(bounds, @cf_bounds);
            %   setup = setup.addThetaPrior('normal', ...
            %                               struct('mean',0.5,'sd',0.1), 't_1');
            arguments
                obj
                dist = 'uniform';
                params = struct('min',0,'max',1);
                pname = '';
            end

            pnames = fieldnames(obj.bounds);
            if isempty(pname) || ~any(strcmp(pname, pnames))
                error('CalibSetup:addThetaPrior:badName', ...
                      ['No parameter name given or parameter not in set of ' ...
                       'input names for any model in setup.models']);
            end
            required = struct('normal', {{'mean','sd'}}, ...
                              'lognormal', {{'meanlog','sdlog'}}, ...
                              'beta', {{'shape1','shape2'}}, ...
                              'uniform', {{'min','max'}}, ...
                              'gamma', {{'shape','rate'}}, ...
                              'cauchy', {{'location','scale'}});
            if ~isfield(required, lower(dist))
                error('CalibSetup:addThetaPrior:badDist', ...
                      'Unsupported dist: %s. Supported distributions are %s.', ...
                      dist, strjoin(fieldnames(required), ', '));
            end
            missing = setdiff(required.(lower(dist)), fieldnames(params), 'stable');
            if ~isempty(missing)
                error('CalibSetup:addThetaPrior:missingParams', ...
                      'Missing parameters %s for dist ''%s''.', ...
                      strjoin(missing, ', '), dist);
            end

            obj.theta_prior{end+1} = struct('name', pname, 'dist', dist, ...
                                            'params', params);
        end

        function obj = addJointThetaPrior(obj, pnames, log_density_fn)
            % This method adds a joint prior over multiple calibration
            % parameters. Priors are evaluated on the native (unnormalized)
            % scale of the parameters.
            %
            % pnames: cell array of parameter names the joint prior applies
            %         to, each must be a field name of bounds
            % log_density_fn: function handle taking a struct of scalar
            %                 parameter values (fields named by pnames) and
            %                 returning a scalar log-density
            %
            % returns an object of class CalibSetup
            %
            % Example:
            %   % joint normal prior on t_1 and t_2
            %   my_joint_prior = @(pp) log(mvnpdf([pp.t_1, pp.t_2], ...
            %                                     [0, 0], [1 0.5; 0.5 1]));
            %   bounds.t_1 = [0, 1];
            %   bounds.t_2 = [0, 1];
            %   setup = CalibSetup(bounds, @cf_bounds);
            %   setup = setup.addJointThetaPrior({'t_1','t_2'}, my_joint_prior);
            arguments
                obj
                pnames
                log_density_fn
            end

            if ischar(pnames) || isstring(pnames)
                pnames = cellstr(pnames);
            end
            pnames = reshape(pnames, 1, []);

            valid_names = fieldnames(obj.bounds);
            if ~all(ismember(pnames, valid_names))
                error('CalibSetup:addJointThetaPrior:badNames', ...
                      ['One or more parameter names not in set of input ' ...
                       'names for models in setup']);
            end

            if ~isa(log_density_fn, 'function_handle')
                error('CalibSetup:addJointThetaPrior:badFunction', ...
                      'log_density_fn must be a function handle');
            end

            obj.theta_prior{end+1} = struct('names', {pnames}, ...
                                            'log_density_fn', log_density_fn);
        end

        function obj = addVecExperiments(obj, yobs, model, sd_est, s2_df, s2_ind, meas_error_cor, theta_ind, D, discrep_tau, sd_lower, sd_upper, wt)
            % This method adds vector experiments to calibration object
            %
            % yobs: a vector of the experiment or observation
            % model: emulator or model object (e.g. ModelBassPca_func,
            %        ModelBassPca_func_mf, ModelBpprPca_func, ModelmvBayes,
            %        ModelF)
            % sd_est: estimate of standard deviation, one per error group
            % s2_df: degrees of freedom of inverse gamma prior (scalar or
            %        one per error group). s2_df = 0 gives an improper
            %        prior: p(s2) proportional to 1/s2 when s2 is updated
            %        by Gibbs sampling (model.s2 'gibbs' or 'gibbs_trunc'),
            %        and a half-Cauchy-type kernel 1/(1 + s2) when it is
            %        updated by Metropolis-Hastings (model.s2 'MH')
            % s2_ind: error group (1..length(sd_est)) of each element of yobs
            % meas_error_cor: measurement error correlation (default: NaN)
            % theta_ind: theta index of each element of yobs, used by
            %            hierarchical and clustered calibration (default:
            %            NaN, a single theta). Any distinct labels are
            %            allowed; they are mapped to 1..ntheta.
            % D: discrepency basis (matrix of columns of basis, default:
            %    NaN)
            % discrep_tau: discrepency sampling tau
            % sd_lower, sd_upper: bounds on the error sd, one per error
            %                     group, used when model.s2 is 'gibbs_trunc'
            % wt: positive weight of each element of yobs (default: all
            %     ones). A weight acts like a fractional number of
            %     replicates: it scales that observation's squared residual
            %     and its log-variance term in the likelihood.
            % returns an object of class CalibSetup
            arguments
                obj
                yobs (1,:) {mustBeNumeric}
                model
                sd_est
                s2_df
                s2_ind
                meas_error_cor=NaN;
                theta_ind=NaN;
                D=NaN;
                discrep_tau=1;
                sd_lower=[];
                sd_upper=[];
                wt=[];
            end

            n = length(yobs);
            sd_est = reshape(sd_est, 1, []);
            ns2 = length(sd_est);
            s2_ind = reshape(s2_ind, 1, []);
            if isscalar(s2_ind) && n > 1
                s2_ind = repmat(s2_ind, 1, n);
            end
            if any(s2_ind == 0)
                % allow 0-based indices (as used by the python package)
                s2_ind = s2_ind + 1;
            end
            if length(s2_ind) ~= n
                error('CalibSetup:addVecExperiments:badS2Ind', ...
                      'length(s2_ind) must equal length(yobs)');
            end
            if any(s2_ind < 1) || any(s2_ind > ns2) || any(s2_ind ~= round(s2_ind))
                error('CalibSetup:addVecExperiments:badS2Ind', ...
                      's2_ind must take integer values between 1 and length(sd_est)');
            end
            if isempty(wt)
                wt = ones(1, n);
            end
            wt = reshape(double(wt), 1, []);
            if isscalar(wt) && n > 1
                wt = repmat(wt, 1, n);
            end
            if length(wt) ~= n || any(~isfinite(wt)) || any(wt <= 0)
                error('CalibSetup:addVecExperiments:badWt', ...
                      'wt must have one positive weight per element of yobs');
            end
            s2_df = reshape(s2_df, 1, []);
            if isscalar(s2_df)
                s2_df = repmat(s2_df, 1, ns2);
            end

            if isscalar(theta_ind) && isnan(theta_ind)
                theta_ind = ones(1,n);
            end
            theta_ind = reshape(theta_ind, 1, []);
            if length(theta_ind) ~= n
                error('CalibSetup:addVecExperiments:badThetaInd', ...
                      'length(theta_ind) must equal length(yobs)');
            end
            [~, ~, theta_ind] = unique(theta_ind, 'stable');
            theta_ind = reshape(theta_ind, 1, []);
            ntheta = max(theta_ind);
            for k = 1:ns2
                if numel(unique(theta_ind(s2_ind == k))) > 1
                    warning('CalibSetup:addVecExperiments:sharedError', ...
                            'Cannot have multiple thetas with shared measurement error.');
                    break
                end
            end

            valid_s2 = {'gibbs','gibbs_trunc','MH','fix'};
            if ~any(strcmp(model.s2, valid_s2))
                error('CalibSetup:addVecExperiments:badS2', ...
                      'model.s2 must be one of: %s', strjoin(valid_s2, ', '));
            end
            if strcmp(model.s2, 'gibbs_trunc')
                if isempty(sd_lower) || isempty(sd_upper)
                    error('CalibSetup:addVecExperiments:needBounds', ...
                          'sd_lower and sd_upper are required when model.s2 is gibbs_trunc');
                end
            end
            if ~isempty(sd_lower)
                sd_lower = reshape(sd_lower, 1, []);
                sd_upper = reshape(sd_upper, 1, []);
            end

            obj.ys{end+1} = yobs;
            obj.y_lens{end+1} = n;
            model.exp_ind = theta_ind;
            model.nexp = ntheta;
            obj.theta_ind{end+1} = theta_ind;
            obj.ntheta = [obj.ntheta ntheta];
            model.yobs = yobs;
            if ~all(isnan(meas_error_cor(:)))
                model.meas_error_cor = meas_error_cor;
            end

            if ~all(isnan(D(:)))
                model.D = D;
                model.nd = size(D,2);
                model.discrep_tau = discrep_tau;
            end

            obj.models{end+1} = model;
            if isprop(obj.models{1}, 'constants')
                obj.constants = obj.models{1}.constants;
            end
            obj.nexp = obj.nexp + 1;
            obj.sd_est{end+1} = sd_est;
            obj.wt{end+1} = wt;
            obj.sd_lower{end+1} = sd_lower;
            obj.sd_upper{end+1} = sd_upper;
            obj.s2_df{end+1} = s2_df;
            obj.ig_a{end+1} = s2_df/2;
            obj.ig_b{end+1} = s2_df/2 .* sd_est.^2;
            obj.s2_ind{end+1} = s2_ind;
            obj.s2_exp_ind{end+1} = 1:ns2;
            obj.ns2{end+1} = ns2;
            % (weighted) number of observations in each error group, used by
            % the Gibbs update of s2; equals the count when all weights are 1
            vec = zeros(1, ns2);
            for i = 1:ns2
                vec(i) = sum(wt(s2_ind==i));
            end
            obj.ny_s2{end+1} = vec;
            obj.nclustmax = max(sum(obj.ntheta),10);
            if any(s2_df == 0)
                obj.s2_prior_kern{end+1} = @ldhc_kern;
            else
                obj.s2_prior_kern{end+1} = @ldig_kern;
            end
        end

        function obj = setTemperatureLadder(obj, temperature_ladder, start_temper, swap_scheme, adapt_ladder, adapt_ladder_until)
            % This function setups up temperature ladder for tempering
            %
            % temperature_ladder: array of increasing temperatures, all
            %                     at least 1, e.g. 1.05.^(0:19)
            % start_temper: what MCMC sample to start tempering (default:
            %               1000)
            % swap_scheme: how swap pairs are chosen in each swap round.
            %              'deo' (default) uses deterministic even/odd
            %              swaps between adjacent temperatures
            %              (non-reversible parallel tempering, Syed et al.
            %              2022); 'seo' picks the even/odd parity at
            %              random; 'random' reproduces the scheme of
            %              earlier versions (random disjoint pairs from the
            %              whole ladder)
            % adapt_ladder: if true, re-space the temperatures during
            %               burn-in so that the swap rejection rate is the
            %               same between every pair of neighbouring
            %               temperatures (Syed et al. 2022, Algorithm 4).
            %               The coldest and hottest temperatures are kept
            %               fixed. Experimental (default: false)
            % adapt_ladder_until: last MCMC iteration at which the ladder
            %                     may change (default: nburn from setMCMC
            %                     if positive, otherwise floor(nmcmc/2)).
            %                     Discard samples up to this iteration as
            %                     burn-in. The adapted ladder is returned
            %                     as temperature_ladder in the output.
            % returns an object of class CalibSetup
            arguments
                obj
                temperature_ladder
                start_temper = 1000;
                swap_scheme = 'deo';
                adapt_ladder = false;
                adapt_ladder_until = [];
            end
            SwapScheduler.validate(swap_scheme, adapt_ladder);
            obj.tl = reshape(double(temperature_ladder), 1, []);
            obj.itl = 1./obj.tl;
            obj.ntemps = length(obj.tl);
            obj.nswap_per = floor(obj.ntemps / 2);
            obj.start_temper = start_temper;
            obj.swap_scheme = char(swap_scheme);
            obj.adapt_ladder = logical(adapt_ladder);
            obj.adapt_ladder_until = adapt_ladder_until;
        end

        function sched = makeSwapScheduler(obj)
            % Create the SwapScheduler used by the samplers.
            until = obj.adapt_ladder_until;
            if isempty(until)
                if obj.nburn > 0
                    until = obj.nburn;
                else
                    until = floor(obj.nmcmc / 2);
                end
            end
            sched = SwapScheduler(obj.itl, obj.swap_scheme, obj.adapt_ladder, until);
        end

        function obj = setMCMC(obj, nmcmc, nburn, thin, decor, start_var_theta, start_tau_theta, start_var_ls2, start_tau_ls2, start_adapt_iter)
            % This function setups up MCMC parameters for adapative MCMC,
            % also includes tempering and decorrelation steps
            %
            % nmcmc: number of mcmc iterations
            % nburn: number of mcmc burn in iterations (default: 0)
            % thin: number of samples to thin (default: 1)
            % decor: number of mcmc iterations before decorrelation step
            %        (default: 100)
            % start_var_theta: start variance of theta proposal (default: 1e-8)
            % start_tau_theta: start tau of theta (default: 0)
            % start_var_ls2: start variance of sigma proposal (default: 1e-5)
            % start_tau_ls2: start tau of sigma (default: 0)
            % start_adpat_iter: number of iterations before adapation
            %                   (default: 300)
            %
            % returns an object of class CalibSetup
            arguments
                obj
                nmcmc
                nburn=0;
                thin=1;
                decor=100;
                start_var_theta=1e-8;
                start_tau_theta = 0.;
                start_var_ls2=1e-5;
                start_tau_ls2=0.;
                start_adapt_iter=300;
            end
            obj.nmcmc = nmcmc;
            obj.nburn = nburn;
            obj.thin = thin;
            obj.decor = decor;
            obj.start_var_theta = start_var_theta;
            obj.start_tau_theta = start_tau_theta;
            obj.start_var_ls2 = start_var_ls2;
            obj.start_tau_ls2 = start_tau_ls2;
            obj.start_adapt_iter = start_adapt_iter;
        end

        function obj = setHierPriors(obj, theta0_prior_mean, theta0_prior_cov, Sigma0_prior_df, Sigma0_prior_scale)
            % Define hierarchical model hyperparameters, where
            % theta_i ~ N(theta0, Sigma0), theta0 ~ N(theta0_prior_mean,
            % theta0_prior_cov) and Sigma0 ~ InvWishart(Sigma0_prior_df,
            % Sigma0_prior_scale). Everything is on the 0-1 scale, and
            % theta_i is restricted to the bounds. The prior mean of Sigma0
            % is Sigma0_prior_scale/(Sigma0_prior_df-p-1).
            %
            % The restriction of theta_i to the bounds is applied without
            % its normalising constant, so the effective prior on
            % (theta0, Sigma0) is the stated prior times
            % prod_i P(theta_i in bounds | theta0, Sigma0), which favours
            % small Sigma0 and theta0 away from the bounds.
            % hier_truncation_weights measures that contraction for a
            % finished run and reweights the output to remove it.
            %
            % theta0_prior_mean: 1 x p vector, usually 0.5*ones(1,p)
            % theta0_prior_cov: p x p matrix
            % Sigma0_prior_df: prior degrees of freedom (> p+1); larger
            %                  values pull the theta_i toward theta0
            % Sigma0_prior_scale: p x p prior scale; smaller values pull
            %                     the theta_i toward theta0
            %
            % returns an object of class CalibSetup
            arguments
                obj
                theta0_prior_mean
                theta0_prior_cov
                Sigma0_prior_df
                Sigma0_prior_scale
            end
            if numel(theta0_prior_mean) ~= obj.p || ~isequal(size(theta0_prior_cov), [obj.p obj.p]) ...
                    || ~isequal(size(Sigma0_prior_scale), [obj.p obj.p])
                error('CalibSetup:setHierPriors:badSize', ...
                      'theta0_prior_mean must have p elements and the matrices must be p x p');
            end
            if Sigma0_prior_df <= obj.p + 1
                error('CalibSetup:setHierPriors:badDf', ...
                      'Sigma0_prior_df must be greater than p+1 so the prior mean exists');
            end
            obj.theta0_prior_mean = reshape(theta0_prior_mean, 1, []);
            obj.theta0_prior_cov = theta0_prior_cov;
            obj.Sigma0_prior_df = Sigma0_prior_df;
            obj.Sigma0_prior_scale = Sigma0_prior_scale;
        end

        function theta_new = sample_hier_parent(obj, theta0, Sigma0, maxiter)
            % Draw parameters for a new experiment from the fitted
            % hierarchical distribution, theta_new ~ N(theta0, Sigma0), one
            % draw per row, on the 0-1 scale, redrawing rows that violate
            % the constraints.
            %
            % theta0: n x p theta0 samples on the 0-1 scale, as returned
            %         in calibHier(...).theta0 (e.g. squeeze(out.theta0(:,1,:)))
            % Sigma0: n x p x p Sigma0 samples, as returned by calibHier
            % maxiter: maximum number of redraws (default 10000)
            %
            % returns an n x p matrix on the 0-1 scale
            arguments
                obj
                theta0
                Sigma0
                maxiter = 10000;
            end
            n = size(theta0, 1);
            p = size(theta0, 2);
            Sigma0 = reshape(Sigma0, n, p, p);
            theta_new = sample_constrained(theta0, Sigma0, obj, maxiter);
        end

        function obj = setClusterPriors(obj, nclustmax, eta_prior_shape, eta_prior_rate, scale_proposal)
            % Define clustered calibration hyperparameters. The Dirichlet
            % process concentration eta ~ Gamma(eta_prior_shape,
            % eta_prior_rate), so E(eta) = shape/rate; larger values favor
            % more clusters. Clustered calibration also uses the
            % hierarchical priors from setHierPriors.
            %
            % nclustmax: maximum number of clusters (default:
            %            max(sum(ntheta),10))
            % eta_prior_shape: gamma shape (default: 2)
            % eta_prior_rate: gamma rate (default: 0.1)
            % scale_proposal: divide each cluster's proposal covariance by
            %                 its number of members, which can help when
            %                 cluster sizes are very uneven (default: false)
            %
            % returns an object of class CalibSetup
            arguments
                obj
                nclustmax = max(sum(obj.ntheta), 10);
                eta_prior_shape = 2;
                eta_prior_rate = 0.1;
                scale_proposal = false;
            end
            obj.nclustmax = nclustmax;
            obj.eta_prior_shape = eta_prior_shape;
            obj.eta_prior_rate = eta_prior_rate;
            obj.scale_proposal = scale_proposal;
        end
    end
end

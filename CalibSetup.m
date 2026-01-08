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
        checkConstraints
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
            %                  bounds
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
            obj.checkConstraints = constraint_func;
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
            obj.s2_prior_kern = {};
        end

        function obj = addVecExperiments(obj, yobs, model, sd_est, s2_df, s2_ind, meas_error_cor, theta_ind, D, discrep_tau)
            % This method adds vector experiments to calibration object
            %
            % yobs: a vector of the experiment or observation
            % model: emulator (currently expecting a object of class
            %        ModelBassPca_func, ModelBassPca_func_mf, or
            %        ModelBpprPca_func
            % sd_est: estimate of standard deviation
            % s2_df: degrees of freedom of inverse gamma prior
            % s2_ind: indices of function
            % meas_error_cor: measurement error correlation (default: NaN) 
            % theta_ind: indices of theta (default: NaN)
            % D: discrepency basis (matrix of columns of basis, default:
            %    NaN)
            % discrep_tau: discrepency sampling tau    
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
            end

            obj.ys{end+1} = yobs;
            obj.y_lens{end+1} = length(yobs);
            if isnan(theta_ind)
                theta_ind = zeros(1,length(yobs));
            end

            model.exp_ind = theta_ind;
            obj.theta_ind{end+1} = theta_ind;
            obj.ntheta = [obj.ntheta length(unique(theta_ind))];
            model.yobs = yobs;
            if ~isnan(meas_error_cor)
                model.meas_error_cor = meas_error_cor;
            end

            if ~isnan(D)
                model.D = D;
                model.nd = size(D,2);
                model.discrep_tau = discrep_tau;
            end

            obj.models{end+1} = model;
            obj.nexp = obj.nexp + 1;
            obj.sd_est{end+1} = sd_est;
            obj.s2_df{end+1} = s2_df;
            obj.ig_a{end+1} = s2_df/2;
            obj.ig_b{end+1} = s2_df/2 * sd_est^2;
            obj.s2_ind{end+1} = s2_ind;
            obj.s2_exp_ind{end+1} = 1:length(sd_est);
            obj.ns2{end+1} = length(sd_est);
            vec = zeros(length(sd_est), 1);
            for i = 1:length(vec)
                vec(i) = sum(s2_ind==i);
            end
            obj.ny_s2{end+1} = vec;
            obj.nclustmax = max(sum(obj.ntheta),10);
            if sum(s2_df == 0) > 1
                obj.s2_prior_kern{end+1} = @ldhc_kern;
            else
                obj.s2_prior_kern{end+1} = @ldig_kern;
            end
        end

        function obj = setTemperatureLadder(obj, temperature_ladder, start_temper)
            % This function setups up temperature ladder for tempering
            %
            % temperature_ladder: array of temperatures
            % start_temper: what MCMC sample to start tempering (default:
            %               1000)
            % returns an object of class CalibSetup
            arguments
                obj
                temperature_ladder
                start_temper = 1000;
            end
            obj.tl = temperature_ladder;
            obj.itl = 1./obj.tl;
            obj.ntemps = length(obj.tl);
            obj.nswap_per = floor(obj.ntemps / 2);
            obj.start_temper = start_temper;
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
            obj.theta0_prior_mean = theta0_prior_mean;
            obj.theta0_prior_cov = theta0_prior_cov;
            obj.Sigma0_prior_df = Sigma0_prior_df;
            obj.Sigma0_prior_scale = Sigma0_prior_scale;
        end

        function obj = setClusterPriors(obj, nclustmax)
            obj.nclustmax = nclustmax;
        end
    end
end

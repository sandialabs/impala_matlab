classdef ModelF < handle
    % Custom simulator/emulator model defined by a MATLAB function

    properties
        mod
        stochastic
        input_names
        meas_error_cor
        yobs
        nd
        discrep_tau
        D
        nexp
        exp_ind
        s2
    end

    methods
        function obj = ModelF(f, input_names, exp_ind, s2)
            % **Custom Model**
            %
            % f: function handle taking an n x p matrix of inputs (columns
            %    ordered as input_names, on the native scale) and returning
            %    an n x ny matrix of predictions (one row per input row)
            % input_names: cell array of strings of input variable names
            % exp_ind: experiment (theta) indices (default: NaN)
            % s2: how to sample error variance, one of 'gibbs' (default),
            %     'gibbs_trunc', 'MH', or 'fix'
            %
            % returns an object of class ModelF
            arguments
                f function_handle
                input_names
                exp_ind = NaN;
                s2 = 'gibbs';
            end
            obj.mod = f;
            obj.stochastic = false;
            obj.input_names = input_names;
            obj.meas_error_cor = 1;
            obj.yobs = NaN;
            obj.nd = 0;
            obj.discrep_tau = 1;
            obj.D = NaN;
            if isnan(exp_ind)
                exp_ind = 1;
            end
            obj.nexp = max(exp_ind);
            obj.exp_ind = exp_ind;
            obj.s2 = s2;
        end

        function obj = step(obj)
            % deterministic model, nothing to update
        end

        function pred = eval(obj, parmat, pool, ~)
            arguments
                obj
                parmat
                pool = true;
                ~
            end
            fn = obj.input_names;
            parmat_array = zeros(length(parmat.(fn{1})), numel(fn));
            for i = 1:numel(fn)
                parmat_array(:,i) = parmat.(fn{i});
            end
            pred = obj.mod(parmat_array);
            pred = reshape(pred, size(parmat_array,1), []);
            if ~pool
                pred = unpool_pred(pred, obj.exp_ind);
            end
        end

        function out = llik(~, yobs, pred, cov)
            vec = yobs(:) - pred(:);
            out = -0.5*cov.ldet - 0.5*sum(vec.^2 .* cov.inv(:));
        end

        function out = lik_cov_inv(~, s2vec, ~)
            % diagonal covariance; the optional third argument (observation
            % indices) is not needed here
            out.inv = 1 ./ s2vec(:);
            out.ldet = sum(log(s2vec(:)));
        end

        function discrep_vars = discrep_sample(obj, yobs, pred, cov, itemp)
            S = eye(obj.nd) ./ obj.discrep_tau + obj.D' * (cov.inv(:) .* obj.D);
            m = obj.D' * (cov.inv(:) .* (yobs(:) - pred(:)));
            Sinv = inv(S);
            Sinv = (Sinv + Sinv')/2;
            discrep_vars = chol_sample(Sinv*m, Sinv./itemp);
        end
    end
end

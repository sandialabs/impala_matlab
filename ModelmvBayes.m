classdef ModelmvBayes < handle
    % PCA Based Model Emulator using mvBayes object

    properties
        model
        stochastic
        nmcmc
        input_names
        basis
        meas_error_cor
        discrep_cov
        ii
        trunc_error_cov
        mod_s2
        emu_vars
        yobs
        marg_lik_cov
        discrep_vars
        nd
        discrep_tau
        D
        discrep
        nexp
        exp_ind
        s2
    end

    methods
        function obj = ModelmvBayes(bmod, input_names, exp_ind, s2)
            % **PCA Based Model Emulator using mvBayes Framework**
            %
            % This function setups up emulator object
            %
            % bmod: a object of the type mvBayes
            % input_names: cell array of strings of input variable names
            % exp_ind: experiment indices (default: NaN)
            % s2: how to sample error variance (default: 'MH')
            % 
            % returns an object of class ModelmvBayes
            arguments
                bmod mvBayes
                input_names
                exp_ind = NaN;
                s2 = 'MH';
            end

            obj.model = bmod;
            obj.stochastic = true;
            obj.nmcmc = length(bmod.bmList{1}.samples.s2);
            obj.input_names = input_names;
            obj.basis = obj.model.basisInfo.basis';
            obj.meas_error_cor = eye(size(obj.basis,1));
            obj.discrep_cov = eye(size(obj.basis,1))*1e-12;
            obj.ii = 1;
            npc = obj.model.basisInfo.nBasis;
            obj.trunc_error_cov = cov(obj.model.basisInfo.truncError);
            obj.mod_s2 = zeros(obj.nmcmc, npc);
            for i = 1:npc
                obj.mod_s2(:,i) = obj.model.bmList{i}.samples.s2;
            end
            obj.emu_vars = obj.mod_s2(obj.ii,:);
            obj.yobs = NaN;
            obj.marg_lik_cov = NaN;
            obj.discrep_vars = NaN;
            obj.nd = 0;
            obj.discrep_tau = 1.;
            obj.D = NaN;
            obj.discrep = 0.;
            if isnan(exp_ind)
                exp_ind = 1;
            end
            obj.nexp = max(exp_ind);
            obj.exp_ind = exp_ind;
            obj.s2 = s2;
            if strcmp(s2,'gibbs')
                error( "Cannot use Gibbs s2 for emulator models.")
            end
        end

        function obj = step(obj)
            obj.ii = randsample(1:obj.nmcmc,1);
            obj.emu_vars = obj.mod_s2(obj.ii,:);
        end

        function discrep_vars = discrep_sample(obj, yobs, pred, cov, itemp)
            S = eye(obj.nd) ./ obj.discrep_tau + obj.D'*cov.inv*obj.D;
            m = obj.D' * cov.inv * (yobs-pred)';
            discrep_vars = chol_sample(S\m, S./itemp);
        end

        function pred = eval(obj, parmat, pool, nugget)
            arguments
                obj
                parmat
                pool = true;
                nugget = false;
            end

            fn = obj.input_names;
            parmat_array = zeros(length(parmat.(fn{1})),numel(fn));
            for i = 1:numel(fn)
                parmat_array(:,i) = parmat.(fn{i});
            end
            if pool
                pred = obj.model.predict(parmat_array, obj.ii, nugget);
                pred = squeeze(pred);
            else
                keyboard
            end

        end

        function out = llik(~, yobs, pred, cov)
            vec = yobs(:) - pred(:);
            out = -0.5*(cov.ldet + vec'*cov.inv*vec);
        end
        
        function out = lik_cov_inv(obj, s2vec)
            n = length(s2vec);
            Sigma = cor2cov(obj.meas_error_cor(1:n,1:n), sqrt(s2vec));
            mat = Sigma + obj.trunc_error_cov + obj.discrep_cov + obj.basis * diag(obj.emu_vars) * obj.basis';
            try
                R = chol(mat);
            catch
                R = chol(mat+.0001*eye(size(mat,1)));
            end
            out.ldet = 2 * sum(log(diag(R)));
            out.inv = inv(mat);
        end

    end
end

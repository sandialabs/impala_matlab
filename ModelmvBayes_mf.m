classdef ModelmvBayes_mf < handle
    % PCA Based Model Emulator using mvBayes object

    properties
        model
        mod_corr
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
        function obj = ModelmvBayes_mf(bmod, bmod_corr, input_names, exp_ind, s2)
            % **PCA Based Model Emulator using mvBayes MultiFidelity Framework**
            %
            % This function setups up emulator object
            %
            % bmod: a object of the type mvBayes
            % bmod_corr: a cell array of objects of the type mvBayes,
            %            these are the corrections to the LF bmod
            % input_names: cell array of strings of input variable names
            % exp_ind: experiment indices (default: NaN)
            % s2: how to sample error variance (default: 'MH')
            %
            % returns an object of class ModelmvBayes_mf
            arguments
                bmod mvBayes
                bmod_corr cell
                input_names
                exp_ind = NaN;
                s2 = 'MH';
            end

            obj.model = bmod;
            obj.mod_corr = bmod_corr;
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
            Sinv = inv(S);
            Sinv = (Sinv + Sinv')/2;
            discrep_vars = chol_sample(Sinv*m, Sinv./itemp);
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
            pred = obj.model.predict(parmat_array, 'idxSamples', obj.ii);
            pred = squeeze(pred);
            if size(pred,2) == 1
                pred = pred';
            end
            for i = 1:length(obj.mod_corr)
                pred1 = obj.mod_corr{i}.predict(parmat_array, 'idxSamples', obj.ii);
                pred1 = squeeze(pred1);
                if size(pred1,2) == 1
                    pred1 = pred1';
                end
                pred = pred + pred1;
            end
            % some predict methods drop the singleton dimension for a single row
            pred = reshape(pred, size(parmat_array,1), []);
            if ~pool
                pred = unpool_pred(pred, obj.exp_ind);
            end

        end

        function out = llik(~, yobs, pred, cov)
            vec = yobs(:) - pred(:);
            out = -0.5*(cov.ldet + vec'*cov.inv*vec);
        end

        function out = lik_cov_inv(obj, s2vec, inds)
            % inds: indices of the observations s2vec refers to (default:
            %       all), used by hierarchical/clustered calibration
            arguments
                obj
                s2vec
                inds = 1:numel(s2vec);
            end
            Sigma = cor2cov(obj.meas_error_cor(inds,inds), sqrt(s2vec(:)));
            B = obj.basis(inds,:);
            mat = Sigma + obj.trunc_error_cov(inds,inds) + obj.discrep_cov(inds,inds) + B * diag(obj.emu_vars) * B';
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

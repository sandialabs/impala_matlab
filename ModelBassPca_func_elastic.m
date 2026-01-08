classdef ModelBassPca_func_elastic < handle
    % PCA Based Model Emulator

    properties
        model
        mod_warp
        stochastic
        nmcmc
        input_names
        basis
        meas_error_cor
        discrep_cov
        ii
        trunc_error_var
        mod_s2
		mod_warp_s2
        emu_vars
		emu_warp_vars
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
        function obj = ModelBassPca_func_elastic(bmod, bmod_warp, input_names, exp_ind, s2)
            % **PCA Based Model Emulator using BASS MultiFidelity**
            %
            % This function setups up emulator object
            %
            % bmod: a object of the type BassBassis
            % bmod_warp: a object of the type BassBassis
            % input_names: cell array of strings of input variable names
            % exp_ind: experiment indices (default: NaN)
            % s2: how to sample error variance (default: 'MH')
            % 
            % returns an object of class ModelBassPca_func
            arguments
                bmod BassBasis
                bmod_warp BassBasis
                input_names
                exp_ind = NaN;
                s2 = 'MH';
            end

            obj.model = bmod;
            obj.mod_warp = bmod_warp;
            obj.stochastic = true;
            obj.nmcmc = length(bmod.bm_list{1}.samples.s2);
            obj.input_names = input_names;
            obj.basis = obj.model.basis;
            obj.meas_error_cor = eye(size(obj.basis,1));
            obj.discrep_cov = eye(size(obj.basis,1))*1e-12;
            obj.ii = 1;
            npc = obj.model.nbasis;
            if npc > 1
                obj.trunc_error_var = diag(cov(obj.model.trunc_error'))+diag(cov(obj.mod_warp.trunc_error'));
            else
                obj.trunc_error_var = diag(reshape(cov(obj.model.trunc_error'),1,1)) + diag(reshape(cov(obj.mod_warp.trunc_error'),1,1));
            end
            obj.mod_s2 = zeros(obj.nmcmc, npc);
            for i = 1:npc
                obj.mod_s2(:,i) = obj.model.bm_list{i}.samples.s2;
            end
			obj.mod_warp_s2 = zeros(obj.nmcmc, obj.mod_warp.nbasis);
            for i = 1:obj.mod_warp.nbasis
                obj.mod_warp_s2(:,i) = obj.mod_warp.bm_list{i}.samples.s2;
            end
            obj.emu_vars = obj.mod_s2(obj.ii,:);
            obj.emu_warp_vars = obj.mod_warp_s2(obj.ii,:);
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
			obj.emu_warp_vars = obj.mod_warp_s2(obj.ii,:);
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
                predf = obj.model.predict(parmat_array, obj.ii, nugget);
                predw = obj.mod_warp.predict(parmat_array, obj.ii, nugget);
                gam = v_to_gam(predw);
                pred = predf;
                for i = 1:size(gam,1)
                    pred(i,:) = warp_f_gamma(predf(i,:),gam(i,:),linspace(0,1,size(gam,2)));
                end
            else
                keyboard
            end

        end

        function out = llik(~, yobs, pred, cov)
            vec = yobs(:) - pred(:);
            out = -0.5*(cov.ldet + vec'*cov.inv*vec);
        end
        
        function out = lik_cov_inv(obj, s2vec)
            vec = obj.trunc_error_var + s2vec(:);
            Ainv = diag(1./vec);
            Aldet = sum(log(vec));
            out = obj.swm(Ainv, obj.basis, diag(1./obj.emu_vars), obj.basis', Aldet, sum(log(obj.emu_vars)));
        end

        function out = chol_solve(~, x)
            R = chol(x);
            ldet = 2 * sum(log(diag(R)));
            inv1 = inv(x);
            out.inv = inv1;
            out.ldet = ldet;
        end

        function out = swm(obj, Ainv, U, Cinv, V, Aldet, Cldet)
            in_mat = obj.chol_solve(Cinv + V*Ainv*U);
            inv1 = Ainv - Ainv * U * in_mat.inv * V * Ainv;
            ldet = in_mat.ldet + Aldet + Cldet;
            out.inv = inv1;
            out.ldet = ldet;
        end
    end
end

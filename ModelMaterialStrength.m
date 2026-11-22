classdef ModelMaterialStrength < ModelF
    % Material strength model (e.g. PTW) for Hopkinson-bar / quasistatic
    % stress-strain experiments, built on physics.MaterialModel. Each
    % experiment is a strain history at a given temperature and strain rate;
    % the model predicts the flow stress at the measured strains.

    properties
        meas_strain_histories  % cell array of measured strain vectors
        meas_strain_max  % maximum measured strain of each experiment
        strain_max
        nhists  % total number of measured strains
        model  % physics.MaterialModel
        model_info  % names of the submodels used
        constants  % struct of constants of the physics model
        options  % struct of options of the physics model
        temps  % temperature of each experiment (Kelvin)
        edots  % strain rate of each experiment
        nexp_strain  % number of experiments (strain histories)
        Nhist  % number of points at which the state history is computed
        pool
        y_cols  % cell array, columns of yobs belonging to each experiment
    end

    methods
        function obj = ModelMaterialStrength(temps, edots, consts, strain_histories, ...
                flow_stress_model, melt_model, shear_model, specific_heat_model, ...
                density_model, pool, s2, options)
            % **Material Strength Model**
            %
            % temps: temperature of each experiment (Kelvin)
            % edots: strain rate of each experiment
            % consts: struct of constants of the physics model; see the
            %         consts of the chosen submodels in physics.submodel
            % strain_histories: cell array of measured strain vectors, one
            %                   per experiment (a single numeric vector is
            %                   treated as one experiment)
            % flow_stress_model, melt_model, shear_model,
            % specific_heat_model, density_model: submodel names, see
            %                   physics.list_models()
            % pool: true for pooled calibration (default), false for
            %       hierarchical
            % s2: how to sample error variance, one of 'gibbs' (default),
            %     'gibbs_trunc', 'MH', or 'fix'
            % options: struct of optional physics constants overriding
            %          their defaults (default: none)
            %
            % The observations passed to CalibSetup.addVecExperiments must
            % be the measured stresses of all experiments concatenated in
            % the order of strain_histories.
            %
            % returns an object of class ModelMaterialStrength
            arguments
                temps
                edots
                consts struct
                strain_histories
                flow_stress_model
                melt_model
                shear_model
                specific_heat_model
                density_model
                pool = true;
                s2 = 'gibbs';
                options struct = struct();
            end
            if ~iscell(strain_histories)
                strain_histories = {strain_histories};
            end
            strain_histories = cellfun(@(v) reshape(v, 1, []), ...
                                       strain_histories, 'UniformOutput', false);
            mm = physics.MaterialModel(flow_stress_model=flow_stress_model, ...
                shear_modulus_model=shear_model, ...
                specific_heat_model=specific_heat_model, ...
                melt_model=melt_model, density_model=density_model);
            obj@ModelF(@(x) x, mm.get_parameter_list(), NaN, s2);
            obj.meas_strain_histories = strain_histories;
            obj.meas_strain_max = cellfun(@max, strain_histories);
            obj.strain_max = max(obj.meas_strain_max);
            obj.nhists = sum(cellfun(@numel, strain_histories));
            obj.model = mm;
            obj.model_info = {flow_stress_model, shear_model, ...
                specific_heat_model, melt_model, density_model};
            obj.constants = consts;
            obj.options = options;
            obj.nexp_strain = numel(strain_histories);
            obj.temps = reshape(temps, 1, []) + zeros(1, obj.nexp_strain);
            obj.edots = reshape(edots, 1, []) + zeros(1, obj.nexp_strain);
            obj.Nhist = 100;
            obj.pool = pool;
            ends = cumsum(cellfun(@numel, strain_histories));
            starts = [1, ends(1:end-1) + 1];
            obj.y_cols = arrayfun(@(a, b) a:b, starts, ends, 'UniformOutput', false);
        end

        function pred = eval(obj, parmat, pool, ~)
            % parmat: struct of parameter values (native scale), one field
            %         per calibration parameter. Parameters of the physics
            %         model that are not calibrated are taken from the
            %         constants.
            % pool: true (default) for pooled predictions, nrep x ny; false
            %       for hierarchical predictions, where parmat holds
            %       nrep*ntheta rows stacked by theta block (see
            %       unpool_pred) and the columns of each experiment are
            %       predicted from the block of its theta index
            arguments
                obj
                parmat
                pool = true;
                ~
            end
            fn = fieldnames(parmat);
            nt = numel(parmat.(fn{1}));
            E = obj.nexp_strain;
            if pool
                nrep = nt;
                blk = zeros(1, E);
            else
                theta_of_exp = cellfun(@(c) obj.exp_ind(c(1)), obj.y_cols);
                nrep = nt/max(obj.exp_ind);
                blk = (theta_of_exp - 1)*nrep;
            end
            % one physics run per (replicate, experiment) pair
            [r, e] = ndgrid(1:nrep, 1:E);
            r = r(:);
            e = e(:);
            prow = r + reshape(blk(e), [], 1);
            big = struct();
            for i = 1:numel(fn)
                v = parmat.(fn{i});
                big.(fn{i}) = reshape(v(prow), [], 1);
            end
            ntot = numel(r);
            strain_maxs = obj.meas_strain_max(e);
            obj.model.set_history_variables(strain_maxs, obj.edots(e), obj.Nhist);
            obj.model.initialize(big, obj.constants, obj.options);
            obj.model.initialize_state(reshape(obj.temps(e), [], 1), ...
                                       zeros(ntot, 1), zeros(ntot, 1));
            res = obj.model.compute_state_history();
            sim_strains = reshape(res(:, 2, :), obj.Nhist, ntot);
            sim_stresses = reshape(res(:, 3, :), obj.Nhist, ntot);

            pred = zeros(nrep, obj.nhists);
            for k = 1:ntot
                pred(r(k), obj.y_cols{e(k)}) = interp1(sim_strains(:, k), ...
                    sim_stresses(:, k), obj.meas_strain_histories{e(k)}, 'linear');
            end
        end
    end
end

classdef MaterialModel < handle
    % Material strength model built from property submodels (flow stress,
    % specific heat, shear modulus, melt temperature, density and heating).
    % MATLAB port of impala.physics.physical_models_vec.MaterialModel.
    %
    % Example:
    %   ptw = physics.MaterialModel(flow_stress_model="PTW_Yield_Stress", ...
    %                               shear_modulus_model="Stein_Shear_Modulus");
    %   ptw.set_history_variables(0.6, 2500e-6, 100);
    %   ptw.initialize(params, consts);
    %   ptw.initialize_state(1000);
    %   res = ptw.compute_state_history();  % Nhist x 6 x nrep
    %
    % All vectors (parameters, state) are columns with one element per
    % replicate (e.g. per parameter draw and/or experiment). Scalars are
    % broadcast.

    properties
        parameters  % struct of current parameter, constant and option values
        params  % cell array of calibration parameter names
        consts  % cell array of constant names
        options  % struct of options and their default values
        state  % struct with fields T, Tmelt, stress, strain, G, Cv, rho
        flow_stress  % submodel structs, see physics.submodel
        specific_heat
        shear_modulus
        melt_model
        density
        heating_model
        emax
        edot
        Nhist
        strain_history
    end

    methods
        function obj = MaterialModel(opts)
            % **Material Model**
            %
            % Name-value arguments (submodel names, see physics.list_models):
            %   flow_stress_model   (default: 'Constant_Yield_Stress')
            %   specific_heat_model (default: 'Constant_Specific_Heat')
            %   shear_modulus_model (default: 'Constant_Shear_Modulus')
            %   melt_model          (default: 'Constant_Melt_Temperature')
            %   density_model       (default: 'Constant_Density')
            %   heating_model       (default: 'Taylor_Quinney_Heating')
            %
            % returns an object of class physics.MaterialModel
            arguments
                opts.flow_stress_model {mustBeTextScalar} = 'Constant_Yield_Stress'
                opts.specific_heat_model {mustBeTextScalar} = 'Constant_Specific_Heat'
                opts.shear_modulus_model {mustBeTextScalar} = 'Constant_Shear_Modulus'
                opts.melt_model {mustBeTextScalar} = 'Constant_Melt_Temperature'
                opts.density_model {mustBeTextScalar} = 'Constant_Density'
                opts.heating_model {mustBeTextScalar} = 'Taylor_Quinney_Heating'
            end
            obj.flow_stress = load_submodel(opts.flow_stress_model, 'flow_stress');
            obj.specific_heat = load_submodel(opts.specific_heat_model, 'specific_heat');
            obj.shear_modulus = load_submodel(opts.shear_modulus_model, 'shear_modulus');
            obj.melt_model = load_submodel(opts.melt_model, 'melt');
            obj.density = load_submodel(opts.density_model, 'density');
            obj.heating_model = load_submodel(opts.heating_model, 'heating');

            subs = {obj.flow_stress, obj.specific_heat, obj.shear_modulus, ...
                    obj.melt_model, obj.density, obj.heating_model};
            p = {};
            c = {};
            o = struct();
            for i = 1:numel(subs)
                p = [p, subs{i}.params]; %#ok<AGROW>
                c = [c, subs{i}.consts]; %#ok<AGROW>
                fo = fieldnames(subs{i}.options);
                for k = 1:numel(fo)
                    o.(fo{k}) = subs{i}.options.(fo{k});
                end
            end
            c = unique(c, 'stable');
            if numel(unique(p)) ~= numel(p)
                error('physics:MaterialModel:duplicateParams', ...
                      'Some Duplicate Parameters between models');
            end
            if ~isempty(intersect(p, c))
                error('physics:MaterialModel:duplicateParamConst', ...
                      'Duplicate item in parameters and constants');
            end
            obj.params = p;
            obj.consts = c;
            obj.options = o;
            obj.parameters = struct();
            obj.state = struct('T', 300, 'Tmelt', [], 'stress', 0, ...
                               'strain', 0, 'G', [], 'Cv', [], 'rho', []);
        end

        function out = get_parameter_list(obj)
            % list of calibration parameters used in the model
            out = obj.params;
        end

        function out = get_constants_list(obj)
            % list of constants used in the model
            out = obj.consts;
        end

        function out = get_options_list(obj)
            % list of options used in the model
            out = fieldnames(obj.options)';
        end

        function update_state(obj, edot, dt)
            % updates the material state (specific heat, density, T, Tmelt,
            % strain, shear modulus, stress) for a step of length dt at
            % strain rate edot
            mp = obj.parameters;
            edot = edot(:);
            obj.state.Cv = col(obj.specific_heat.value(mp, obj.state, edot, dt));
            obj.state.rho = col(obj.density.value(mp, obj.state, edot, dt));
            obj.state.T = col(obj.heating_model.value(mp, obj.state, edot, dt));
            obj.state.strain = col(obj.state.strain + edot.*dt);
            obj.state.Tmelt = col(obj.melt_model.value(mp, obj.state, edot, dt));
            obj.state.G = col(obj.shear_modulus.value(mp, obj.state, edot, dt));
            obj.state.stress = col(obj.flow_stress.value(mp, obj.state, edot, dt));
        end

        function update_parameters(obj, x)
            % updates the current values of the calibration parameters
            %
            % x: struct with one field per parameter, or a numeric vector
            %    (or cell array) ordered as get_parameter_list()
            if isstruct(x)
                for i = 1:numel(obj.params)
                    obj.parameters.(obj.params{i}) = col(x.(obj.params{i}));
                end
            elseif isnumeric(x) || iscell(x)
                if numel(x) ~= numel(obj.params)
                    error('physics:MaterialModel:badParameters', ...
                          'Incorrect number of parameters!');
                end
                for i = 1:numel(obj.params)
                    if iscell(x)
                        obj.parameters.(obj.params{i}) = col(x{i});
                    else
                        obj.parameters.(obj.params{i}) = x(i);
                    end
                end
            else
                error('physics:MaterialModel:badParameters', ...
                      'Type %s is not supported.', class(x));
            end
        end

        function initialize(obj, parameters, constants, options)
            % Initialize the model at a given set of parameters, constants
            % and options
            %
            % parameters: struct of calibration parameter values. Any model
            %             parameter missing here is treated as a constant
            %             and taken from constants instead.
            % constants: struct of constant values (extra fields are ignored)
            % options: struct of options overriding the defaults (optional)
            arguments
                obj
                parameters struct
                constants struct
                options struct = struct()
            end
            user_constants = setdiff(obj.params, fieldnames(parameters), 'stable');
            free = setdiff(obj.params, user_constants, 'stable');
            need_consts = [obj.consts, user_constants];
            missing = setdiff(need_consts, fieldnames(constants), 'stable');
            if ~isempty(missing)
                error('physics:MaterialModel:missingConstants', ...
                      '%s missing from list of supplied parameters/constants', ...
                      strjoin(missing, ', '));
            end
            mp = struct();
            for i = 1:numel(free)
                mp.(free{i}) = col(parameters.(free{i}));
            end
            for i = 1:numel(need_consts)
                mp.(need_consts{i}) = col(constants.(need_consts{i}));
            end
            % default options, then user options
            fo = fieldnames(obj.options);
            for i = 1:numel(fo)
                mp.(fo{i}) = obj.options.(fo{i});
            end
            fo = fieldnames(options);
            for i = 1:numel(fo)
                mp.(fo{i}) = options.(fo{i});
            end
            obj.parameters = mp;
        end

        function initialize_state(obj, T, stress, strain)
            % sets the initial temperature, stress and strain
            arguments
                obj
                T = 300;
                stress = 0;
                strain = 0;
            end
            obj.state.T = col(T);
            obj.state.stress = col(stress);
            obj.state.strain = col(strain);
        end

        function set_history_variables(obj, emax, edot, nhist)
            % sets emax, edot and Nhist and stores the corresponding
            % constant strain-rate histories in strain_history
            obj.emax = emax;
            obj.edot = edot;
            obj.Nhist = nhist;
            obj.strain_history = physics.generate_strain_history(emax, edot, nhist);
        end

        function out = get_history_variables(obj)
            out = {obj.emax, obj.edot, obj.Nhist};
        end

        function results = compute_state_history(obj, strain_history)
            % Integrates the material state along the strain histories
            %
            % strain_history: struct from physics.generate_strain_history
            %                 (default: the one set by set_history_variables)
            %
            % returns an Nhist x 6 x nrep array whose second dimension is
            % [time, strain, stress, temperature, shear modulus, density]
            arguments
                obj
                strain_history = obj.strain_history
            end
            strains = strain_history.strains;
            times = strain_history.times;
            strain_rate = strain_history.strain_rate;
            [nrep, nh] = size(strains);

            results = zeros(nh, 6, nrep);
            obj.update_state(strain_rate(:, 1), 0);
            results(1, :, :) = obj.snapshot(times(:, 1), nrep);
            for i = 2:nh
                obj.update_state(strain_rate(:, i-1), times(:, i) - times(:, i-1));
                results(i, :, :) = obj.snapshot(times(:, i), nrep);
            end
        end
    end

    methods (Access = private)
        function out = snapshot(obj, t, nrep)
            s = obj.state;
            z = zeros(nrep, 1);
            out = reshape([t(:) + z, s.strain + z, s.stress + z, ...
                           s.T + z, s.G + z, s.rho + z]', 1, 6, nrep);
        end
    end
end


function m = load_submodel(name, kind)
m = physics.submodel(char(name));
if ~strcmp(m.kind, kind)
    error('physics:MaterialModel:wrongKind', ...
          '%s is a %s model, not a %s model.', name, m.kind, kind);
end
end


function v = col(v)
if ~isscalar(v)
    v = v(:);
end
end

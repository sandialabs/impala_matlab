function m = submodel(name)
% Specification of a material property submodel used by physics.MaterialModel
%
% name: name of the submodel, one of the names returned by
%       physics.list_models()
%
% returns a struct with fields
%   name    : the submodel name
%   kind    : 'flow_stress', 'specific_heat', 'shear_modulus', 'melt',
%             'density' or 'heating'
%   params  : cell array of calibration parameter names
%   consts  : cell array of constant names
%   options : struct of optional constants and their default values
%   value   : function handle @(mp, state, edot, dt) returning the property
%             at the current material state. mp is the struct of current
%             parameter, constant and option values; state is the material
%             state struct (see physics.MaterialModel)
%
% The submodels mirror the classes of impala.physics.physical_models_vec.

m = struct('name', name, 'kind', '', 'params', {{}}, 'consts', {{}}, ...
           'options', struct(), 'value', []);

switch name
    %% heating from plastic work
    case 'Taylor_Quinney_Heating'
        % heating from plastic work using the Taylor-Quinney coefficient chi
        % above a critical strain rate edotcrit_chi (default 0, i.e. always)
        m.kind = 'heating';
        m.params = {'chi'};
        m.options = struct('edotcrit_chi', 0);
        m.value = @(mp, s, edot, dt) s.T + (edot > mp.edotcrit_chi) ...
            .*mp.chi.*s.stress.*edot.*dt./(s.Cv.*s.rho);

    %% specific heat
    case 'Constant_Specific_Heat'
        m.kind = 'specific_heat';
        m.consts = {'Cv0'};
        m.value = @(mp, s, ~, ~) mp.Cv0.*ones(numel(s.T), 1);
    case 'Linear_Specific_Heat'
        m.kind = 'specific_heat';
        m.consts = {'c0', 'c1'};
        m.value = @(mp, s, ~, ~) physics.functions.Cubic_Specific_Heat(...
            mp.c0, mp.c1, 0, 0, s.T);
    case 'Quadratic_Specific_Heat'
        m.kind = 'specific_heat';
        m.consts = {'c0', 'c1', 'c2'};
        m.value = @(mp, s, ~, ~) physics.functions.Cubic_Specific_Heat(...
            mp.c0, mp.c1, mp.c2, 0, s.T);
    case 'Cubic_Specific_Heat'
        m.kind = 'specific_heat';
        m.consts = {'c0', 'c1', 'c2', 'c3'};
        m.value = @(mp, s, ~, ~) physics.functions.Cubic_Specific_Heat(...
            mp.c0, mp.c1, mp.c2, mp.c3, s.T);
    case 'Piecewise_Linear_Specific_Heat'
        m.kind = 'specific_heat';
        m.consts = {'T_t', 'c0_0', 'c1_0', 'c0_1', 'c1_1'};
        m.value = @(mp, s, ~, ~) physics.functions.Piecewise_Cubic_Specific_Heat(...
            mp.T_t, mp.c0_0, mp.c0_1, mp.c1_0, mp.c1_1, 0, 0, 0, 0, s.T);
    case 'Piecewise_Quadratic_Specific_Heat'
        m.kind = 'specific_heat';
        m.consts = {'T_t', 'c0_0', 'c1_0', 'c2_0', 'c0_1', 'c1_1', 'c2_1'};
        m.value = @(mp, s, ~, ~) physics.functions.Piecewise_Cubic_Specific_Heat(...
            mp.T_t, mp.c0_0, mp.c0_1, mp.c1_0, mp.c1_1, mp.c2_0, mp.c2_1, ...
            0, 0, s.T);
    case 'Piecewise_Cubic_Specific_Heat'
        m.kind = 'specific_heat';
        m.consts = {'T_t', 'c0_0', 'c1_0', 'c2_0', 'c3_0', ...
                    'c0_1', 'c1_1', 'c2_1', 'c3_1'};
        m.value = @(mp, s, ~, ~) physics.functions.Piecewise_Cubic_Specific_Heat(...
            mp.T_t, mp.c0_0, mp.c0_1, mp.c1_0, mp.c1_1, mp.c2_0, mp.c2_1, ...
            mp.c3_0, mp.c3_1, s.T);

    %% density
    case 'Constant_Density'
        m.kind = 'density';
        m.consts = {'rho0'};
        m.value = @(mp, s, ~, ~) mp.rho0.*ones(numel(s.T), 1);
    case 'Linear_Density'
        m.kind = 'density';
        m.consts = {'r0', 'r1'};
        m.value = @(mp, s, ~, ~) physics.functions.Cubic_Density(...
            mp.r0, mp.r1, 0, 0, s.T);
    case 'Quadratic_Density'
        m.kind = 'density';
        m.consts = {'r0', 'r1', 'r2'};
        m.value = @(mp, s, ~, ~) physics.functions.Cubic_Density(...
            mp.r0, mp.r1, mp.r2, 0, s.T);
    case 'Cubic_Density'
        m.kind = 'density';
        m.consts = {'r0', 'r1', 'r2', 'r3'};
        m.value = @(mp, s, ~, ~) physics.functions.Cubic_Density(...
            mp.r0, mp.r1, mp.r2, mp.r3, s.T);

    %% melt temperature
    case 'Constant_Melt_Temperature'
        m.kind = 'melt';
        m.consts = {'Tmelt0'};
        m.value = @(mp, s, ~, ~) mp.Tmelt0.*ones(numel(s.T), 1);
    case 'Linear_Melt_Temperature'
        m.kind = 'melt';
        m.consts = {'tm0', 'tm1'};
        m.value = @(mp, s, ~, ~) physics.functions.Cubic_Melt_Temperature(...
            mp.tm0, mp.tm1, 0, 0, s.rho);
    case 'Quadratic_Melt_Temperature'
        m.kind = 'melt';
        m.consts = {'tm0', 'tm1', 'tm2'};
        m.value = @(mp, s, ~, ~) physics.functions.Cubic_Melt_Temperature(...
            mp.tm0, mp.tm1, mp.tm2, 0, s.rho);
    case 'Cubic_Melt_Temperature'
        m.kind = 'melt';
        m.consts = {'tm0', 'tm1', 'tm2', 'tm3'};
        m.value = @(mp, s, ~, ~) physics.functions.Cubic_Melt_Temperature(...
            mp.tm0, mp.tm1, mp.tm2, mp.tm3, s.rho);
    case 'BGP_Melt_Temperature'
        m.kind = 'melt';
        m.consts = {'Tm_0', 'rho_m', 'gamma_1', 'gamma_3', 'q3'};
        m.value = @(mp, s, ~, ~) physics.functions.BGP_Melt_Temperature(...
            mp.Tm_0, mp.rho_m, mp.gamma_1, mp.gamma_3, mp.q3, s.rho);

    %% shear modulus
    case 'Constant_Shear_Modulus'
        m.kind = 'shear_modulus';
        m.consts = {'G0'};
        m.value = @(mp, s, ~, ~) mp.G0.*ones(numel(s.T), 1);
    case 'Linear_Cold_PW_Shear_Modulus'
        m.kind = 'shear_modulus';
        m.consts = {'g0', 'g1', 'alpha'};
        m.value = @(mp, s, ~, ~) physics.functions.Quadratic_Cold_PW_Shear_Modulus(...
            mp.g0, mp.g1, 0, mp.alpha, s.rho, s.T, s.Tmelt);
    case 'Quadratic_Cold_PW_Shear_Modulus'
        m.kind = 'shear_modulus';
        m.consts = {'g0', 'g1', 'g2', 'alpha'};
        m.value = @(mp, s, ~, ~) physics.functions.Quadratic_Cold_PW_Shear_Modulus(...
            mp.g0, mp.g1, mp.g2, mp.alpha, s.rho, s.T, s.Tmelt);
    case 'Simple_Shear_Modulus'
        m.kind = 'shear_modulus';
        m.consts = {'G0', 'alpha'};
        m.value = @(mp, s, ~, ~) physics.functions.Simple_Shear_Modulus(...
            mp.G0, mp.alpha, s.T, s.Tmelt);
    case 'BGP_PW_Shear_Modulus'
        m.kind = 'shear_modulus';
        m.consts = {'G0', 'rho_0', 'gamma_1', 'gamma_2', 'q2', 'alpha'};
        m.options = struct('melt_strength', 1e-5);
        m.value = @(mp, s, ~, ~) physics.functions.BGP_PW_Shear_Modulus(...
            mp.G0, mp.rho_0, mp.gamma_1, mp.gamma_2, mp.q2, mp.alpha, ...
            s.rho, s.T, s.Tmelt, mp.melt_strength);
    case 'Stein_Shear_Modulus'
        % Steinberg-Guinan shear modulus at constant density and pressure
        m.kind = 'shear_modulus';
        m.consts = {'G0', 'sgB'};
        m.value = @(mp, s, ~, ~) physics.functions.Stein_Shear_Modulus(...
            mp.G0, mp.sgB, s.T, s.Tmelt);

    %% flow stress
    case 'Constant_Yield_Stress'
        m.kind = 'flow_stress';
        m.consts = {'yield_stress'};
        m.value = @(mp, s, ~, ~) mp.yield_stress.*ones(numel(s.T), 1);
    case 'JC_Yield_Stress'
        % Johnson-Cook
        m.kind = 'flow_stress';
        m.params = {'A', 'B', 'C', 'n', 'm'};
        m.consts = {'Tref', 'edot0'};
        m.value = @(mp, s, edot, ~) physics.functions.JC_Yield_Stress(edot, ...
            mp.A, mp.B, mp.C, mp.n, mp.m, mp.Tref, mp.edot0, ...
            s.strain, s.T, s.Tmelt);
    case 'PTW_Yield_Stress'
        % Preston-Tonks-Wallace
        m.kind = 'flow_stress';
        m.params = {'theta', 'p', 's0', 'beta', 'sInf', 'kappa', 'lgamma', ...
                    'y0', 'yInf', 'y1', 'y2'};
        m.consts = {'rho0', 'matomic'};
        m.value = @ptw_value;
    case 'Stein_Flow_Stress'
        % Steinberg-Guinan
        m.kind = 'flow_stress';
        m.params = {'y0', 'beta', 'n', 'ymax'};
        m.consts = {'G0', 'epsi'};
        m.value = @(mp, s, ~, ~) physics.functions.Stein_Flow_Stress(...
            mp.y0, mp.beta, mp.n, mp.ymax, mp.G0, mp.epsi, ...
            s.G, s.strain, s.T, s.Tmelt);

    otherwise
        error('physics:submodel:unknown', ...
              'Unknown physics submodel: %s. See physics.list_models().', name);
end
end


function out = ptw_value(mp, s, edot, ~)
good = physics.functions.PTW_goodparam(mp.s0, mp.sInf, mp.y0, mp.yInf, ...
                                       mp.y1, mp.y2, mp.beta);
if ~all(good)
    error('physics:ConstraintError', 'PTW bad val');
end
out = physics.functions.PTW_Yield_Stress(mp.p, mp.kappa, mp.s0, mp.sInf, ...
    mp.y0, mp.yInf, mp.y1, mp.y2, mp.beta, mp.theta, mp.lgamma, edot, ...
    mp.rho0, mp.matomic, s.G, s.strain, s.T, s.Tmelt, 1e-10);
end

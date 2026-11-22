function good = constraints_ptw(x, bounds, constants)
% Constraint function for calibrating the PTW flow stress model: checks the
% bounds and that the PTW parameter set is valid (physics.functions.PTW_goodparam).
%
% x: struct of parameter values, one field per calibration parameter
% bounds: struct of [lower upper] bounds, one field per calibration parameter
% constants: struct of PTW parameters kept constant during calibration
%            (default: none). CalibSetup passes the constants of the first
%            model added with addVecExperiments.
%
% Use it as CalibSetup(bounds, @constraints_ptw).
arguments
    x
    bounds
    constants = struct();
end
y = constants;
if isempty(y)
    y = struct();
end
fx = fieldnames(x);
for i = 1:numel(fx)
    y.(fx{i}) = x.(fx{i});
end
good = physics.functions.PTW_goodparam(y.s0, y.sInf, y.y0, y.yInf, ...
                                       y.y1, y.y2, y.beta);
k = fieldnames(bounds);
for i = 1:numel(k)
    good = good & (x.(k{i}) < bounds.(k{i})(2)) & (x.(k{i}) > bounds.(k{i})(1));
end
end

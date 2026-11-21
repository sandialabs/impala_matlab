function [f, x, bounds] = calibToyProblem()
%CALIBTOYPROBLEM Small two-parameter model shared by the calibration tests.
x = linspace(0, 1, 30);
f = @(X) X(:,1).*sin(2*pi*x) + X(:,2).*x;
bounds = struct('a', [0 1], 'b', [0 3]);
end

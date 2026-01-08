function out = probit(x)
% Probit Transformation: For x in (0,1), y in (-inf,inf)
out = sqrt(2) * erfinv(2*x-1);
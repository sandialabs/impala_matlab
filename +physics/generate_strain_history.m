function h = generate_strain_history(emax, edot, nhist)
% Generates constant strain-rate histories from zero to emax
%
% emax: maximum strain, scalar or one per history
% edot: strain rate, scalar or one per history
% nhist: number of points in each history
%
% returns a struct with fields (nexp = number of histories)
%   times       : nexp x nhist
%   strains     : nexp x nhist
%   strain_rate : nexp x (nhist-1)
emax = emax(:);
edot = edot(:);
n = max(numel(emax), numel(edot));
emax = emax + zeros(n, 1);
edot = edot + zeros(n, 1);
tmax = emax./edot;
u = linspace(0, 1, nhist);
strains = emax.*u;
times = tmax.*u;
% make the end points exact, as linspace does
strains(:, end) = emax;
times(:, end) = tmax;
h.times = times;
h.strains = strains;
h.strain_rate = diff(strains, 1, 2)./diff(times, 1, 2);
end

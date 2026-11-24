function q = weighted_quantile(x, p, w)
% Weighted quantiles of a 1-d sample.
%
% x: vector of values
% p: scalar or array of probabilities in [0,1]
% w: vector of non-negative weights, the same length as x
%
% returns an array the size of p
%
% Uses the same definition as an unweighted empirical quantile: the
% weights are accumulated to the midpoint of each value's own weight and
% the result is interpolated linearly. Equal weights give the ordinary
% empirical quantile (the median matches median() exactly; other
% quantiles use the midpoint convention rather than the one in plims).
x = double(reshape(x, [], 1));
w = double(reshape(w, [], 1));
if numel(x) ~= numel(w)
    error('weighted_quantile:sizeMismatch', 'x and w must have the same number of elements');
end
if any(w < 0) || ~(sum(w) > 0)
    error('weighted_quantile:badWeights', 'w must be non-negative with a positive sum');
end
[x, ord] = sort(x);
w = w(ord);
cw = (cumsum(w) - 0.5*w) / sum(w);
% drop ties (zero-weight values share a cumulative position) so interp1
% gets strictly increasing sample points
[cwu, iu] = unique(cw, 'last');
if numel(cwu) < 2
    q = repmat(x(iu(end)), size(p));
    return
end
% clamp instead of extrapolating, like numpy's interp
q = interp1(cwu, x(iu), min(max(p, cwu(1)), cwu(end)), 'linear');
end

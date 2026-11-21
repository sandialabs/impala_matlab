function ld = mvn_logpdf_batch(X, mu, prec, ldet)
% Multivariate normal log-density, one mean/covariance per temperature.
%
% X:    ntemps x n x p array of points
% mu:   ntemps x p means
% prec: ntemps x p x p precision matrices
% ldet: ntemps x 1 log-determinants of the covariance matrices
%
% returns an ntemps x n matrix of log-densities
[ntemps, n, p] = size(X);
if ndims(X) == 2 %#ok<ISMAT>
    p = size(mu, 2);
    n = numel(X) / (ntemps * p);
end
X = reshape(X, ntemps, n, p);
ld = zeros(ntemps, n);
for t = 1:ntemps
    d = reshape(X(t,:,:), n, p) - mu(t,:);
    P = reshape(prec(t,:,:), p, p);
    ld(t,:) = -0.5*p*log(2*pi) - 0.5*ldet(t) - 0.5*sum((d*P).*d, 2)';
end
end

function lp = log_box_prob(mu, cov)
% log P(theta in [0,1]^p) for theta ~ N(mu, cov), i.e. the log of the
% normalising constant of the hierarchical distribution truncated to the
% 0-1 scaled bounds.
%
% Uses the closed form (a product of normal CDFs) when cov is diagonal and
% MATLAB's mvncdf (Genz's algorithm) otherwise.
%
% mu: 1 x p mean on the 0-1 scale
% cov: p x p covariance
%
% returns a scalar
p = numel(mu);
mu = reshape(mu, 1, p);
if p == 1
    sd = sqrt(cov(1,1));
    val = normcdf((1 - mu(1))/sd) - normcdf(-mu(1)/sd);
    lp = log(max(val, 1e-300));
    return
end
off = cov - diag(diag(cov));
if ~any(abs(off(:)) > 1e-12 * max(max(abs(cov(:))), 1))
    sd = sqrt(diag(cov))';
    val = normcdf((1 - mu)./sd) - normcdf(-mu./sd);
    lp = sum(log(max(val, 1e-300)));
    return
end
val = mvncdf(zeros(1, p), ones(1, p), mu, (cov + cov')/2);
lp = log(max(val, 1e-300));
end

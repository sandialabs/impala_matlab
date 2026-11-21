function ld = invwishart_logpdf(W, df, scale)
% Log-density of the inverse Wishart distribution with degrees of freedom
% df and scale matrix scale, evaluated at each W(t,:,:).
%
% W: ntemps x p x p array (or a single p x p matrix)
% returns an ntemps x 1 vector
p = size(scale, 1);
if ismatrix(W) && size(W,1) == p && size(W,2) == p
    W = reshape(W, 1, p, p);
end
nt = size(W, 1);
ld = zeros(nt, 1);
lscale = logdet_chol(scale);
lmg = p*(p-1)/4*log(pi) + sum(gammaln(df/2 + (1 - (1:p))/2));
for t = 1:nt
    Wt = reshape(W(t,:,:), p, p);
    ld(t) = 0.5*df*lscale - lmg - 0.5*df*p*log(2) ...
        - 0.5*(df + p + 1)*logdet_chol(Wt) - 0.5*trace(scale / Wt);
end
end

function ld = logdet_chol(A)
R = chol((A + A')/2);
ld = 2*sum(log(diag(R)));
end

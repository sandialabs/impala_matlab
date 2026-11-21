function cand = sample_constrained(means, covs, setup, maxiter)
% Draw one multivariate normal sample per row of means (on the 0-1 scale)
% and redraw the rows that violate setup.checkConstraints.
%
% means: n x p matrix of means
% covs:  p x p covariance shared by all rows, or n x p x p array
% setup: CalibSetup object (provides constraints and bounds)
% maxiter: maximum number of redraws (default 10000)
%
% returns an n x p matrix
arguments
    means
    covs
    setup
    maxiter = 10000;
end
[n, p] = size(means);
if ndims(covs) == 2 && isequal(size(covs), [p p]) %#ok<ISMAT>
    L = chol((covs + covs')/2, 'lower');
    draw = @(idx) means(idx,:) + randn(numel(idx), p) * L';
else
    Ls = zeros(n, p, p);
    for k = 1:n
        C = reshape(covs(k,:,:), p, p);
        Ls(k,:,:) = chol((C + C')/2, 'lower');
    end
    draw = @(idx) draw_each(means, Ls, idx, p);
end
names = fieldnames(setup.bounds);
cand = draw(1:n);
good = reshape(setup.checkConstraints(tran_unif(cand, setup.bounds_mat, names), setup.bounds), n, 1);
it = 0;
while any(~good)
    it = it + 1;
    if it > maxiter
        error('sample_constrained:maxiter', ...
              'Failed to find samples that fulfill the constraints after %d iterations.', maxiter);
    end
    bad = find(~good);
    cand(bad,:) = draw(bad);
    good(bad) = reshape(setup.checkConstraints(tran_unif(cand(bad,:), setup.bounds_mat, names), setup.bounds), [], 1);
end
end

function out = draw_each(means, Ls, idx, p)
out = zeros(numel(idx), p);
for k = 1:numel(idx)
    out(k,:) = means(idx(k),:) + (reshape(Ls(idx(k),:,:), p, p) * randn(p,1))';
end
end

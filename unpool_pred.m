function out = unpool_pred(pred, exp_ind)
% Convert predictions made for stacked, theta-specific inputs into one
% prediction per replicate (used by hierarchical/clustered calibration).
%
% pred: (nrep*nexp) x ny matrix of predictions. Rows are stacked by
%       experiment block: rows (j-1)*nrep + (1:nrep) were evaluated at the
%       inputs for theta index j (this is the order produced by reshaping an
%       nrep x nexp x p array to (nrep*nexp) x p).
% exp_ind: 1 x ny vector giving the theta index (1..nexp) of each output.
%
% returns an nrep x ny matrix where column k is taken from the row block of
% theta index exp_ind(k).
ny = size(pred, 2);
if isscalar(exp_ind)
    exp_ind = repmat(exp_ind, 1, ny);
end
nexp = max(exp_ind);
nrep = size(pred, 1) / nexp;
if nrep ~= round(nrep)
    error('unpool_pred:badSize', ...
          'Number of rows (%d) is not a multiple of the number of thetas (%d).', ...
          size(pred,1), nexp);
end
out = zeros(nrep, ny);
for j = 1:nexp
    cols = exp_ind == j;
    out(:, cols) = pred((j-1)*nrep + (1:nrep), cols);
end
end

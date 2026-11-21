function out = tran_probit(th, bounds, names)
% Transform th from the probit scale to the native scale: the inverse
% probit gives the 0-1 scale, which is then unnormalized by the bounds.
out = tran_unif(invprobit(th), bounds, names);
end

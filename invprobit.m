function out = invprobit(y)
out = 0.5 * (1 + erf(y/sqrt(2)));
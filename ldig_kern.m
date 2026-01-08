function out = ldig_kern(x,a,b)
out = (-a-1) .* log(x) - b./x;
function ld = gamma_logpdf(s, shape, rate)
% log-density of a gamma(shape, rate) distribution at s
ld = shape.*log(rate) - gammaln(shape) + (shape - 1).*log(s) - rate.*s;
end

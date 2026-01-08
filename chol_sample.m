function out = chol_sample(mean, cov)
out = mean + chol(cov, 'lower')*randn(size(mean));
function eta = sample_eta(curr_eta, nclust, ndat, prior_shape, prior_rate)
% Sample the Dirichlet process concentration parameter eta (Escobar & West
% 1995) for each temperature.
%
% curr_eta: ntemps x 1 current values
% nclust:   ntemps x 1 number of occupied clusters
% ndat:     total number of thetas being clustered
curr_eta = curr_eta(:);
nclust = nclust(:);
g = betarnd(curr_eta + 1, ndat);
aa = prior_shape + nclust;
bb = prior_rate - log(g);
% Escobar & West (1995), eq. 13: mixture of Gamma(a + k, b - log g) with
% probability w and Gamma(a + k - 1, b - log g) with probability 1 - w,
% where w / (1 - w) = (a + k - 1) / (n (b - log g)).
w = (aa - 1) ./ (ndat .* bb + aa - 1);
shape = aa - (rand(size(aa)) >= w);  % a + k with probability w
eta = gamrnd(shape, 1 ./ bb);
end

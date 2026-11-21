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
w = (aa - 1) ./ (ndat .* bb + aa - 1);
shape = aa - (rand(size(aa)) >= w);
eta = gamrnd(shape, 1 ./ bb);
end

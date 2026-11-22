function kern = s2_swap_kern(setup, i)
% Log prior kernel of s2 for experiment i to use in tempering swap ratios.
%
% It must be the prior that the within-chain s2 update actually targets.
% The conjugate Gibbs updates ('gibbs', 'gibbs_trunc') draw from the
% inverse-gamma full conditional with shape ig_a and scale ig_b (both 0
% when s2_df == 0, i.e. p(s2) proportional to 1/s2), so the swap must use
% the inverse-gamma kernel ldig_kern with the same parameters. The
% Metropolis-Hastings update uses setup.s2_prior_kern{i} directly (the
% half-Cauchy kernel when any s2_df == 0), so the swap uses that too.
if any(strcmp(setup.models{i}.s2, {'gibbs', 'gibbs_trunc'}))
    kern = @ldig_kern;
else
    kern = setup.s2_prior_kern{i};
end
end

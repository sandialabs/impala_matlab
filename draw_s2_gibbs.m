function ls2 = draw_s2_gibbs(dev_sq, setup, i, itl, trunc)
% Gibbs draw of log error variances for one experiment at one temperature.
%
% dev_sq: 1 x ns2 sums of squared residuals for each error group
% setup:  CalibSetup object
% i:      experiment index
% itl:    inverse temperature
% trunc:  true to truncate to [sd_lower^2, sd_upper^2] (gibbs_trunc)
%
% returns a 1 x ns2 vector of log variances
shape = itl .* (setup.ny_s2{i}(:)'/2 + setup.ig_a{i} + 1) - 1;
rate = itl .* (setup.ig_b{i} + dev_sq/2);
ls2 = log(1 ./ gamrnd(shape, 1 ./ rate));
if trunc
    lo = log(setup.sd_lower{i}.^2);
    hi = log(setup.sd_upper{i}.^2);
    bad = ls2 < lo | ls2 > hi;
    ct = 0;
    while any(bad)
        ls2(bad) = log(1 ./ gamrnd(shape(bad), 1 ./ rate(bad)));
        bad = ls2 < lo | ls2 > hi;
        ct = ct + 1;
        if ct >= 50
            ls2 = min(max(ls2, lo), hi);
            break
        end
    end
end
end

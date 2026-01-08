function out = cor2cov(R, s)
    out = (s * s') * R;
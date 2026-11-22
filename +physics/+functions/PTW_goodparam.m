function good = PTW_goodparam(s0, sInf, y0, yInf, y1, y2, beta)
% Checks whether the given PTW parameter set is valid
good = (sInf < s0) & (yInf < y0) & (y0 < s0) & (yInf < sInf) ...
    & (y1 > s0) & (y2 > beta);
end

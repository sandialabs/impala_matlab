function u = clip_unit(u, eps)
% Clip values on the 0-1 scale away from 0 and 1 (for probit).
arguments
    u
    eps = 1e-12;
end
u = min(max(u, eps), 1 - eps);
end

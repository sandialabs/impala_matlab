function Y = JC_Yield_Stress(edot, A, B, C, n, m, Tref, edot0, eps, T, Tmelt)
% Johnson-Cook yield stress model; see Johnson, Cook, Engineering Fracture
% Mechanics, 21(1):31-48, 1985, https://doi.org/10.1016/0013-7944(85)90052-9
%
% edot: current strain rate
% eps: current strain
% T: temperature
% Tmelt: melting temperature
% edot0, Tref: reference strain rate and temperature
% A, B, C, n, m: model parameters
th = max(0, (T - Tref)./(Tmelt - Tref));
Y = (A + B.*eps.^n) .* (1 + C.*log(edot./edot0)) .* (1 - th.^m);
end

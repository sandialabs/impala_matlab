function Cv = Piecewise_Cubic_Specific_Heat(Tt, c00, c01, c10, c11, c20, c21, c30, c31, T)
% Piecewise cubic specific heat model
%
%   Cv(T) = c00 + c10*T + c20*T^2 + c30*T^3   for T <= Tt
%   Cv(T) = c01 + c11*T + c21*T^2 + c31*T^3   for T >  Tt
%
% T: current temperature
% Tt: temperature separating the two cubic models
% c00, c10, c20, c30: model parameters used up to temperature Tt
% c01, c11, c21, c31: model parameters used above Tt
hi = T > Tt;
pow0 = c00 + (c01 - c00).*hi;
pow1 = c10 + (c11 - c10).*hi;
pow2 = c20 + (c21 - c20).*hi;
pow3 = c30 + (c31 - c30).*hi;
Cv = pow0 + pow1.*T + pow2.*T.^2 + pow3.*T.^3;
end

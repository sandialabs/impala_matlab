function Cv = Cubic_Specific_Heat(c0, c1, c2, c3, T)
% Cubic specific heat model: Cv(T) = c0 + c1*T + c2*T^2 + c3*T^3
%
% T: temperature
% c0, c1, c2, c3: model parameters
Cv = c0 + c1.*T + c2.*T.^2 + c3.*T.^3;
end

function Tm = Cubic_Melt_Temperature(tm0, tm1, tm2, tm3, rho)
% Cubic melt temperature model: Tm(rho) = tm0 + tm1*rho + tm2*rho^2 + tm3*rho^3
%
% rho: material density
% tm0, tm1, tm2, tm3: model parameters
Tm = tm0 + tm1.*rho + tm2.*rho.^2 + tm3.*rho.^3;
end

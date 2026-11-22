function Tm = BGP_Melt_Temperature(Tm0, rhom, gamma1, gamma3, q3, rho)
% Burakovsky-Greeff-Preston melt temperature model,
% see doi.org/10.1103/PhysRevB.67.094107
%
% rho: current material density
% rhom: reference density
% Tm0: melt temperature at reference density rhom
% gamma1, gamma3, q3: model parameters
Tm = Tm0 .* nthroot(rho./rhom, 3) .* exp( ...
    6.*gamma1.*(1./nthroot(rhom, 3) - 1./nthroot(rho, 3)) ...
    + 2.*gamma3./q3.*(rhom.^(-q3) - rho.^(-q3)));
end

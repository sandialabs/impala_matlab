function G = BGP_PW_Shear_Modulus(G0, rho_0, gamma_1, gamma_2, q2, alpha, rho, T, Tmelt, melt_strength)
% Burakovsky-Greeff-Preston cold shear combined with a (linear) PW
% temperature dependence, see Burakovsky, Greeff, Preston, Phys. Rev. B67
% (2003) 094107, DOI:10.1103/PhysRevB.67.094107
%
% rho: current material density
% T: temperature
% Tmelt: melting temperature
% rho_0: reference density
% G0: shear modulus at reference density rho_0
% gamma_1, gamma_2, q2, alpha: model parameters
% melt_strength: shear modulus at and above melt; [] leaves it as computed
%                (default: 0)
arguments
    G0
    rho_0
    gamma_1
    gamma_2
    q2
    alpha
    rho
    T
    Tmelt
    melt_strength = 0
end
cold_shear = G0 .* (rho./rho_0).^(4/3) .* exp( ...
    6.*gamma_1.*(1./nthroot(rho_0, 3) - 1./nthroot(rho, 3)) ...
    + 2.*gamma_2./q2.*(rho_0.^(-q2) - rho.^(-q2)));
G = cold_shear .* (1 - alpha.*min(T./Tmelt, 1));
if ~isempty(melt_strength)
    melted = T >= Tmelt;
    G(melted) = melt_strength;
end
G(G < 0) = 0;
end

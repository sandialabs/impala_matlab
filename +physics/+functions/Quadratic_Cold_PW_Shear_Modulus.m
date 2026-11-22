function G = Quadratic_Cold_PW_Shear_Modulus(g0, g1, g2, alpha, rho, T, Tmelt)
% Quadratic cold PW shear modulus
%
% rho: current material density
% T: temperature
% Tmelt: melting temperature
% g0, g1, g2, alpha: model parameters
cold_shear = g0 + g1.*rho + g2.*rho.^2;
G = cold_shear .* (1 - alpha.*(T./Tmelt));
G(T >= Tmelt | G < 0) = 0;
end

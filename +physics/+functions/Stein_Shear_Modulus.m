function G = Stein_Shear_Modulus(G0, sgB, T, Tmelt)
% Steinberg-Guinan shear modulus assuming constant density and pressure,
% so only the temperature dependence is included; see Steinberg, Cochran,
% and Guinan, J. Appl. Phys. 51 (1980) 1498, https://doi.org/10.1063/1.327799
%
% T: temperature
% Tmelt: melting temperature
% G0, sgB: model parameters
aterm = 0;
bterm = sgB.*(T - 300);
G = G0 .* (1 + aterm - bterm);
G(T >= Tmelt | G < 0) = 0;
end

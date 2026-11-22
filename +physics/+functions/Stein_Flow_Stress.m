function Y = Stein_Flow_Stress(y0, beta, n, ymax, G0, epsi, shear, eps, T, Tmelt)
% Steinberg-Guinan flow stress model (without the thermally activated part
% of the yield strength); see Steinberg, Cochran, and Guinan, J. Appl.
% Phys. 51 (1980) 1498, https://doi.org/10.1063/1.327799 and Steinberg,
% Lund, J. Appl. Phys. 65 (1989) 1528, https://doi.org/10.1063/1.342968
%
% eps: current strain
% T: temperature
% Tmelt: melting temperature
% shear: current shear modulus
% beta, n: empirical work hardening model parameters
% epsi: initial equivalent plastic strain
% y0: initial flow stress
% ymax: upper limit of the hardening term
% G0: reference shear modulus at ambient conditions
fnow = (1 + beta.*(epsi + eps)).^n;
ycap = (ymax./y0) + 0.*fnow;
cap = fnow.*y0 > ymax;
fnow(cap) = ycap(cap);
fnow(T > Tmelt) = 0;
Y = y0 .* fnow .* shear ./ G0;
end

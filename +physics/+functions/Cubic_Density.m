function rho = Cubic_Density(r0, r1, r2, r3, T)
% Cubic density model: rho(T) = r0 + r1*T + r2*T^2 + r3*T^3
%
% T: temperature
% r0, r1, r2, r3: model parameters
rho = r0 + r1.*T + r2.*T.^2 + r3.*T.^3;
end

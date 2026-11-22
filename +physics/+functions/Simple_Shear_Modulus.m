function G = Simple_Shear_Modulus(G0, alpha, T, Tmelt)
% Simple shear modulus: G = G0*(1 - alpha*T/Tmelt)
%
% T: temperature
% Tmelt: melting temperature
% G0, alpha: model parameters
G = G0 .* (1 - alpha.*(T./Tmelt));
end

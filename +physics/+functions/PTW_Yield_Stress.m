function stress = PTW_Yield_Stress(p, kappa, s0, sInf, y0, yInf, y1, y2, beta, theta, lgamma, edot, rho0, matomic, shear, eps, T, Tmelt, small)
% Preston-Tonks-Wallace flow stress model. Returns the flow stress at the
% current material state and the given strain rate; see Preston, Tonks,
% Wallace, J. Appl. Phys. 93 (2003) 211, doi.org/10.1063/1.1524706
%
% edot: current strain rate
% eps: current strain
% T: temperature
% Tmelt: melting temperature
% shear: shear modulus (its temperature dependence is assumed to be
%        handled by the shear modulus model)
% p, kappa, s0, sInf, y0, yInf, y1, y2, beta, theta, lgamma: model parameters
% rho0, matomic: density and atomic mass
% small: tolerance (default: 1e-10)
arguments
    p
    kappa
    s0
    sInf
    y0
    yInf
    y1
    y2
    beta
    theta
    lgamma
    edot
    rho0
    matomic
    shear
    eps
    T
    Tmelt
    small = 1e-10
end
% expand everything to a common column length so that element-wise
% subsetting below works whether an input is a scalar constant or a vector
n = max(cellfun(@numel, {p, kappa, s0, sInf, y0, yInf, y1, y2, beta, ...
    theta, lgamma, edot, rho0, matomic, shear, eps, T, Tmelt}));
ex = @(v) v(:) + zeros(n, 1);
p = ex(p); kappa = ex(kappa); s0 = ex(s0); sInf = ex(sInf); y0 = ex(y0);
yInf = ex(yInf); y1 = ex(y1); y2 = ex(y2); beta = ex(beta); theta = ex(theta);
lgamma = ex(lgamma); edot = ex(edot); rho0 = ex(rho0); matomic = ex(matomic);
shear = ex(shear); eps = ex(eps); T = ex(T); Tmelt = ex(Tmelt);

Avogadro = 6.022e23;
t_hom = T./Tmelt;

afact = (4/3)*pi*rho0./matomic;
% ainv is 1/a where 4/3 pi a^3 is the atomic volume
ainv = nthroot(afact, 3);
% transverse wave velocity up to units
xfact = sqrt(shear./rho0);
% PTW characteristic strain rate; only its ratio to edot matters
xiDot = 0.5*ainv.*xfact*nthroot(Avogadro, 3);

log_xid_ed = log(xiDot./edot);
argErf = kappa.*t_hom.*(lgamma + log_xid_ed);
Erfres = erf(argErf);

saturation1 = s0 - (s0 - sInf).*Erfres;
saturation2 = s0.*exp(beta.*(-lgamma - log_xid_ed));
tau_s = max(saturation1, saturation2);

ayield = y0 - (y0 - yInf).*Erfres;
byield = y1.*exp(-y2.*(lgamma + log_xid_ed));
cyield = s0.*exp(-beta.*(lgamma + log_xid_ed));
tau_y = max(ayield, min(byield, cyield));

scaled_stress = tau_s;
ind = (p > small) & (abs(tau_s - tau_y) > small);
eArg1 = p(ind).*(tau_s(ind) - tau_y(ind))./(s0(ind) - tau_y(ind));
eArg2 = eps(ind).*p(ind).*theta(ind)./(s0(ind) - tau_y(ind))./(exp(eArg1) - 1);
theLog = log(1 - (1 - exp(-eArg1)).*exp(-eArg2));
scaled_stress(ind) = tau_s(ind) + (s0(ind) - tau_y(ind)).*theLog./p(ind);

ind2 = (p <= small) & (tau_s > tau_y);
scaled_stress(ind2) = tau_s(ind2) - (tau_s(ind2) - tau_y(ind2)) ...
    .*exp(-eps(ind2).*theta(ind2)./(tau_s(ind2) - tau_y(ind2)));

stress = scaled_stress.*shear*2;
end

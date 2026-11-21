classdef AMcov_hier < handle
    % Stores and updates the proposal covariances for Adaptive Metropolis in
    % a hierarchical calibration: one covariance per temperature and per
    % experiment-specific theta.
    %
    % Chains are stored per experiment as nmcmc x ntemps x ntheta(i) x p.

    properties
        eps
        AM_SCALAR
        tau        % cell, ntemps x ntheta(i)
        S          % cell, ntemps x ntheta(i) x p x p
        cov        % cell, ntemps x ntheta(i) x p x p
        mu         % cell, ntemps x ntheta(i) x p
        nexp
        ntheta
        ntemps
        p
        start_adapt_iter
        count_100  % cell, ntemps x ntheta(i)
    end

    methods
        function obj = AMcov_hier(nexp, ntheta, ntemps, p, start_var, start_adapt_iter, tau_start)
            arguments
                nexp
                ntheta
                ntemps
                p
                start_var = 1e-4;
                start_adapt_iter = 300;
                tau_start = 0;
            end
            obj.eps = 1.0e-12;
            obj.AM_SCALAR = 2.4^2 / p;
            obj.nexp = nexp;
            obj.ntheta = ntheta;
            obj.ntemps = ntemps;
            obj.p = p;
            obj.start_adapt_iter = start_adapt_iter;
            obj.tau = cell(1, nexp);
            obj.S = cell(1, nexp);
            obj.cov = cell(1, nexp);
            obj.mu = cell(1, nexp);
            obj.count_100 = cell(1, nexp);
            for i = 1:nexp
                nth = ntheta(i);
                obj.tau{i} = tau_start * ones(ntemps, nth);
                obj.S{i} = repmat(reshape(eye(p) * start_var, 1, 1, p, p), ntemps, nth);
                obj.cov{i} = zeros(ntemps, nth, p, p);
                obj.mu{i} = zeros(ntemps, nth, p);
                obj.count_100{i} = zeros(ntemps, nth);
            end
        end

        function obj = update(obj, x, m)
            % x: cell of chains; called in iteration m (latest is m-1)
            if m < obj.start_adapt_iter
                return
            end
            for i = 1:obj.nexp
                nth = obj.ntheta(i);
                if m > obj.start_adapt_iter
                    xl = reshape(x{i}(m-1,:,:,:), obj.ntemps, nth, obj.p);
                    obj.mu{i} = obj.mu{i} + (xl - obj.mu{i}) ./ m;
                    d = xl - obj.mu{i};
                    for t = 1:obj.ntemps
                        for j = 1:nth
                            dj = reshape(d(t,j,:), 1, obj.p);
                            obj.cov{i}(t,j,:,:) = ((m-1)/m) * reshape(obj.cov{i}(t,j,:,:), obj.p, obj.p) + ...
                                ((m-1)/(m*m)) * (dj' * dj);
                        end
                    end
                else
                    xs = reshape(x{i}(1:m,:,:,:), m, obj.ntemps, nth, obj.p);
                    obj.mu{i} = reshape(mean(xs, 1), obj.ntemps, nth, obj.p);
                    for t = 1:obj.ntemps
                        for j = 1:nth
                            obj.cov{i}(t,j,:,:) = cov(reshape(xs(:,t,j,:), m, obj.p));
                        end
                    end
                end
                for t = 1:obj.ntemps
                    for j = 1:nth
                        obj.S{i}(t,j,:,:) = obj.AM_SCALAR * exp(obj.tau{i}(t,j)) * ...
                            (reshape(obj.cov{i}(t,j,:,:), obj.p, obj.p) + eye(obj.p) * obj.eps);
                    end
                end
            end
        end

        function obj = update_tau(obj, m)
            % diminishing adaptation based on acceptance rates
            if (mod(m,100) == 0) && (m > obj.start_adapt_iter)
                delta = min(0.5, 5/sqrt(m+1));
                for i = 1:obj.nexp
                    obj.tau{i}(obj.count_100{i} < 23) = obj.tau{i}(obj.count_100{i} < 23) - delta;
                    obj.tau{i}(obj.count_100{i} > 23) = obj.tau{i}(obj.count_100{i} > 23) + delta;
                    obj.count_100{i}(:) = 0;
                end
            end
        end

        function x_cand = gen_cand(obj, x, m)
            % returns a cell of ntemps x ntheta(i) x p candidates
            x_cand = cell(1, obj.nexp);
            for i = 1:obj.nexp
                nth = obj.ntheta(i);
                xl = reshape(x{i}(m-1,:,:,:), obj.ntemps, nth, obj.p);
                xc = xl;
                for t = 1:obj.ntemps
                    for j = 1:nth
                        L = chol(reshape(obj.S{i}(t,j,:,:), obj.p, obj.p), 'lower');
                        xc(t,j,:) = reshape(xl(t,j,:), obj.p, 1) + L * randn(obj.p, 1);
                    end
                end
                x_cand{i} = xc;
            end
        end
    end
end

classdef AMcov_pool < handle
    % Stores and updates the proposal covariance for Adaptive Metropolis in
    % a pooled calibration (one covariance per temperature).
    %
    % Chains are stored as nmcmc x ntemps x p arrays. All slices are taken
    % with reshape (never squeeze) so ntemps == 1 and p == 1 both work.

    properties
        eps
        AM_SCALAR
        tau
        S
        cov
        mu
        ntemps
        p
        start_adapt_iter
        count_100
    end

    methods
        function obj = AMcov_pool(ntemps, p, start_var, start_adapt_iter, tau_start)
            arguments
                ntemps
                p
                start_var=1e-4;
                start_adapt_iter=300;
                tau_start=0.;
            end
            obj.eps = 1.0e-12;
            obj.AM_SCALAR = 2.4^2 / p;
            obj.tau  = repmat(tau_start, ntemps, 1);
            obj.S    = zeros(ntemps, p, p);
            for i = 1:ntemps
                obj.S(i,:,:) = eye(p) * start_var;
            end
            obj.cov  = zeros(ntemps, p, p);
            obj.mu   = zeros(ntemps, p);
            obj.ntemps = ntemps;
            obj.p = p;
            obj.start_adapt_iter = start_adapt_iter;
            obj.count_100 = zeros(ntemps, 1);
        end

        function obj = update(obj, x, m)
            % update the covariance of the previous MCMC samples; called in
            % the m-th iteration, so the latest value is x(m-1,:,:)
            if m > obj.start_adapt_iter
                xl = reshape(x(m-1,:,:), obj.ntemps, obj.p);
                obj.mu = obj.mu + (xl - obj.mu) ./ m;
                d = xl - obj.mu;
                for t = 1:obj.ntemps
                    obj.cov(t,:,:) = ((m-1)/m) * reshape(obj.cov(t,:,:), obj.p, obj.p) + ...
                        ((m-1)/(m*m)) * (d(t,:)' * d(t,:));
                end
                obj.setS();
            elseif m == obj.start_adapt_iter
                xs = reshape(x(1:m,:,:), m, obj.ntemps, obj.p);
                obj.mu = reshape(mean(xs, 1), obj.ntemps, obj.p);
                for t = 1:obj.ntemps
                    obj.cov(t,:,:) = cov(reshape(xs(:,t,:), m, obj.p));
                end
                obj.setS();
            end
        end

        function obj = setS(obj)
            for t = 1:obj.ntemps
                obj.S(t,:,:) = obj.AM_SCALAR * exp(obj.tau(t)) * ...
                    (reshape(obj.cov(t,:,:), obj.p, obj.p) + eye(obj.p) * obj.eps);
            end
        end

        function obj = update_tau(obj, m)
            % diminishing adaptation based on acceptance rate for each temperature
            if (mod(m,100) == 0) && (m > obj.start_adapt_iter)
                delta = min(0.5, 5/sqrt(m+1));
                obj.tau(obj.count_100 < 23) = obj.tau(obj.count_100 < 23) - delta;
                obj.tau(obj.count_100 > 23) = obj.tau(obj.count_100 > 23) + delta;
                obj.count_100(:) = 0;
            end
        end

        function x_cand = gen_cand(obj, x, m)
            % returns an ntemps x p candidate
            xl = reshape(x(m-1,:,:), obj.ntemps, obj.p);
            x_cand = zeros(obj.ntemps, obj.p);
            for t = 1:obj.ntemps
                L = chol(reshape(obj.S(t,:,:), obj.p, obj.p), 'lower');
                x_cand(t,:) = xl(t,:) + (L * randn(obj.p, 1))';
            end
        end
    end
end

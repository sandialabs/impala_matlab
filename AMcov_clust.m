classdef AMcov_clust < handle
    % Adaptive Metropolis proposal covariances for clustered calibration.
    %
    % Cluster labels switch, so adaptation cannot track the cluster thetas
    % directly. Instead it tracks the experiment-specific histories
    % (theta_hist, which always have the same dimension) with an AMcov_hier
    % object, and each cluster's covariance combines the history means and
    % covariances of the thetas currently assigned to it.

    properties
        hist       % AMcov_hier tracking theta_hist
        S          % ntemps x nclustmax x p x p proposal covariances
        cov_clust  % ntemps x nclustmax x p x p combined covariances
        tau        % ntemps x 1 (adaptation of tau is turned off)
        AM_SCALAR
        eps
        ntemps
        nclustmax
        p
        nexp
        ntheta
        start_adapt_iter
        count_100
    end

    methods
        function obj = AMcov_clust(nexp, ntheta, ntemps, p, nclustmax, start_var, start_adapt_iter, tau_start)
            arguments
                nexp
                ntheta
                ntemps
                p
                nclustmax
                start_var = 1e-4;
                start_adapt_iter = 300;
                tau_start = 0;
            end
            obj.hist = AMcov_hier(nexp, ntheta, ntemps, p, start_var, start_adapt_iter, tau_start);
            obj.S = repmat(reshape(eye(p) * start_var, 1, 1, p, p), ntemps, nclustmax);
            obj.cov_clust = zeros(ntemps, nclustmax, p, p);
            obj.tau = tau_start * ones(ntemps, 1);
            obj.AM_SCALAR = 2.4^2 / p;
            obj.eps = 1.0e-12;
            obj.ntemps = ntemps;
            obj.nclustmax = nclustmax;
            obj.p = p;
            obj.nexp = nexp;
            obj.ntheta = ntheta;
            obj.start_adapt_iter = start_adapt_iter;
            obj.count_100 = zeros(ntemps, 1);
        end

        function obj = update(obj, x, m, delta)
            % x: cell of theta_hist chains; delta: cell of current cluster
            % labels (ntemps x ntheta(i))
            if m < obj.start_adapt_iter
                return
            end
            obj.hist.update(x, m);
            p = obj.p;
            for t = 1:obj.ntemps
                for k = 1:obj.nclustmax
                    mus = zeros(0, p);
                    covs = zeros(p, p);
                    nmem = 0;
                    for i = 1:obj.nexp
                        js = find(delta{i}(t,:) == k);
                        for j = js
                            mus(end+1,:) = reshape(obj.hist.mu{i}(t,j,:), 1, p); %#ok<AGROW>
                            covs = covs + reshape(obj.hist.cov{i}(t,j,:,:), p, p);
                            nmem = nmem + 1;
                        end
                    end
                    if nmem == 0
                        C = zeros(p, p);
                    else
                        d = mus - mean(mus, 1);
                        C = (covs + d' * d) / nmem;
                    end
                    obj.cov_clust(t,k,:,:) = C;
                    obj.S(t,k,:,:) = obj.AM_SCALAR * exp(obj.tau(t)) * (C + eye(p) * obj.eps);
                end
            end
        end

        function obj = update_tau(obj, ~)
            % label switching makes acceptance rates hard to track, so tau
            % is not adapted (same as the python implementation)
        end
    end
end

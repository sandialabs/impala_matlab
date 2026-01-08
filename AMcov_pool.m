classdef AMcov_pool < handle

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
            obj.tau  = repelem(tau_start, ntemps);
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
            if m > obj.start_adapt_iter
                obj.mu = obj.mu + (squeeze(x(m-1,:,:))-obj.mu) ./ m;
                tmp = squeeze(x(m-1,:,:)) - obj.mu;
                if size(tmp,1) == 1
                    tmp = tmp';
                end
                obj.cov = ( ...
                    ((m-1)/m) * obj.cov  + ...
                    ((m-1)/(m*m)) * tensorproduct('tij', tmp , 'ti' , tmp , 'tj'));
                eyetmp = repmat(eye(obj.p),1,1,size(obj.cov,1));
                eyetmp = permute(eyetmp,[3,1,2]);
                obj.S = obj.AM_SCALAR * tensorproduct('ijk' , obj.cov + eyetmp * obj.eps , 'ijk' , exp(obj.tau') , 'i');
            elseif m == obj.start_adapt_iter
                obj.mu = squeeze(mean(x(1:m,:,:),1));
                obj.cov = cov_3d_pcm(x(1:m,:,:), obj.mu);
                eyetmp = repmat(eye(obj.p),1,1,size(obj.cov,1));
                eyetmp = permute(eyetmp,[3,1,2]);
                obj.S = obj.AM_SCALAR * tensorproduct('ijk' , obj.cov + eyetmp * obj.eps , 'ijk' , exp(obj.tau') , 'i');
            end
        end

        function obj = update_tau(obj, m)
            % diminishing adaptation based on acceptance rate for each temperature
            if (mod(m,100) == 0) && (m > obj.start_adapt_iter)
                delta = min(0.5, 5/sqrt(m+1));
                obj.tau(find(obj.count_100 < 23)) = obj.tau(find(obj.count_100 < 23)) - delta;
                obj.tau(find(obj.count_100 > 23)) = obj.tau(find(obj.count_100 > 23)) + delta;
                obj.count_100 = obj.count_100 * 0;
            end
        end

        function x_cand = gen_cand(obj, x, m)
            tmp = zeros(size(obj.S));
            for i = 1:size(tmp,1)
                tmp(i,:,:) = chol(squeeze(obj.S(i,:,:)), 'lower');
            end
            xtmp = squeeze(x(m-1,:,:));
            if size(xtmp,1) == 1
                xtmp = xtmp(:);
                tmp = squeeze(tmp);
            end
            x_cand = xtmp + tensorproduct('ij' , tmp , 'ijk' , randn(obj.ntemps, obj.p) , 'ik');
        end
    end
end

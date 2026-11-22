classdef SwapScheduler < handle
    % Swap scheduling and temperature-ladder adaptation for parallel
    % tempering, shared by calibPool, calibHier and calibClust.
    %
    % It decides which pairs of temperatures attempt a state swap in each
    % swap round, keeps running estimates of the swap rejection rate
    % between neighbouring temperatures, and (optionally) re-spaces the
    % inverse-temperature ladder during burn-in so that every neighbouring
    % pair has the same rejection rate.
    %
    % Swap schemes
    %   'deo'    (default) deterministic even/odd swaps between adjacent
    %            temperatures: non-reversible parallel tempering (Okabe et
    %            al. 2001; Syed et al. 2022). Rounds alternate between the
    %            pairs (1,2),(3,4),... and (2,3),(4,5),...  An accepted
    %            state keeps moving in the same direction along the ladder,
    %            so the ladder is crossed in O(N) rounds instead of O(N^2).
    %   'seo'    stochastic even/odd: the parity of each round is random
    %            (the reversible counterpart of 'deo', for comparison).
    %   'random' the scheme of earlier versions: floor(ntemps/2) disjoint
    %            pairs drawn uniformly at random from the whole ladder.
    %
    % Ladder adaptation
    %   The rejection rate r_k between itl(k) and itl(k+1) is estimated with
    %   the Rao-Blackwellised estimator 1 - min(1, exp(log_alpha)). The
    %   cumulative rejection Lambda (the communication barrier of Syed et
    %   al. 2022) is then made equally spaced along a new ladder (their
    %   Algorithm 4). The coldest and hottest inverse temperatures are kept
    %   fixed. Adaptation happens at the end of rounds whose length doubles
    %   (starting at 100 iterations) and stops at adapt_until, so the
    %   sampler is a fixed Markov kernel after burn-in.
    %
    % The sampler must use obj.itl in place of setup.itl and re-read it
    % (and rebuild anything derived from it) whenever maybe_adapt returns
    % true.
    %
    % References
    %   Okabe, Kawata, Okamoto & Mikami (2001). Chem. Phys. Lett. 335,
    %     435-439.
    %   Syed, Bouchard-Cote, Deligiannidis & Doucet (2022). Non-reversible
    %     parallel tempering: a scalable highly parallel MCMC scheme. JRSS-B
    %     84(2), 321-350.
    %   Vousden, Farr & Mandel (2016). MNRAS 455(2), 1919-1937.

    properties (Constant)
        SCHEMES = {'deo', 'seo', 'random'}
    end

    properties
        itl            % 1 x ntemps working inverse-temperature ladder
        ntemps
        scheme
        adapt
        adapt_until
        round_length
        next_adapt
        n_rounds
        n_adaptations
        rejection_total  % summed rejection per edge since the last adaptation
        attempts_total   % attempts per edge since the last adaptation
        round_trips      % completed cold -> hot -> cold replica round trips
    end

    properties (Access = private)
        rej_sum
        rej_n
        label
        seen_hot
    end

    methods
        function obj = SwapScheduler(itl, scheme, adapt, adapt_until, first_round)
            % itl: inverse temperatures (decreasing, itl(1) = 1)
            % scheme: 'deo' (default), 'seo' or 'random'
            % adapt: adapt the ladder (default false)
            % adapt_until: last iteration at which the ladder may change
            %              (default [] = no limit)
            % first_round: length of the first adaptation round (default 100)
            arguments
                itl
                scheme = 'deo'
                adapt = false
                adapt_until = []
                first_round = 100
            end
            SwapScheduler.validate(scheme, adapt);
            obj.itl = reshape(double(itl), 1, []);
            obj.ntemps = numel(obj.itl);
            obj.scheme = char(scheme);
            obj.adapt = logical(adapt) && obj.ntemps > 2;
            obj.adapt_until = adapt_until;
            obj.round_length = first_round;
            obj.next_adapt = [];
            obj.n_rounds = 0;
            obj.n_adaptations = 0;
            nedge = max(obj.ntemps - 1, 0);
            obj.rej_sum = zeros(1, nedge);
            obj.rej_n = zeros(1, nedge);
            obj.rejection_total = zeros(1, nedge);
            obj.attempts_total = zeros(1, nedge);
            % which replica sits at each temperature, and whether it has
            % reached the hottest temperature since it was last cold
            obj.label = 1:obj.ntemps;
            obj.seen_hot = false(1, obj.ntemps);
            if obj.ntemps > 1
                obj.seen_hot(obj.ntemps) = true;
            end
            obj.round_trips = 0;
        end

        function sw = pairs(obj)
            % npairs x 2 matrix of temperature indices for the next round,
            % with sw(:,1) < sw(:,2)
            switch obj.scheme
                case 'deo'
                    parity = mod(obj.n_rounds, 2);
                case 'seo'
                    parity = randi(2) - 1;
                otherwise
                    npair = floor(obj.ntemps / 2);
                    sw = sort(reshape(randperm(obj.ntemps, 2*npair), 2, npair), 1)';
                    obj.n_rounds = obj.n_rounds + 1;
                    return
            end
            obj.n_rounds = obj.n_rounds + 1;
            sw = SwapScheduler.even_odd_pairs(obj.ntemps, parity);
        end

        function record(obj, sw, log_alpha, accepted)
            % Record one round of swap attempts.
            %
            % sw: the pairs returned by pairs()
            % log_alpha: log acceptance ratio of each pair
            % accepted: logical, which swaps were accepted (used to count
            %           round trips)
            if isempty(sw)
                return
            end
            if nargin > 3 && ~isempty(accepted)
                obj.track_round_trips(sw(logical(accepted), :));
            end
            adjacent = sw(:,2) - sw(:,1) == 1;
            if ~any(adjacent)
                return
            end
            edge = sw(adjacent, 1);
            la = reshape(double(log_alpha), [], 1);
            la = la(adjacent);
            la(isnan(la)) = -Inf;
            rej = 1 - exp(min(la, 0));
            obj.rej_sum = obj.rej_sum + accumarray(edge, rej, [obj.ntemps-1 1])';
            obj.rej_n = obj.rej_n + accumarray(edge, 1, [obj.ntemps-1 1])';
            obj.rejection_total = obj.rejection_total + accumarray(edge, rej, [obj.ntemps-1 1])';
            obj.attempts_total = obj.attempts_total + accumarray(edge, 1, [obj.ntemps-1 1])';
        end

        function r = rejection_rates(obj)
            % estimated rejection rate on every adjacent edge (NaN if
            % unvisited since the last adaptation)
            r = obj.rejection_total ./ obj.attempts_total;
        end

        function changed = maybe_adapt(obj, m)
            % Adapt the ladder if an adaptation round has just finished.
            % m: current MCMC iteration. Returns true if obj.itl changed.
            changed = false;
            if ~obj.adapt
                return
            end
            if ~isempty(obj.adapt_until) && m > obj.adapt_until
                return
            end
            if isempty(obj.next_adapt)
                obj.next_adapt = m + obj.round_length;
                return
            end
            if m < obj.next_adapt || any(obj.rej_n == 0)
                return
            end
            rej = obj.rej_sum ./ obj.rej_n;
            obj.itl = SwapScheduler.communication_barrier_ladder(obj.itl, rej);
            obj.rej_sum(:) = 0;
            obj.rej_n(:) = 0;
            % report post-adaptation statistics only
            obj.rejection_total(:) = 0;
            obj.attempts_total(:) = 0;
            obj.round_length = obj.round_length * 2;
            obj.next_adapt = m + obj.round_length;
            obj.n_adaptations = obj.n_adaptations + 1;
            changed = true;
        end
    end

    methods (Access = private)
        function track_round_trips(obj, swapped)
            if isempty(swapped)
                return
            end
            a = swapped(:,1);
            b = swapped(:,2);
            la = obj.label(a);
            obj.label(a) = obj.label(b);
            obj.label(b) = la;
            obj.seen_hot(obj.label(obj.ntemps)) = true;
            cold = obj.label(1);
            if obj.seen_hot(cold)
                obj.round_trips = obj.round_trips + 1;
                obj.seen_hot(cold) = false;
            end
        end
    end

    methods (Static)
        function validate(scheme, adapt)
            if ~(ischar(scheme) || isstring(scheme)) || ~any(strcmp(scheme, SwapScheduler.SCHEMES))
                error('SwapScheduler:badScheme', ...
                      'swap_scheme must be one of: %s', strjoin(SwapScheduler.SCHEMES, ', '));
            end
            if adapt && strcmp(scheme, 'random')
                error('SwapScheduler:badAdapt', ...
                      'Ladder adaptation needs adjacent swaps; use swap_scheme ''deo'' or ''seo''.');
            end
        end

        function sw = even_odd_pairs(ntemps, parity)
            % Disjoint adjacent pairs (k, k+1) of 1-based indices, starting
            % at k = 1 for parity 0 and k = 2 for parity 1.
            first = (1 + parity):2:(ntemps - 1);
            sw = [first(:) first(:) + 1];
        end

        function new_itl = communication_barrier_ladder(itl, rejection, min_rejection)
            % Re-space an inverse-temperature ladder so that the estimated
            % cumulative rejection (communication barrier) is equally
            % spaced along it.
            %
            % itl: inverse temperatures, decreasing, itl(1) = 1
            % rejection: rejection rates between itl(k) and itl(k+1)
            % min_rejection: floor on the rates so the barrier is strictly
            %                increasing (default 1e-4)
            arguments
                itl
                rejection
                min_rejection = 1e-4
            end
            itl = reshape(double(itl), 1, []);
            rejection = min(max(reshape(double(rejection), 1, []), min_rejection), 1);
            % increasing grid (hot -> cold)
            beta = fliplr(itl);
            rej = fliplr(rejection);
            barrier = [0 cumsum(rej)];
            targets = linspace(0, barrier(end), numel(beta));
            new_beta = interp1(barrier, beta, targets);
            new_beta(1) = beta(1);
            new_beta(end) = beta(end);
            new_itl = fliplr(new_beta);
        end
    end
end

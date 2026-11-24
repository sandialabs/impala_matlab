function out = calibHier(setup)
% This function runs hierarchical Bayesian Model Calibration with adaptive
% MCMC, tempering, and decorrelation steps.
%
% Each experiment i can have several thetas (setup.ntheta(i), defined by
% the theta_ind argument of addVecExperiments). Every theta gets its own
% value, tied together by theta_ij ~ N(theta0, Sigma0) with
% theta0 ~ N(theta0_prior_mean, theta0_prior_cov) and
% Sigma0 ~ InvWishart(Sigma0_prior_df, Sigma0_prior_scale), all on the
% 0-1 scale, with theta_ij restricted to the bounds. Set these with
% setup.setHierPriors before calling.
%
% The restriction of theta_ij to the bounds contracts the effective
% hyperprior on theta0 and Sigma0; hier_truncation_weights measures that
% contraction for a finished run and reweights the output to remove it.
%
% Only the hierarchy and the likelihood are tempered; the hyperpriors on
% theta0 and Sigma0 enter every chain at full strength, so they cancel in
% the tempering swap ratio.
%
% Priors added with addThetaPrior/addJointThetaPrior are not used here
% (the hierarchical prior replaces them).
%
% setup: an object of class CalibSetup
%
% returns a struct with the following fields
% theta: cell, mcmc samples of theta per experiment (nmcmc x ntemps x
%        ntheta(i) x p, 0-1 scale)
% s2: cell, mcmc samples of error variance (nmcmc x ntemps x ns2)
% count: cell, acceptance counts per temperature and theta
% count_s2: acceptance counts for M-H error variance updates
% count_decor2: acceptance counts of the decorrelation step (ntemps x p)
% cov_theta_cand: final theta proposal covariances (AMcov_hier)
% cov_ls2_cand: final error variance proposal covariances
% count_temper: accepted tempering swaps (symmetric)
% pred_curr: current predictions
% theta0: mcmc samples of theta0 (nmcmc x ntemps x p, 0-1 scale)
% Sigma0: mcmc samples of Sigma0 (nmcmc x ntemps x p x p, 0-1 scale)
% llik: log likelihood at temperature 1
% theta_native: cell, theta at temperature 1 on the native scale (struct
%               of nmcmc x ntheta(i) arrays)
% theta0_native: theta0 at temperature 1 on the native scale
% temperature_ladder: inverse temperatures actually used (after any
%                     ladder adaptation, see setTemperatureLadder)
% swap_rejection: estimated swap rejection rate on each adjacent pair of
%                 temperatures (since the last ladder adaptation)
% round_trips: number of cold -> hot -> cold replica round trips
arguments
    setup CalibSetup
end

check_hier_setup(setup);
names = fieldnames(setup.bounds);
nt = setup.ntemps;
p = setup.p;
nexp = setup.nexp;
nmcmc = setup.nmcmc;
% working copy of the inverse-temperature ladder; adapted during burn-in
% when setup.adapt_ladder is true (see SwapScheduler)
sched = setup.makeSwapScheduler();
itl = sched.itl;
ntheta = setup.ntheta;
Ntheta = sum(ntheta);

% observation indices for each theta
obs = cell(1, nexp);
s2_ind_mat = cell(1, nexp);
for i = 1:nexp
    obs{i} = cell(1, ntheta(i));
    for j = 1:ntheta(i)
        obs{i}{j} = find(setup.theta_ind{i} == j);
    end
    s2_ind_mat{i} = double(setup.s2_ind{i}(:) == 1:setup.ns2{i});
end

% hyperpriors
theta0_prior_mean = setup.theta0_prior_mean;
theta0_prior_cov = setup.theta0_prior_cov;
theta0_prior_prec = inv(theta0_prior_cov);
theta0_prior_ldet = logdet(theta0_prior_cov);
Sigma0_prior_df = setup.Sigma0_prior_df;
Sigma0_prior_scale = setup.Sigma0_prior_scale;

% storage
theta0 = zeros(nmcmc, nt, p);
Sigma0 = zeros(nmcmc, nt, p, p);
theta = cell(1, nexp);
log_s2 = cell(1, nexp);
ls2_curr = cell(1, nexp);
for i = 1:nexp
    theta{i} = zeros(nmcmc, nt, ntheta(i), p);
    log_s2{i} = zeros(nmcmc, nt, setup.ns2{i});
    ls2_curr{i} = repmat(log(setup.sd_est{i}.^2), nt, 1);
end

% starting values
if ~isempty(setup.theta0_start)
    th0 = reshape(setup.theta0_start, nt, p);
else
    th0 = rand(nt, p);
    good = reshape(setup.checkConstraints(tran_unif(th0, setup.bounds_mat, names), setup.bounds), [], 1);
    ntry = 0;
    while any(~good)
        th0(~good,:) = rand(sum(~good), p);
        good(~good) = reshape(setup.checkConstraints(tran_unif(th0(~good,:), setup.bounds_mat, names), setup.bounds), [], 1);
        ntry = ntry + 1;
        if ntry > 10000
            error('calibHier:badStart', 'Failed to find a theta0 start that fulfills the constraints.');
        end
    end
end
Sig0 = repmat(reshape(Sigma0_prior_scale / (Sigma0_prior_df - p - 1), 1, p, p), nt, 1);
[Sig0_inv, Sig0_ldet] = inv_ldet(Sig0);

th = cell(1, nexp);
pred_curr = cell(1, nexp);
llik_curr = cell(1, nexp);
marg_lik_cov_cur = cell(1, nexp);
for i = 1:nexp
    nth = ntheta(i);
    th{i} = reshape(sample_constrained(repmat(th0, nth, 1), reshape(Sig0(1,:,:), p, p), setup), nt, nth, p);
    pred_curr{i} = eval_unpooled(i, th{i});
    marg_lik_cov_cur{i} = cell(nt, nth);
    llik_curr{i} = zeros(nt, nth);
    for t = 1:nt
        for j = 1:nth
            marg_lik_cov_cur{i}{t,j} = lik_cov(i, ls2_curr{i}(t,:), j);
            llik_curr{i}(t,j) = llik_tj(i, pred_curr{i}(t,:), j, marg_lik_cov_cur{i}{t,j});
        end
    end
end
store(1);

cov_theta_cand = AMcov_hier(nexp, ntheta, nt, p, setup.start_var_theta, setup.start_adapt_iter, setup.start_tau_theta);
cov_ls2_cand = cell(1, nexp);
for i = 1:nexp
    cov_ls2_cand{i} = AMcov_pool(nt, setup.ns2{i}, setup.start_var_ls2, setup.start_adapt_iter, setup.start_tau_ls2);
end

count_temper = zeros(nt, nt);
count = cell(1, nexp);
for i = 1:nexp
    count{i} = zeros(nt, ntheta(i));
end
count_decor2 = zeros(nt, p);
count_s2 = zeros(nexp, nt);
llik = zeros(1, nmcmc);
llik(1) = total_llik(1);

pbar = ProgressBar(nmcmc, 'Title', 'Running MCMC');
for m = 2:nmcmc

    % update emulators
    for i = 1:nexp
        setup.models{i}.step();
        if setup.models{i}.stochastic
            pred_curr{i} = eval_unpooled(i, th{i});
            for t = 1:nt
                for j = 1:ntheta(i)
                    llik_curr{i}(t,j) = llik_tj(i, pred_curr{i}(t,:), j, marg_lik_cov_cur{i}{t,j});
                end
            end
        end
    end

    % adaptive Metropolis for each theta_ij
    cov_theta_cand.update(theta, m);
    theta_cand = cov_theta_cand.gen_cand(theta, m);
    for i = 1:nexp
        nth = ntheta(i);
        good = good_thetas(theta_cand{i});
        th_eval = th{i};
        th_eval(repmat(good, 1, 1, p)) = theta_cand{i}(repmat(good, 1, 1, p));
        pred_cand = eval_unpooled(i, th_eval);
        llik_cand = llik_curr{i};
        for t = 1:nt
            for j = find(good(t,:))
                llik_cand(t,j) = llik_tj(i, pred_cand(t,:), j, marg_lik_cov_cur{i}{t,j});
            end
        end
        alpha = -Inf(nt, nth);
        lr = itl(:) .* (llik_cand - llik_curr{i} ...
            + mvn_logpdf_batch(theta_cand{i}, th0, Sig0_inv, Sig0_ldet) ...
            - mvn_logpdf_batch(th{i}, th0, Sig0_inv, Sig0_ldet));
        alpha(good) = lr(good);
        acc = log(rand(nt, nth)) < alpha;
        th{i}(repmat(acc, 1, 1, p)) = theta_cand{i}(repmat(acc, 1, 1, p));
        for t = 1:nt
            for j = find(acc(t,:))
                pred_curr{i}(t, obs{i}{j}) = pred_cand(t, obs{i}{j});
            end
        end
        llik_curr{i}(acc) = llik_cand(acc);
        count{i}(acc) = count{i}(acc) + 1;
        cov_theta_cand.count_100{i}(acc) = cov_theta_cand.count_100{i}(acc) + 1;
    end
    cov_theta_cand.update_tau(m);

    % update s2
    update_s2(m);

    % Gibbs update of theta0 (truncated to satisfy the constraints)
    tbar = zeros(nt, p);
    for i = 1:nexp
        tbar = tbar + reshape(sum(th{i}, 2), nt, p);
    end
    tbar = tbar / Ntheta;
    means0 = zeros(nt, p);
    covs0 = zeros(nt, p, p);
    for t = 1:nt
        Sinv = reshape(Sig0_inv(t,:,:), p, p);
        cc = inv(Ntheta * itl(t) * Sinv + theta0_prior_prec);
        cc = (cc + cc')/2;
        dd = itl(t) * Ntheta * Sinv * tbar(t,:)' + theta0_prior_prec * theta0_prior_mean';
        means0(t,:) = (cc * dd)';
        covs0(t,:,:) = cc;
    end
    th0 = sample_constrained(means0, covs0, setup);

    % Gibbs update of Sigma0
    for t = 1:nt
        mat = zeros(p, p);
        for i = 1:nexp
            d = reshape(th{i}(t,:,:), ntheta(i), p) - th0(t,:);
            mat = mat + d' * d;
        end
        sc = Sigma0_prior_scale + itl(t) * mat;
        Sig0(t,:,:) = iwishrnd((sc + sc')/2, Sigma0_prior_df + Ntheta * itl(t));
    end
    [Sig0_inv, Sig0_ldet] = inv_ldet(Sig0);

    % decorrelation step: shift theta0 and every theta_ij together
    if mod(m, setup.decor) == 0
        for k = 1:p
            z = 0.1 * randn;
            th0_cand = th0;
            th0_cand(:,k) = th0_cand(:,k) + z;
            good0 = reshape(setup.checkConstraints(tran_unif(th0_cand, setup.bounds_mat, names), setup.bounds), nt, 1);
            % theta0 prior: counted once and not tempered, consistent with
            % the Gibbs update of theta0
            alpha_tot = mvn_prior0(th0_cand) - mvn_prior0(th0);
            alpha_tot(~good0) = -Inf;
            cand = cell(1, nexp);
            pred_cand = cell(1, nexp);
            llik_cand = cell(1, nexp);
            for i = 1:nexp
                nth = ntheta(i);
                cand{i} = th{i};
                cand{i}(:,:,k) = cand{i}(:,:,k) + z;
                good = good_thetas(cand{i}) & good0;
                th_eval = th{i};
                th_eval(repmat(good, 1, 1, p)) = cand{i}(repmat(good, 1, 1, p));
                pred_cand{i} = eval_unpooled(i, th_eval);
                llik_cand{i} = llik_curr{i};
                for t = 1:nt
                    for j = 1:nth
                        llik_cand{i}(t,j) = llik_tj(i, pred_cand{i}(t,:), j, marg_lik_cov_cur{i}{t,j});
                    end
                end
                a = itl(:) .* (llik_cand{i} - llik_curr{i} ...
                    + mvn_logpdf_batch(cand{i}, th0_cand, Sig0_inv, Sig0_ldet) ...
                    - mvn_logpdf_batch(th{i}, th0, Sig0_inv, Sig0_ldet));
                a(~good) = -Inf;
                alpha_tot = alpha_tot + sum(a, 2);
            end
            acc = log(rand(nt, 1)) < alpha_tot;
            th0(acc,:) = th0_cand(acc,:);
            for i = 1:nexp
                th{i}(acc,:,:) = cand{i}(acc,:,:);
                pred_curr{i}(acc,:) = pred_cand{i}(acc,:);
                llik_curr{i}(acc,:) = llik_cand{i}(acc,:);
            end
            count_decor2(acc,k) = count_decor2(acc,k) + 1;
        end
    end

    % tempering swaps
    if m > setup.start_temper && nt > 1
        for kk = 1:setup.nswap
            sw = sched.pairs();
            if isempty(sw)
                continue
            end
            s1 = sw(:,1)'; s2s = sw(:,2)';
            dtl = itl(s2s) - itl(s1);
            lp = temper_logpost();
            sw_alpha = dtl .* (lp(s1) - lp(s2s));
            accepted = log(rand(1, numel(s1))) < sw_alpha;
            sched.record(sw, sw_alpha, accepted);
            for tti = find(accepted)
                tt = sw(tti,:)';
                rev = tt([2 1]);
                count_temper(tt(1), tt(2)) = count_temper(tt(1), tt(2)) + 1;
                for i = 1:nexp
                    th{i}(tt,:,:) = th{i}(rev,:,:);
                    ls2_curr{i}(tt,:) = ls2_curr{i}(rev,:);
                    pred_curr{i}(tt,:) = pred_curr{i}(rev,:);
                    llik_curr{i}(tt,:) = llik_curr{i}(rev,:);
                    marg_lik_cov_cur{i}(tt,:) = marg_lik_cov_cur{i}(rev,:);
                end
                th0(tt,:) = th0(rev,:);
                Sig0(tt,:,:) = Sig0(rev,:,:);
                Sig0_inv(tt,:,:) = Sig0_inv(rev,:,:);
                Sig0_ldet(tt) = Sig0_ldet(rev);
            end
        end
        if sched.maybe_adapt(m)
            itl = sched.itl;  % ladder changed
        end
    end

    store(m);
    llik(m) = total_llik(1);
    pbar.step([], [], []);
end
pbar.release();

s2 = cell(1, nexp);
theta_native = cell(1, nexp);
for i = 1:nexp
    s2{i} = exp(log_s2{i});
    theta_native{i} = struct();
    tn = tran_unif(reshape(theta{i}(:,1,:,:), nmcmc*ntheta(i), p), setup.bounds_mat, names);
    for k = 1:p
        theta_native{i}.(names{k}) = reshape(tn.(names{k}), nmcmc, ntheta(i));
    end
end

out.theta = theta;
out.s2 = s2;
out.count = count;
out.count_s2 = count_s2;
out.count_decor2 = count_decor2;
out.cov_theta_cand = cov_theta_cand;
out.cov_ls2_cand = cov_ls2_cand;
out.count_temper = count_temper + count_temper' - diag(diag(count_temper));
out.pred_curr = pred_curr;
out.theta0 = theta0;
out.Sigma0 = Sigma0;
out.llik = llik;
out.theta_native = theta_native;
out.theta0_native = tran_unif(reshape(theta0(:,1,:), nmcmc, p), setup.bounds_mat, names);
out.temperature_ladder = sched.itl;
out.swap_rejection = sched.rejection_rates();
out.round_trips = sched.round_trips;

    % ---------------------------------------------------------------------
    % nested helpers (share the sampler state)

    function pred = eval_unpooled(i, thi)
        % thi: ntemps x ntheta(i) x p -> ntemps x ny predictions
        X = reshape(thi, nt*ntheta(i), p);
        pred = setup.models{i}.eval(tran_unif(X, setup.bounds_mat, names), false);
    end

    function good = good_thetas(thi)
        nth = size(thi, 2);
        X = reshape(thi, nt*nth, p);
        good = reshape(setup.checkConstraints(tran_unif(X, setup.bounds_mat, names), setup.bounds), nt, nth);
    end

    function c = lik_cov(i, ls2_row, j)
        ob = obs{i}{j};
        c = setup.models{i}.lik_cov_inv(exp(ls2_row(setup.s2_ind{i}(ob))), ob, setup.wt{i}(ob));
    end

    function v = llik_tj(i, pred_row, j, c)
        ob = obs{i}{j};
        v = setup.models{i}.llik(setup.ys{i}(ob), pred_row(ob), c, setup.wt{i}(ob));
    end

    function v = mvn_prior0(x)
        d = x - theta0_prior_mean;
        v = -0.5*p*log(2*pi) - 0.5*theta0_prior_ldet - 0.5*sum((d*theta0_prior_prec).*d, 2);
    end

    function update_s2(m)
        for ii = 1:nexp
            mdl = setup.models{ii};
            nth = ntheta(ii);
            switch mdl.s2
                case {'gibbs', 'gibbs_trunc'}
                    trunc = strcmp(mdl.s2, 'gibbs_trunc');
                    dev_sq = (setup.wt{ii} .* (setup.ys{ii} - pred_curr{ii}).^2) * s2_ind_mat{ii};
                    for t2 = 1:nt
                        ls2_curr{ii}(t2,:) = draw_s2_gibbs(dev_sq(t2,:), setup, ii, itl(t2), trunc);
                        for j2 = 1:nth
                            marg_lik_cov_cur{ii}{t2,j2} = lik_cov(ii, ls2_curr{ii}(t2,:), j2);
                            llik_curr{ii}(t2,j2) = llik_tj(ii, pred_curr{ii}(t2,:), j2, marg_lik_cov_cur{ii}{t2,j2});
                        end
                    end
                case 'fix'
                    % error variance stays at sd_est^2
                otherwise
                    cov_ls2_cand{ii}.update(log_s2{ii}, m);
                    ls2_cand = cov_ls2_cand{ii}.gen_cand(log_s2{ii}, m);
                    llik_candi = zeros(nt, nth);
                    cov_candi = cell(nt, nth);
                    for t2 = 1:nt
                        for j2 = 1:nth
                            cov_candi{t2,j2} = lik_cov(ii, ls2_cand(t2,:), j2);
                            llik_candi(t2,j2) = llik_tj(ii, pred_curr{ii}(t2,:), j2, cov_candi{t2,j2});
                        end
                    end
                    kern = setup.s2_prior_kern{ii};
                    % Jacobian of the random walk on log(s2): it comes from the change
                    % of variables, not from the target, so it is not tempered; each
                    % chain targets [p(s2) L]^itl in s2, as the Gibbs updates and the
                    % swap ratio assume.
                    alpha_s2 = itl .* (sum(llik_candi, 2)' - sum(llik_curr{ii}, 2)') ...
                        + itl .* sum(kern(exp(ls2_cand), setup.ig_a{ii}, setup.ig_b{ii}), 2)' + sum(ls2_cand, 2)' ...
                        - itl .* sum(kern(exp(ls2_curr{ii}), setup.ig_a{ii}, setup.ig_b{ii}), 2)' - sum(ls2_curr{ii}, 2)';
                    for t2 = find(log(rand(1, nt)) < alpha_s2)
                        count_s2(ii,t2) = count_s2(ii,t2) + 1;
                        llik_curr{ii}(t2,:) = llik_candi(t2,:);
                        ls2_curr{ii}(t2,:) = ls2_cand(t2,:);
                        marg_lik_cov_cur{ii}(t2,:) = cov_candi(t2,:);
                        cov_ls2_cand{ii}.count_100(t2) = cov_ls2_cand{ii}.count_100(t2) + 1;
                    end
                    cov_ls2_cand{ii}.update_tau(m);
            end
        end
    end

    function lp = temper_logpost()
        % tempered part of the log posterior at each temperature (1 x ntemps).
        % The hyperpriors p(theta0) p(Sigma0) are not tempered (the Gibbs
        % updates use them at full strength in every chain), so they cancel
        % in the swap ratio and must not appear here.
        lp = zeros(1, nt);
        for ii = 1:nexp
            kern = s2_swap_kern(setup, ii);
            lp = lp + sum(kern(exp(ls2_curr{ii}), setup.ig_a{ii}, setup.ig_b{ii}), 2)' ...
                + sum(mvn_logpdf_batch(th{ii}, th0, Sig0_inv, Sig0_ldet), 2)' ...
                + sum(llik_curr{ii}, 2)';
        end
    end

    function v = total_llik(t2)
        v = 0;
        for ii = 1:nexp
            v = v + sum(llik_curr{ii}(t2,:));
        end
    end

    function store(m)
        theta0(m,:,:) = reshape(th0, 1, nt, p);
        Sigma0(m,:,:,:) = reshape(Sig0, 1, nt, p, p);
        for ii = 1:nexp
            theta{ii}(m,:,:,:) = reshape(th{ii}, 1, nt, ntheta(ii), p);
            log_s2{ii}(m,:,:) = reshape(ls2_curr{ii}, 1, nt, setup.ns2{ii});
        end
    end
end

function check_hier_setup(setup)
if isempty(setup.theta0_prior_mean) || isempty(setup.Sigma0_prior_df)
    error('calibHier:noPriors', ...
          'Set the hierarchical priors with setup.setHierPriors before calibrating.');
end
if ~isempty(setup.theta_prior)
    warning('calibHier:thetaPriorIgnored', ...
            'Priors from addThetaPrior/addJointThetaPrior are ignored by hierarchical and clustered calibration.');
end
end

function [Sinv, ldet] = inv_ldet(S)
nt = size(S, 1);
p = size(S, 2);
Sinv = zeros(size(S));
ldet = zeros(nt, 1);
for t = 1:nt
    St = reshape(S(t,:,:), p, p);
    Sinv(t,:,:) = inv(St);
    ldet(t) = logdet(St);
end
end

function v = logdet(A)
R = chol((A + A')/2);
v = 2*sum(log(diag(R)));
end

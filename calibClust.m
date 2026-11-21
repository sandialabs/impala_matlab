function out = calibClust(setup)
% This function runs clustered Bayesian Model Calibration with adaptive
% MCMC and tempering.
%
% Like calibHier, each experiment i can have several thetas (setup.ntheta(i),
% from the theta_ind argument of addVecExperiments), but the thetas are
% grouped into at most setup.nclustmax clusters through a Dirichlet process
% mixture. Every cluster has one theta value, drawn from
% N(theta0, Sigma0), with the hierarchical priors from setHierPriors and a
% Gamma(eta_prior_shape, eta_prior_rate) prior on the concentration eta
% from setClusterPriors.
%
% Priors added with addThetaPrior/addJointThetaPrior are not used here.
%
% setup: an object of class CalibSetup
%
% returns a struct with the following fields
% theta: cluster thetas (nmcmc x ntemps x nclustmax x p, 0-1 scale)
% theta_hist: cell, theta assigned to each experiment theta (nmcmc x
%             ntemps x ntheta(i) x p, 0-1 scale)
% s2: cell, mcmc samples of error variance (nmcmc x ntemps x ns2)
% count: accepted cluster theta moves per temperature
% count_s2: acceptance counts for M-H error variance updates
% count_temper: accepted tempering swaps (symmetric)
% pred_curr: current predictions
% theta0: mcmc samples of theta0 (nmcmc x ntemps x p)
% Sigma0: mcmc samples of Sigma0 (nmcmc x ntemps x p x p)
% delta: cell, cluster labels (nmcmc x ntemps x ntheta(i))
% eta: Dirichlet process concentration (nmcmc x ntemps)
% nclustmax: maximum number of clusters
% theta_am: final proposal covariances (AMcov_clust)
% llik: log likelihood at temperature 1
% theta_hist_native: cell, theta_hist at temperature 1 on the native scale
% theta0_native: theta0 at temperature 1 on the native scale
arguments
    setup CalibSetup
end

if isempty(setup.theta0_prior_mean) || isempty(setup.Sigma0_prior_df)
    error('calibClust:noPriors', ...
          'Set the hierarchical priors with setup.setHierPriors before calibrating.');
end
if ~isempty(setup.theta_prior)
    warning('calibClust:thetaPriorIgnored', ...
            'Priors from addThetaPrior/addJointThetaPrior are ignored by hierarchical and clustered calibration.');
end

names = fieldnames(setup.bounds);
nt = setup.ntemps;
p = setup.p;
nexp = setup.nexp;
nmcmc = setup.nmcmc;
K = setup.nclustmax;
itl = reshape(setup.itl, 1, []);
ntheta = setup.ntheta;
Ntheta = sum(ntheta);

obs = cell(1, nexp);
s2_ind_mat = cell(1, nexp);
for i = 1:nexp
    obs{i} = cell(1, ntheta(i));
    for j = 1:ntheta(i)
        obs{i}{j} = find(setup.theta_ind{i} == j);
    end
    s2_ind_mat{i} = double(setup.s2_ind{i}(:) == 1:setup.ns2{i});
end

theta0_prior_mean = setup.theta0_prior_mean;
theta0_prior_prec = inv(setup.theta0_prior_cov);
theta0_prior_ldet = logdet(setup.theta0_prior_cov);
Sigma0_prior_df = setup.Sigma0_prior_df;
Sigma0_prior_scale = setup.Sigma0_prior_scale;

% storage
theta = zeros(nmcmc, nt, K, p);
theta0 = zeros(nmcmc, nt, p);
Sigma0 = zeros(nmcmc, nt, p, p);
eta = zeros(nmcmc, nt);
theta_hist = cell(1, nexp);
delta = cell(1, nexp);
log_s2 = cell(1, nexp);
ls2_curr = cell(1, nexp);
for i = 1:nexp
    theta_hist{i} = zeros(nmcmc, nt, ntheta(i), p);
    delta{i} = zeros(nmcmc, nt, ntheta(i));
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
            error('calibClust:badStart', 'Failed to find a theta0 start that fulfills the constraints.');
        end
    end
end
Sig0 = repmat(reshape(Sigma0_prior_scale / (Sigma0_prior_df - p - 1), 1, p, p), nt, 1);
[Sig0_inv, Sig0_ldet] = inv_ldet(Sig0);
eta_curr = repmat(setup.eta_prior_shape / setup.eta_prior_rate, nt, 1);

dl = cell(1, nexp);
for i = 1:nexp
    dl{i} = randi(K, nt, ntheta(i));
end
if ~isempty(setup.theta_start)
    thc = reshape(setup.theta_start, nt, K, p);
else
    thc = reshape(sample_constrained(repmat(th0, K, 1), reshape(Sig0(1,:,:), p, p), setup), nt, K, p);
end
counts = cluster_counts();
ext = counts > 0;

pred_curr_delta = cell(1, nexp);   % ntemps x K x ny, prediction under each cluster
llik_curr_delta = cell(1, nexp);   % ntemps x K x ntheta(i)
pred_curr_theta = cell(1, nexp);   % ntemps x ny, prediction under assigned clusters
llik_curr_theta = cell(1, nexp);   % ntemps x ntheta(i)
marg_lik_cov_cur = cell(1, nexp);  % ntemps x ntheta(i) cell
for i = 1:nexp
    marg_lik_cov_cur{i} = cell(nt, ntheta(i));
    for t = 1:nt
        for j = 1:ntheta(i)
            marg_lik_cov_cur{i}{t,j} = lik_cov(i, ls2_curr{i}(t,:), j);
        end
    end
    pred_curr_delta{i} = eval_clusters(i, thc);
    llik_curr_delta{i} = zeros(nt, K, ntheta(i));
    refresh_llik_delta(i, true(nt, K));
    refresh_theta(i);
end
store(1);

cov_theta_cand = AMcov_clust(nexp, ntheta, nt, p, K, setup.start_var_theta, setup.start_adapt_iter, setup.start_tau_theta);
cov_ls2_cand = cell(1, nexp);
for i = 1:nexp
    cov_ls2_cand{i} = AMcov_pool(nt, setup.ns2{i}, setup.start_var_ls2, setup.start_adapt_iter, setup.start_tau_ls2);
end

count = zeros(nt, 1);
count_s2 = zeros(nexp, nt);
count_temper = zeros(nt, nt);
llik = zeros(1, nmcmc);
llik(1) = total_llik(1);

pbar = ProgressBar(nmcmc, 'Title', 'Running MCMC');
for m = 2:nmcmc

    % update emulators
    for i = 1:nexp
        setup.models{i}.step();
        if setup.models{i}.stochastic
            pred_curr_delta{i} = eval_clusters(i, thc);
            refresh_llik_delta(i, true(nt, K));
            refresh_theta(i);
        end
    end

    % update cluster assignments (Neal 2000, algorithm 8 with the empty
    % clusters as auxiliary components)
    counts = cluster_counts();
    for i = 1:nexp
        for j = 1:ntheta(i)
            rows = sub2ind([nt K], (1:nt)', dl{i}(:,j));
            counts(rows) = counts(rows) - 1;
            empty = counts == 0;
            djs = empty .* (eta_curr ./ (sum(empty, 2) + 1e-9));
            lw = log(counts + djs) + itl(:) .* llik_curr_delta{i}(:,:,j);
            lw = lw - max(lw, [], 2);
            cp = cumsum(exp(lw), 2);
            cp = cp ./ cp(:, end);
            newk = sum(rand(nt, 1) > cp, 2) + 1;
            dl{i}(:,j) = newk;
            rows = sub2ind([nt K], (1:nt)', newk);
            counts(rows) = counts(rows) + 1;
        end
    end
    ext = counts > 0;
    nclust = sum(ext, 2);
    for i = 1:nexp
        refresh_theta(i);
    end

    % update s2
    s2_changed = update_s2(m);

    % Gibbs update of theta0 (truncated to satisfy the constraints)
    means0 = zeros(nt, p);
    covs0 = zeros(nt, p, p);
    for t = 1:nt
        Sinv = reshape(Sig0_inv(t,:,:), p, p);
        thext = reshape(thc(t, ext(t,:), :), nclust(t), p);
        tbar = mean(thext, 1);
        cc = inv(nclust(t) * itl(t) * Sinv + theta0_prior_prec);
        cc = (cc + cc')/2;
        dd = itl(t) * nclust(t) * Sinv * tbar' + theta0_prior_prec * theta0_prior_mean';
        means0(t,:) = (cc * dd)';
        covs0(t,:,:) = cc;
    end
    th0 = sample_constrained(means0, covs0, setup);

    % Gibbs update of Sigma0 (occupied clusters only)
    for t = 1:nt
        d = reshape(thc(t, ext(t,:), :), nclust(t), p) - th0(t,:);
        sc = Sigma0_prior_scale + itl(t) * (d' * d);
        Sig0(t,:,:) = iwishrnd((sc + sc')/2, Sigma0_prior_df + nclust(t) * itl(t));
    end
    [Sig0_inv, Sig0_ldet] = inv_ldet(Sig0);

    % update cluster thetas: adaptive Metropolis for occupied clusters, a
    % fresh draw from N(theta0, Sigma0) for empty ones
    cov_theta_cand.update(theta_hist, m, dl);
    S = cov_theta_cand.S;
    if setup.scale_proposal
        S = S ./ max(counts, 1);
    end
    cand = thc;
    for t = 1:nt
        for k = find(ext(t,:))
            L = chol(reshape(S(t,k,:,:), p, p), 'lower');
            cand(t,k,:) = reshape(thc(t,k,:), p, 1) + L * randn(p, 1);
        end
    end
    good = good_clusters(cand) & ext;
    fresh = reshape(sample_constrained(repmat(th0, K, 1), reshape_covs(Sig0, K), setup), nt, K, p);
    th_eval = thc;
    th_eval(repmat(good, 1, 1, p)) = cand(repmat(good, 1, 1, p));
    th_eval(repmat(~ext, 1, 1, p)) = fresh(repmat(~ext, 1, 1, p));

    pred_cand_delta = cell(1, nexp);
    llik_cand_cl = zeros(nt, K);
    llik_curr_cl = zeros(nt, K);
    for i = 1:nexp
        pred_cand_delta{i} = eval_clusters(i, th_eval);
        for t = 1:nt
            for j = 1:ntheta(i)
                k = dl{i}(t,j);
                if good(t,k)
                    llik_cand_cl(t,k) = llik_cand_cl(t,k) + ...
                        llik_tj(i, reshape(pred_cand_delta{i}(t,k,:), 1, []), j, marg_lik_cov_cur{i}{t,j});
                end
                llik_curr_cl(t,k) = llik_curr_cl(t,k) + llik_curr_theta{i}(t,j);
            end
        end
    end
    alpha = -Inf(nt, K);
    lr = itl(:) .* (llik_cand_cl - llik_curr_cl ...
        + mvn_logpdf_batch(th_eval, th0, Sig0_inv, Sig0_ldet) ...
        - mvn_logpdf_batch(thc, th0, Sig0_inv, Sig0_ldet));
    alpha(good) = lr(good);
    acc = log(rand(nt, K)) < alpha;
    changed = acc | ~ext;
    thc(repmat(changed, 1, 1, p)) = th_eval(repmat(changed, 1, 1, p));
    for i = 1:nexp
        pred_curr_delta{i}(repmat(changed, 1, 1, size(pred_curr_delta{i}, 3))) = ...
            pred_cand_delta{i}(repmat(changed, 1, 1, size(pred_curr_delta{i}, 3)));
        if s2_changed(i)
            refresh_llik_delta(i, true(nt, K));
        else
            refresh_llik_delta(i, changed);
        end
        refresh_theta(i);
    end
    count = count + sum(acc, 2);
    cov_theta_cand.count_100 = cov_theta_cand.count_100 + sum(acc, 2);

    % Dirichlet process concentration
    eta_curr = sample_eta(eta_curr, nclust, Ntheta, setup.eta_prior_shape, setup.eta_prior_rate);

    % tempering swaps
    if m > setup.start_temper && nt > 1
        for kk = 1:setup.nswap
            sw = reshape(randperm(nt, 2*setup.nswap_per), 2, setup.nswap_per);
            dtl = itl(sw(2,:)) - itl(sw(1,:));
            lp = temper_logpost();
            sw_alpha = dtl .* (lp(sw(1,:)) - lp(sw(2,:)));
            for tti = find(log(rand(1, setup.nswap_per)) < sw_alpha)
                tt = sw(:, tti);
                rev = tt([2 1]);
                count_temper(tt(1), tt(2)) = count_temper(tt(1), tt(2)) + 1;
                thc(tt,:,:) = thc(rev,:,:);
                th0(tt,:) = th0(rev,:);
                Sig0(tt,:,:) = Sig0(rev,:,:);
                Sig0_inv(tt,:,:) = Sig0_inv(rev,:,:);
                Sig0_ldet(tt) = Sig0_ldet(rev);
                eta_curr(tt) = eta_curr(rev);
                ext(tt,:) = ext(rev,:);
                counts(tt,:) = counts(rev,:);
                for i = 1:nexp
                    dl{i}(tt,:) = dl{i}(rev,:);
                    ls2_curr{i}(tt,:) = ls2_curr{i}(rev,:);
                    marg_lik_cov_cur{i}(tt,:) = marg_lik_cov_cur{i}(rev,:);
                    pred_curr_delta{i}(tt,:,:) = pred_curr_delta{i}(rev,:,:);
                    llik_curr_delta{i}(tt,:,:) = llik_curr_delta{i}(rev,:,:);
                    pred_curr_theta{i}(tt,:) = pred_curr_theta{i}(rev,:);
                    llik_curr_theta{i}(tt,:) = llik_curr_theta{i}(rev,:);
                end
            end
        end
    end

    store(m);
    llik(m) = total_llik(1);
    pbar.step([], [], []);
end
pbar.release();

s2 = cell(1, nexp);
theta_hist_native = cell(1, nexp);
for i = 1:nexp
    s2{i} = exp(log_s2{i});
    theta_hist_native{i} = struct();
    tn = tran_unif(reshape(theta_hist{i}(:,1,:,:), nmcmc*ntheta(i), p), setup.bounds_mat, names);
    for k = 1:p
        theta_hist_native{i}.(names{k}) = reshape(tn.(names{k}), nmcmc, ntheta(i));
    end
end

out.theta = theta;
out.theta_hist = theta_hist;
out.s2 = s2;
out.count = count;
out.count_s2 = count_s2;
out.count_temper = count_temper + count_temper' - diag(diag(count_temper));
out.pred_curr = pred_curr_theta;
out.theta0 = theta0;
out.Sigma0 = Sigma0;
out.delta = delta;
out.eta = eta;
out.nclustmax = K;
out.theta_am = cov_theta_cand;
out.llik = llik;
out.theta_hist_native = theta_hist_native;
out.theta0_native = tran_unif(reshape(theta0(:,1,:), nmcmc, p), setup.bounds_mat, names);

    % ---------------------------------------------------------------------
    % nested helpers (share the sampler state)

    function c = cluster_counts()
        c = zeros(nt, K);
        for ii = 1:nexp
            for jj = 1:ntheta(ii)
                r = sub2ind([nt K], (1:nt)', dl{ii}(:,jj));
                c(r) = c(r) + 1;
            end
        end
    end

    function pred = eval_clusters(i, thk)
        % thk: ntemps x K x p -> ntemps x K x ny predictions
        X = reshape(thk, nt*K, p);
        pr = setup.models{i}.eval(tran_unif(X, setup.bounds_mat, names), true);
        pred = reshape(pr, nt, K, []);
    end

    function g = good_clusters(thk)
        X = reshape(thk, nt*K, p);
        g = reshape(setup.checkConstraints(tran_unif(X, setup.bounds_mat, names), setup.bounds), nt, K);
    end

    function C = reshape_covs(Sg, nk)
        % ntemps x p x p -> (ntemps*nk) x p x p, matching repmat(th0, nk, 1)
        C = repmat(Sg, nk, 1, 1);
    end

    function c = lik_cov(i, ls2_row, j)
        ob = obs{i}{j};
        c = setup.models{i}.lik_cov_inv(exp(ls2_row(setup.s2_ind{i}(ob))), ob);
    end

    function v = llik_tj(i, pred_row, j, c)
        ob = obs{i}{j};
        v = setup.models{i}.llik(setup.ys{i}(ob), pred_row(ob), c);
    end

    function refresh_llik_delta(i, which)
        % recompute llik under each cluster for the (t,k) pairs in which
        for t2 = 1:nt
            for k2 = find(which(t2,:))
                pr = reshape(pred_curr_delta{i}(t2,k2,:), 1, []);
                for j2 = 1:ntheta(i)
                    llik_curr_delta{i}(t2,k2,j2) = llik_tj(i, pr, j2, marg_lik_cov_cur{i}{t2,j2});
                end
            end
        end
    end

    function refresh_theta(i)
        % predictions / log likelihoods under the current assignments
        ny = size(pred_curr_delta{i}, 3);
        pred_curr_theta{i} = zeros(nt, ny);
        llik_curr_theta{i} = zeros(nt, ntheta(i));
        for t2 = 1:nt
            for j2 = 1:ntheta(i)
                k2 = dl{i}(t2,j2);
                ob = obs{i}{j2};
                pred_curr_theta{i}(t2, ob) = reshape(pred_curr_delta{i}(t2,k2,ob), 1, []);
                llik_curr_theta{i}(t2,j2) = llik_curr_delta{i}(t2,k2,j2);
            end
        end
    end

    function changed_s2 = update_s2(m)
        changed_s2 = false(1, nexp);
        for ii = 1:nexp
            mdl = setup.models{ii};
            nth = ntheta(ii);
            switch mdl.s2
                case {'gibbs', 'gibbs_trunc'}
                    changed_s2(ii) = true;
                    trunc = strcmp(mdl.s2, 'gibbs_trunc');
                    dev_sq = (setup.ys{ii} - pred_curr_theta{ii}).^2 * s2_ind_mat{ii};
                    for t2 = 1:nt
                        ls2_curr{ii}(t2,:) = draw_s2_gibbs(dev_sq(t2,:), setup, ii, itl(t2), trunc);
                        for j2 = 1:nth
                            marg_lik_cov_cur{ii}{t2,j2} = lik_cov(ii, ls2_curr{ii}(t2,:), j2);
                            llik_curr_theta{ii}(t2,j2) = llik_tj(ii, pred_curr_theta{ii}(t2,:), j2, marg_lik_cov_cur{ii}{t2,j2});
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
                            llik_candi(t2,j2) = llik_tj(ii, pred_curr_theta{ii}(t2,:), j2, cov_candi{t2,j2});
                        end
                    end
                    kern = setup.s2_prior_kern{ii};
                    alpha_s2 = itl .* (sum(llik_candi, 2)' - sum(llik_curr_theta{ii}, 2)') ...
                        + itl .* (sum(kern(exp(ls2_cand), setup.ig_a{ii}, setup.ig_b{ii}), 2)' + sum(ls2_cand, 2)') ...
                        - itl .* (sum(kern(exp(ls2_curr{ii}), setup.ig_a{ii}, setup.ig_b{ii}), 2)' + sum(ls2_curr{ii}, 2)');
                    for t2 = find(log(rand(1, nt)) < alpha_s2)
                        changed_s2(ii) = true;
                        count_s2(ii,t2) = count_s2(ii,t2) + 1;
                        llik_curr_theta{ii}(t2,:) = llik_candi(t2,:);
                        ls2_curr{ii}(t2,:) = ls2_cand(t2,:);
                        marg_lik_cov_cur{ii}(t2,:) = cov_candi(t2,:);
                        cov_ls2_cand{ii}.count_100(t2) = cov_ls2_cand{ii}.count_100(t2) + 1;
                    end
                    cov_ls2_cand{ii}.update_tau(m);
            end
        end
    end

    function v = mvn_prior0(x)
        dd0 = x - theta0_prior_mean;
        v = -0.5*p*log(2*pi) - 0.5*theta0_prior_ldet - 0.5*sum((dd0*theta0_prior_prec).*dd0, 2);
    end

    function lp = temper_logpost()
        % tempered part of the log posterior at each temperature (1 x ntemps)
        lp = (mvn_prior0(th0) + invwishart_logpdf(Sig0, Sigma0_prior_df, Sigma0_prior_scale) ...
            + gamma_logpdf(eta_curr, setup.eta_prior_shape, setup.eta_prior_rate))';
        lp = lp + sum(mvn_logpdf_batch(thc, th0, Sig0_inv, Sig0_ldet) .* ext, 2)';
        for ii = 1:nexp
            kern = setup.s2_prior_kern{ii};
            lp = lp + sum(kern(exp(ls2_curr{ii}), setup.ig_a{ii}, setup.ig_b{ii}), 2)' ...
                + sum(llik_curr_theta{ii}, 2)';
        end
    end

    function v = total_llik(t2)
        v = 0;
        for ii = 1:nexp
            v = v + sum(llik_curr_theta{ii}(t2,:));
        end
    end

    function store(m)
        theta(m,:,:,:) = reshape(thc, 1, nt, K, p);
        theta0(m,:,:) = reshape(th0, 1, nt, p);
        Sigma0(m,:,:,:) = reshape(Sig0, 1, nt, p, p);
        eta(m,:) = eta_curr';
        for ii = 1:nexp
            delta{ii}(m,:,:) = reshape(dl{ii}, 1, nt, ntheta(ii));
            log_s2{ii}(m,:,:) = reshape(ls2_curr{ii}, 1, nt, setup.ns2{ii});
            th_h = zeros(nt, ntheta(ii), p);
            for t2 = 1:nt
                th_h(t2,:,:) = reshape(thc(t2, dl{ii}(t2,:), :), 1, ntheta(ii), p);
            end
            theta_hist{ii}(m,:,:,:) = reshape(th_h, 1, nt, ntheta(ii), p);
        end
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

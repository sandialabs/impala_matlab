function out = calibPool(setup)
% This function runs pooled Bayesian Model Calibration with adapative MCMC,
% tempering, and decorrelation steps
%
% input theta will be normalized to 0-1 and sampled from uniform priors,
% unless priors have been added to setup with addThetaPrior or
% addJointThetaPrior, in which case those priors are used
%
% The error variance of each experiment is handled according to
% setup.models{i}.s2: 'gibbs' (conjugate Gibbs update), 'gibbs_trunc'
% (Gibbs update truncated to [sd_lower^2, sd_upper^2]), 'MH' (adaptive
% Metropolis-Hastings on log s2) or 'fix' (held at sd_est^2).
%
% setup: an object of class CalibSetup
%
% returns an struct with the following parameters
% theta: mcmc samples of variables (nmcmc x ntemps x p, 0-1 scale)
% s2: mcmc samples of error variance (cell, nmcmc x ntemps x ns2)
% count: number of counts of acceptance (diagonal) and of accepted
%        tempering swaps (off-diagonal, symmetric)
% count_s2: number of counts of acceptance on error variance
% count_decor: number of times decorrelation occured
% cov_theta_cand: final theta covariance
% cov_ls2_cand: final error covariance
% pred_curr: current emulator predictions
% discrep_vars: discrepency coefficients
% llik: log likelihood
% theta_native: mcmc samples of variables in native scale
% temperature_ladder: inverse temperatures actually used (after any
%                     ladder adaptation, see setTemperatureLadder)
% swap_rejection: estimated swap rejection rate on each adjacent pair of
%                 temperatures (since the last ladder adaptation)
% round_trips: number of cold -> hot -> cold replica round trips
arguments
    setup CalibSetup
end

names = fieldnames(setup.bounds);
nt = setup.ntemps;
p = setup.p;
nexp = setup.nexp;
% working copy of the inverse-temperature ladder; adapted during burn-in
% when setup.adapt_ladder is true (see SwapScheduler)
sched = setup.makeSwapScheduler();
itl = sched.itl;

theta = zeros(setup.nmcmc, nt, p);
log_s2 = cell(1,nexp);
ls2_curr = cell(1,nexp);
s2_ind_mat = cell(1,nexp);
for i = 1:nexp
    ns2 = setup.ns2{i};
    log_s2{i} = zeros(setup.nmcmc, nt, ns2);
    ls2_curr{i} = repmat(log(setup.sd_est{i}.^2), nt, 1);
    log_s2{i}(1,:,:) = reshape(ls2_curr{i}, 1, nt, ns2);
    s2_ind_mat{i} = double(setup.s2_ind{i}(:) == 1:ns2);
end

% starting values for theta: satisfy the constraints and have finite prior
th = rand(nt, p);
good = start_ok(th, setup);
ntry = 0;
while any(~good)
    th(~good,:) = rand(sum(~good), p);
    good(~good) = start_ok(th(~good,:), setup);
    ntry = ntry + 1;
    if ntry > 1e5
        error('calibPool:badStart', ...
              ['Could not find a starting value satisfying the constraints ' ...
               'and with finite theta prior density.']);
    end
end
theta(1,:,:) = reshape(th, 1, nt, p);

pred_curr = cell(1,nexp);
discrep_curr = cell(1,nexp);
discrep_vars = cell(1,nexp);
llik_curr = zeros(nexp, nt);
marg_lik_cov_cur = cell(1, nexp);
for i = 1:nexp
    marg_lik_cov_cur{i} = cell(1,nt);
    pred_curr{i} = setup.models{i}.eval(tran_unif(th, setup.bounds_mat, names), true);
    discrep_curr{i} = zeros(size(pred_curr{i}));
    discrep_vars{i} = zeros(setup.nmcmc, nt, setup.models{i}.nd);
    for t = 1:nt
        marg_lik_cov_cur{i}{t} = setup.models{i}.lik_cov_inv(exp(ls2_curr{i}(t, setup.s2_ind{i})), 1:setup.y_lens{i}, setup.wt{i});
        llik_curr(i,t) = setup.models{i}.llik(setup.ys{i}, pred_curr{i}(t,:), marg_lik_cov_cur{i}{t}, setup.wt{i});
    end
end

% current log-prior for theta at each temperature (zeros if no prior set)
lpr_curr = eval_theta_priors(tran_unif(th, setup.bounds_mat, names), setup.theta_prior);

cov_theta_cand = AMcov_pool(nt, p, setup.start_var_theta, setup.start_adapt_iter, setup.start_tau_theta);
cov_ls2_cand = cell(1,nexp);
for i = 1:nexp
    cov_ls2_cand{i} = AMcov_pool(nt, setup.ns2{i}, setup.start_var_ls2, setup.start_adapt_iter, setup.start_tau_ls2);
end

count = zeros(nt, nt);
count_s2 = zeros(nexp, nt);
count_decor = zeros(p, nt);
llik = zeros(1,setup.nmcmc);
llik(1) = sum(llik_curr(:,1));

% start MCMC
pbar = ProgressBar(setup.nmcmc, 'Title', 'Running MCMC');
for m = 2:setup.nmcmc

    for i = 1:nexp
        if setup.models{i}.nd > 0
            for t = 1:nt
                dv = setup.models{i}.discrep_sample(setup.ys{i}, pred_curr{i}(t,:), marg_lik_cov_cur{i}{t}, itl(t), setup.wt{i});
                discrep_vars{i}(m,t,:) = reshape(dv, 1, 1, []);
                discrep_curr{i}(t,:) = (setup.models{i}.D * dv(:))';
            end
        end

        setup.models{i}.step();
        if setup.models{i}.stochastic
            pred_curr{i} = setup.models{i}.eval(tran_unif(th, setup.bounds_mat, names), true);
        end
        if setup.models{i}.nd > 0 || setup.models{i}.stochastic
            for t = 1:nt
                llik_curr(i,t) = setup.models{i}.llik(setup.ys{i}-discrep_curr{i}(t,:), pred_curr{i}(t,:), marg_lik_cov_cur{i}{t}, setup.wt{i});
            end
        end
    end

    % adaptive Metropolis for each temperature
    cov_theta_cand.update(theta, m);
    theta_cand = cov_theta_cand.gen_cand(theta, m);
    [acc, llik_cand, pred_cand, lpr_cand] = mh_theta(theta_cand);
    th(acc,:) = theta_cand(acc,:);
    lpr_curr(acc) = lpr_cand(acc);
    llik_curr(:,acc) = llik_cand(:,acc);
    for i = 1:nexp
        pred_curr{i}(acc,:) = pred_cand{i}(acc,:);
    end
    idx = find(acc);
    count(sub2ind([nt nt], idx, idx)) = count(sub2ind([nt nt], idx, idx)) + 1;
    cov_theta_cand.count_100(acc) = cov_theta_cand.count_100(acc) + 1;

    % diminishing adaptation based on acceptance rate for each temperature
    cov_theta_cand.update_tau(m);

    % decorrelation step: independence proposal for one parameter at a time
    if mod(m, setup.decor) == 0
        for k = 1:p
            theta_cand = th;
            theta_cand(:,k) = rand(nt,1);
            [acc, llik_cand, pred_cand, lpr_cand] = mh_theta(theta_cand);
            th(acc,k) = theta_cand(acc,k);
            lpr_curr(acc) = lpr_cand(acc);
            llik_curr(:,acc) = llik_cand(:,acc);
            for i = 1:nexp
                pred_curr{i}(acc,:) = pred_cand{i}(acc,:);
            end
            count_decor(k,acc) = count_decor(k,acc) + 1;
        end
    end

    % update s2
    for i = 1:nexp
        mdl = setup.models{i};
        switch mdl.s2
            case {'gibbs', 'gibbs_trunc'}
                trunc = strcmp(mdl.s2, 'gibbs_trunc');
                dev_sq = (setup.wt{i} .* (setup.ys{i} - discrep_curr{i} - pred_curr{i}).^2) * s2_ind_mat{i};
                for t = 1:nt
                    ls2_curr{i}(t,:) = draw_s2_gibbs(dev_sq(t,:), setup, i, itl(t), trunc);
                    marg_lik_cov_cur{i}{t} = mdl.lik_cov_inv(exp(ls2_curr{i}(t, setup.s2_ind{i})), 1:setup.y_lens{i}, setup.wt{i});
                    llik_curr(i,t) = mdl.llik(setup.ys{i}-discrep_curr{i}(t,:), pred_curr{i}(t,:), marg_lik_cov_cur{i}{t}, setup.wt{i});
                end

            case 'fix'
                % error variance stays at sd_est^2

            otherwise
                % M-H update of log s2
                cov_ls2_cand{i}.update(log_s2{i}, m);
                ls2_cand = cov_ls2_cand{i}.gen_cand(log_s2{i}, m);

                llik_candi = zeros(1,nt);
                marg_lik_cov_candi = cell(1, nt);
                for t = 1:nt
                    marg_lik_cov_candi{t} = mdl.lik_cov_inv(exp(ls2_cand(t, setup.s2_ind{i})), 1:setup.y_lens{i}, setup.wt{i});
                    llik_candi(t) = mdl.llik(setup.ys{i}-discrep_curr{i}(t,:), pred_curr{i}(t,:), marg_lik_cov_candi{t}, setup.wt{i});
                end

                kern = setup.s2_prior_kern{i};
                alpha_s2 = itl .* (llik_candi - llik_curr(i,:)) ...
                    + itl .* (sum(kern(exp(ls2_cand), setup.ig_a{i}, setup.ig_b{i}), 2)' + sum(ls2_cand, 2)') ...
                    - itl .* (sum(kern(exp(ls2_curr{i}), setup.ig_a{i}, setup.ig_b{i}), 2)' + sum(ls2_curr{i}, 2)');

                for t = find(log(rand(1,nt)) < alpha_s2)
                    count_s2(i, t) = count_s2(i, t) + 1;
                    llik_curr(i, t) = llik_candi(t);
                    ls2_curr{i}(t,:) = ls2_cand(t,:);
                    marg_lik_cov_cur{i}{t} = marg_lik_cov_candi{t};
                    cov_ls2_cand{i}.count_100(t) = cov_ls2_cand{i}.count_100(t) + 1;
                end

                cov_ls2_cand{i}.update_tau(m);
        end
    end

    % tempering swaps
    if m > setup.start_temper && nt > 1
        for k = 1:setup.nswap
            sw = sched.pairs();
            if isempty(sw)
                continue
            end
            s1 = sw(:,1)';
            s2s = sw(:,2)';
            dtl = itl(s2s) - itl(s1);
            sw_alpha = dtl .* (sum(llik_curr(:, s1),1) - sum(llik_curr(:, s2s),1) ...
                + lpr_curr(s1) - lpr_curr(s2s));
            for i = 1:nexp
                % s2 prior kernel targeted by the within-chain s2 update
                kern = s2_swap_kern(setup, i);
                sw_alpha = sw_alpha + dtl .* ...
                    (sum(kern(exp(ls2_curr{i}(s1,:)), setup.ig_a{i}, setup.ig_b{i}), 2)' - ...
                     sum(kern(exp(ls2_curr{i}(s2s,:)), setup.ig_a{i}, setup.ig_b{i}), 2)');
                if setup.models{i}.nd > 0
                    nd = setup.models{i}.nd;
                    dv1 = reshape(discrep_vars{i}(m, s1, :), [], nd);
                    dv2 = reshape(discrep_vars{i}(m, s2s, :), [], nd);
                    sw_alpha = sw_alpha + dtl .* ...
                        (-0.5 * sum(dv1.^2, 2)' + 0.5 * sum(dv2.^2, 2)') ./ setup.models{i}.discrep_tau;
                end
            end
            accepted = log(rand(1, numel(s1))) < sw_alpha;
            sched.record(sw, sw_alpha, accepted);
            for tti = find(accepted)
                tt = sw(tti,:)';
                rev = tt([2 1]);
                for i = 1:nexp
                    ls2_curr{i}(tt,:) = ls2_curr{i}(rev,:);
                    marg_lik_cov_cur{i}(tt) = marg_lik_cov_cur{i}(rev);
                    pred_curr{i}(tt,:) = pred_curr{i}(rev,:);
                    if setup.models{i}.nd > 0
                        discrep_curr{i}(tt,:) = discrep_curr{i}(rev,:);
                        discrep_vars{i}(m,tt,:) = discrep_vars{i}(m,rev,:);
                    end
                end
                llik_curr(:,tt) = llik_curr(:,rev);
                count(tt(1),tt(2)) = count(tt(1),tt(2)) + 1;
                th(tt,:) = th(rev,:);
                lpr_curr(tt) = lpr_curr(rev);
            end
        end
        if sched.maybe_adapt(m)
            itl = sched.itl;  % ladder changed
        end
    end

    theta(m,:,:) = reshape(th, 1, nt, p);
    for i = 1:nexp
        log_s2{i}(m,:,:) = reshape(ls2_curr{i}, 1, nt, setup.ns2{i});
    end
    llik(m) = sum(llik_curr(:,1));
    pbar.step([], [], []);
end
pbar.release();

s2 = log_s2;
for i = 1:nexp
    s2{i} = exp(log_s2{i});
end

theta_native = tran_unif(reshape(theta(:,1,:), setup.nmcmc, p), setup.bounds_mat, names);

out.theta = theta;
out.s2 = s2;
out.count = count + count' - diag(diag(count));
out.count_s2 = count_s2;
out.count_decor = count_decor;
out.cov_theta_cand = cov_theta_cand;
out.cov_ls2_cand = cov_ls2_cand;
out.pred_curr = pred_curr;
out.discrep_vars = discrep_vars;
out.llik = llik;
out.theta_native = theta_native;
out.temperature_ladder = sched.itl;
out.swap_rejection = sched.rejection_rates();
out.round_trips = sched.round_trips;

    function [acc, llik_cand, pred_cand, lpr_cand] = mh_theta(theta_cand)
        % Metropolis accept/reject of an ntemps x p candidate for theta
        tu = tran_unif(theta_cand, setup.bounds_mat, names);
        good_values = reshape(setup.checkConstraints(tu, setup.bounds), 1, nt);
        pred_cand = pred_curr;
        llik_cand = llik_curr;
        lpr_cand = eval_theta_priors(tu, setup.theta_prior);
        good_values = good_values & isfinite(lpr_cand);
        if any(good_values)
            tg = tran_unif(theta_cand(good_values,:), setup.bounds_mat, names);
            for ii = 1:nexp
                pred_cand{ii}(good_values,:) = setup.models{ii}.eval(tg, true);
                for tk = find(good_values)
                    llik_cand(ii,tk) = setup.models{ii}.llik(setup.ys{ii}-discrep_curr{ii}(tk,:), ...
                        pred_cand{ii}(tk,:), marg_lik_cov_cur{ii}{tk}, setup.wt{ii});
                end
            end
        end
        alpha = -Inf(1, nt);
        diffs = (sum(llik_cand,1) + lpr_cand) - (sum(llik_curr,1) + lpr_curr);
        alpha(good_values) = itl(good_values) .* diffs(good_values);
        acc = log(rand(1,nt)) < alpha;
    end
end

function good = start_ok(th, setup)
% A starting value is usable when it satisfies the constraint function and
% has finite prior density. Without any theta priors the second condition is
% always true, so behavior is unchanged.
tu = tran_unif(th, setup.bounds_mat, fieldnames(setup.bounds));
good = setup.checkConstraints(tu, setup.bounds);
lp = eval_theta_priors(tu, setup.theta_prior);
good = reshape(good, [], 1) & reshape(isfinite(lp), [], 1);
end

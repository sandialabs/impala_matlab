function out = calibPool(setup)
% This function runs pooled Bayesian Model Calibration with adapative MCMC,
% tempering, and decorrelation steps
% 
% input theta will be normalized to 0-1 and sampled from uniform priors
%
% setup: an object of class CalibSetup
%
% returns an struct with the following parameters
% theta: mcmc samples of variables
% s2: mcmc samples of error variance
% count: number of counts of acceptance
% count_s2: number of counts of acceptance on error variance
% count_decor: number of times decorrelation occured
% cov_theta_cand: final theta covariance
% cov_ls2_cand: final error covariance
% pred_curr: current emulator predictions
% discrep_vars: discrepency coefficients
% llik: log likelihood
% theta_native: mcmc samples of variables in native scale
arguments
    setup CalibSetup
end

theta = zeros(setup.nmcmc, setup.ntemps, setup.p);
log_s2 = cell(1,setup.nexp);
s2_ind_mat = cell(1,setup.nexp);
for i = 1:setup.nexp
    log_s2{i} = ones(setup.nmcmc, setup.ntemps, setup.ns2{i});
    s2_ind_mat{i} = setup.s2_ind{i} == 1:setup.ns2{i};
end

theta_start = rand(setup.ntemps, setup.p);
good = setup.checkConstraints(tran_unif(theta_start, setup.bounds_mat, fieldnames(setup.bounds)), setup.bounds);

while any(~good)
    theta_start(~good,:) = rand(sum(~good), setup.p);
    good(~good) = setup.checkConstraints(tran_unif(theta_start(~good,:), setup.bounds_mat, fieldnames(setup.bounds)), setup.bounds);
end

theta(1,:,:) = theta_start;

% matrix of temperatures for use with alpha calculation--to skip nested for loops.
itl_mat = cell(1,setup.nexp);
for i = 1:setup.nexp
    itl_mat{i} = (ones(setup.ns2{i}, setup.ntemps) .* setup.itl)';
end

pred_curr = cell(1,setup.nexp);

llik_curr = zeros(setup.nexp, setup.ntemps);
marg_lik_cov_cur = cell(1, setup.nexp);
for i = 1:setup.nexp
    marg_lik_cov_cur{i} = cell(1,setup.ntemps);
    for t = 1:setup.ntemps
        tmp = exp(squeeze(log_s2{i}(1, t, setup.s2_ind{i})));
        marg_lik_cov_cur{i}{t} = setup.models{i}.lik_cov_inv(tmp(setup.s2_ind{i}));
    end
end

for i = 1:setup.nexp
    pred_curr{i} = setup.models{i}.eval(tran_unif(squeeze(theta(1,:,:)), setup.bounds_mat, fieldnames(setup.bounds)), true);

    for t = 1:setup.ntemps
        llik_curr(i,t) = setup.models{i}.llik(setup.ys{i}, pred_curr{i}(t,:), marg_lik_cov_cur{i}{t});
    end

end

cov_theta_cand = AMcov_pool(setup.ntemps, setup.p, setup.start_var_theta, setup.start_adapt_iter, setup.start_tau_theta);
cov_ls2_cand = cell(1,setup.nexp);
for i = 1:setup.nexp
    cov_ls2_cand{i} = AMcov_pool(setup.ntemps, setup.ns2{i}, setup.start_var_ls2, setup.start_adapt_iter, setup.start_tau_ls2);
end

count = zeros(setup.ntemps, setup.ntemps);
count_s2 = zeros(setup.nexp, setup.ntemps);
count_decor = zeros(setup.p, setup.ntemps);

discrep_curr = pred_curr;
for i = 1:length(discrep_curr)
    discrep_curr{i} = discrep_curr{i} * 0;
end

discrep_vars = cell(1, setup.nexp);
for i = 1:setup.nexp
    discrep_vars{i} = zeros(setup.nmcmc, setup.ntemps, setup.models{i}.nd);
end

alpha = ones(1,setup.ntemps) * -Inf;
sw_alpha = zeros(1,setup.nswap_per);

llik = zeros(1,setup.nmcmc);

% start MCMC
pbar = ProgressBar(setup.nmcmc, ...
       'Title', 'Running MCMC' ...
       );
for m = 2:setup.nmcmc
    theta(m,:,:) = theta(m-1,:,:);

    for i = 1:setup.nexp
        log_s2{i}(m,:) = log_s2{i}(m-1,:);
        if setup.models{i}.nd > 0
            for t = 1:setup.ntemps
                discrep_vars{i}(m,t,:) = setup.models{i}.discrep_sample(setup.ys{i}, pred_curr{i}(t,:), marg_lik_cov_cur{i}{t}, setup.itl(t));
                discrep_curr{i}(t,:) = setup.models{i}.D * discrep_vars{i}(m,t,:);
            end
        end

        setup.models{i}.step();
        if setup.models{i}.stochastic
            pred_curr{i} = setup.models{i}.eval(tran_unif(squeeze(theta(m,:,:)), setup.bounds_mat, fieldnames(setup.bounds)), true);
        end
        if setup.models{i}.nd > 0 || setup.models{i}.stochastic
            for t = 1:setup.ntemps
                llik_curr(i,t) = setup.models{i}.llik(setup.ys{i}-discrep_curr{i}(t,:), pred_curr{i}(t,:), marg_lik_cov_cur{i}{t});
            end
        end
    end

    % adaptive Metropolis for each temperature

    cov_theta_cand.update(theta, m);

    % generate proposal
    theta_cand = cov_theta_cand.gen_cand(theta, m);
    good_values = setup.checkConstraints(tran_unif(theta_cand, setup.bounds_mat, fieldnames(setup.bounds)), setup.bounds);

    % get predictions and SSE
    pred_cand = pred_curr;
    llik_cand = llik_curr;
    if any(good_values)
        llik_cand(:, good_values) = 0;
        for i = 1:setup.nexp
            pred_cand{i}(good_values,:) = setup.models{i}.eval(tran_unif(theta_cand(good_values,:), setup.bounds_mat, fieldnames(setup.bounds)), true);
            for t = 1:setup.ntemps
                llik_cand(i,t) = setup.models{i}.llik(setup.ys{i}-discrep_curr{i}(t,:), pred_cand{i}(t,:), marg_lik_cov_cur{i}{t});
            end
        end
    end

    llik_diff = (sum(llik_cand,1) - sum(llik_curr,1));
    llik_diff = llik_diff(good_values);

    alpha(:) = -Inf;
    alpha(good_values) = setup.itl(good_values) .* llik_diff;
    idx = find(log(rand(1,setup.ntemps)) < alpha);
    for t = idx
        theta(m,t,:) = theta_cand(t,:);
        count(t,t) = count(t,t) + 1;
        for i = 1:setup.nexp
            llik_curr(i,t) = llik_cand(i,t);
            pred_curr{i}(t,:) = pred_cand{i}(t,:);
        end
        cov_theta_cand.count_100(t) = cov_theta_cand.count_100(t) + 1;
    end

    % diminishing adaptation based on acceptance rate for each temperature
    cov_theta_cand.update_tau(m);

    % decorrelation step
    if mod(m, setup.decor) == 0
        for k = 1:setup.p
            theta_cand = squeeze(theta(m,:,:));
            theta_cand(:,k) = rand(1,setup.ntemps);
            good_values = setup.checkConstraints(tran_unif(theta_cand, setup.bounds_mat, fieldnames(setup.bounds)), setup.bounds);
            pred_cand = pred_curr;
            llik_cand = llik_curr;

            if any(good_values)
                llik_cand(:, good_values) = 0;
                for i = 1:setup.nexp
                    pred_cand{i}(good_values,:) = setup.models{i}.eval(tran_unif(theta_cand(good_values,:), setup.bounds_mat, fieldnames(setup.bounds)), true);
                    for t = 1:setup.ntemps
                        llik_cand(i,t) = setup.models{i}.llik(setup.ys{i}-discrep_curr{i}(t,:), pred_cand{i}(t,:), marg_lik_cov_cur{i}{t});
                    end
                end
            end

            alpha(:) = -Inf;

            llik_diff = (sum(llik_cand,1) - sum(llik_curr,1));
            llik_diff = llik_diff(good_values);

            alpha(good_values) = setup.itl(good_values) .* llik_diff;

            idx = find(log(rand(1,setup.ntemps)) < alpha);
            for t = idx
                theta(m,t,k) = theta_cand(t,k);
                count_decor(k,t) = count_decor(k,t) + 1;
                for i = 1:setup.nexp
                    pred_curr{i}(t,:) = pred_cand{i}(t,:);
                    llik_curr(i,t) = llik_cand(i,t);
                end
            end

        end
    end

    % update s2
    for i = 1:setup.nexp

        if strcmp(setup.models{i}.s2,'gibbs')
            % gibbs update s2
            difftmp = bsxfun(@minus,pred_curr{i},setup.ys{i});  % sqaured deviations
            dev_sq = (difftmp).^2 * s2_ind_mat{i}(1:length(setup.ys{i}))';
            log_s2{i}(m,:) = log(1./gamrnd(itl_mat{i}.*(setup.ny_s2{i}./2 + setup.ig_a{i} + 1)-1, 1./(itl_mat{i}.*(setup.ig_b{i} + dev_sq ./ 2))));
            for t = 1:setup.ntemps
                tmpi = exp(log_s2{i}(m,t));
                marg_lik_cov_cur{i}{t} = setup.models{i}.lik_cov_inv(tmpi(setup.s2_ind{i}));
                llik_curr(i, t) = setup.models{i}.llik(setup.ys{i}-discrep_curr{i}(t,:), pred_curr{i}(t,:), marg_lik_cov_cur{i}{t});
            end

        else
            % M-H update s2

            cov_ls2_cand{i}.update(log_s2{i}, m);
            ls2_candi = cov_ls2_cand{i}.gen_cand(log_s2{i}, m);

            llik_candi = zeros(1,setup.ntemps);
            marg_lik_cov_candi = cell(1, setup.ntemps);

            for t = 1:setup.ntemps
                tmpi = exp(ls2_candi(t));
                marg_lik_cov_candi{t} = setup.models{i}.lik_cov_inv(tmpi(setup.s2_ind{i}));
                llik_candi(t) = setup.models{i}.llik(setup.ys{i}-discrep_curr{i}(t,:), pred_curr{i}(t,:), marg_lik_cov_candi{t});
            end

            llik_diffi = (llik_candi - llik_curr(i,:));
            alpha_s2 = setup.itl .* llik_diffi;
            alpha_s2 = alpha_s2 + setup.itl .* sum(setup.s2_prior_kern{i}(exp(ls2_candi), setup.ig_a{i}, setup.ig_b{i}),2)';
            alpha_s2 = alpha_s2 + setup.itl .* sum(ls2_candi,2)';
            alpha_s2 = alpha_s2 - setup.itl .* sum(setup.s2_prior_kern{i}(exp(log_s2{i}(m-1,:)'), setup.ig_a{i}, setup.ig_b{i}),2)';
            alpha_s2 = alpha_s2 - setup.itl .* sum(log_s2{i}(m-1,:),1);

            idx = find(log(rand(1,setup.ntemps)) < alpha_s2);
            for t = idx
                count_s2(i, t) = count_s2(i, t) + 1;
                llik_curr(i, t) = llik_candi(t);
                log_s2{i}(m,t) = ls2_candi(t);
                marg_lik_cov_cur{i}{t} = marg_lik_cov_candi{t};
                cov_ls2_cand{i}.count_100 = cov_ls2_cand{i}.count_100 + 1;
            end

            cov_ls2_cand{i}.update_tau(m);
        end
    end

    % tempering swaps
    if m > setup.start_temper && setup.ntemps > 1
        for k = 1:setup.nswap
            sw = randsample(1:setup.ntemps, 2*setup.nswap_per, false);
            sw = reshape(sw,setup.nswap_per,2);
            sw = sw';
            sw_alpha(:) = 0;
            sw_alpha = sw_alpha + (setup.itl(sw(2,:)) - setup.itl(sw(1,:))).*(sum(llik_curr(:, sw(1,:)),1)-sum(llik_curr(:, sw(2,:)),1));
            for i = 1:setup.nexp
                sw_alpha = sw_alpha + (setup.itl(sw(2,:)) - setup.itl(sw(1,:))) .* ...
                    (sum(setup.s2_prior_kern{i}(exp(log_s2{i}(m,sw(1,:))), setup.ig_a{i}, setup.ig_b{i}),1) - ...
                    sum(setup.s2_prior_kern{i}(exp(log_s2{i}(m,sw(2,:))), setup.ig_a{i}, setup.ig_b{i}),1));
                if setup.models{i}.nd > 0
                    sw_alpha = sw_alpha + (setup.itl(sw(2,:)) - setup.itl(sw(1,:))) .* ...
                        (-0.5 * sum(discrep_vars{i}(m,(sw(1,:))).^2,2) ./ setup.models{i}.discrep_tau + ...
                        0.5 * sum(discrep_vars{i}(m,(sw(2,:))).^2,2) ./ setup.models{i}.discrep_tau);
                end
            end
            idx = find(log(rand(1,setup.nswap_per)) < sw_alpha);
            for tti = idx
                tt = sw(:,tti);
                for i = 1:setup.nexp
                    log_s2{i}(m,tt(1)) = log_s2{i}(m,tt(2));
                    log_s2{i}(m,tt(2)) = log_s2{i}(m,tt(1));
                    marg_lik_cov_cur{i}(tt(1)) = marg_lik_cov_cur{i}(tt(2));
                    marg_lik_cov_cur{i}(tt(2)) = marg_lik_cov_cur{i}(tt(1));
                    pred_curr{i}(tt(1),:) = pred_curr{i}(tt(2),:);
                    pred_curr{i}(tt(2),:) = pred_curr{i}(tt(1),:);
                    if setup.models{i}.nd > 0
                        discrep_curr{i}(tt(1),:) = discrep_curr{i}(tt(2),:);
                        discrep_curr{i}(tt(2),:) = discrep_curr{i}(tt(1),:);
                        discrep_vars{i}(m,tt(1)) = discrep_vars{i}(m,tt(2));
                        discrep_vars{i}(m,tt(2)) = discrep_vars{i}(m,tt(1));
                    end
                    llik_curr(i, tt(1)) = llik_curr(i, tt(2));
                    llik_curr(i, tt(2)) = llik_curr(i, tt(1));
                end
                count(tt(1),tt(2)) = count(tt(1),tt(2)) + 1;
                theta(m,tt(1),:) = theta(m,tt(2),:);
                theta(m,tt(2),:) = theta(m,tt(1),:);
            end
        end
    end

    llik(m) = sum(llik_curr(:,1));
    pbar.step([], [], []);
end
pbar.release();

s2 = log_s2;
for i = 1:setup.nexp
    s2{i} = exp(log_s2{i});
end

theta_native = tran_unif(squeeze(theta(:,1,:)), setup.bounds_mat, fieldnames(setup.bounds));

out.theta = theta;
out.s2 = s2;
out.count = count;
out.count_s2 = count_s2;
out.count_decor = count_decor;
out.cov_theta_cand = cov_theta_cand;
out.cov_ls2_cand = cov_ls2_cand;
out.pred_curr = pred_curr;
out.discrep_vars = discrep_vars;
out.llik = llik;
out.theta_native = theta_native;

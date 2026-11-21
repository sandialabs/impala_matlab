function tests = temperingTest
% Tests for SwapScheduler (swap schemes, round trips, ladder adaptation)
% and the tempering options of CalibSetup.setTemperatureLadder.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testEvenOddPairs(tc)
tc.verifyEqual(SwapScheduler.even_odd_pairs(5, 0), [1 2; 3 4]);
tc.verifyEqual(SwapScheduler.even_odd_pairs(5, 1), [2 3; 4 5]);
tc.verifySize(SwapScheduler.even_odd_pairs(1, 0), [0 2]);
end

function testDeoAlternatesParity(tc)
s = SwapScheduler(1 ./ 1.1.^(0:5), 'deo');
sw = s.pairs(); tc.verifyEqual(sw(:,1)', [1 3 5]);
sw = s.pairs(); tc.verifyEqual(sw(:,1)', [2 4]);
sw = s.pairs(); tc.verifyEqual(sw(:,1)', [1 3 5]);
end

function testRandomSchemePairsAreDisjoint(tc)
rng(1);
s = SwapScheduler(1 ./ 1.1.^(0:6), 'random');
sw = s.pairs();
tc.verifySize(sw, [3 2]);
tc.verifyNumElements(unique(sw), 6);
tc.verifyTrue(all(sw(:,1) < sw(:,2)));
end

function testBadOptions(tc)
tc.verifyError(@() SwapScheduler(ones(1,3), 'nope'), 'SwapScheduler:badScheme');
tc.verifyError(@() SwapScheduler(ones(1,3), 'random', true), 'SwapScheduler:badAdapt');
[~, ~, bounds] = calibToyProblem();
s = CalibSetup(bounds);
tc.verifyError(@() s.setTemperatureLadder(1:3, 100, 'nope'), 'SwapScheduler:badScheme');
tc.verifyError(@() s.setTemperatureLadder(1:3, 100, 'random', true), 'SwapScheduler:badAdapt');
end

function testRoundTripRateMatchesTheory(tc)
% With every swap accepted, DEO moves each replica cold -> hot -> cold in
% 2(N - 1) rounds, so N replicas complete about rounds/2 round trips
% (Syed et al. 2022, Theorem 3 with zero rejection). The reversible
% stochastic even/odd scheme is diffusive and much slower.
rng(0);
ntemps = 20; rounds = 20000;
schemes = {'deo', 'seo'};
trips = zeros(1, 2);
for k = 1:2
    s = SwapScheduler(1 ./ 1.1.^(0:ntemps-1), schemes{k});
    for r = 1:rounds
        sw = s.pairs();
        s.record(sw, zeros(size(sw,1),1), true(size(sw,1),1));
    end
    trips(k) = s.round_trips;
end
tc.verifyLessThan(abs(trips(1) - rounds/2), 0.05 * rounds/2);
tc.verifyLessThan(trips(2), trips(1) / 5);
end

function testBarrierLadderEqualisesRejection(tc)
% If rejection is proportional to the gap in inverse temperature, the
% re-spaced ladder must have equal gaps (and hence equal rejection).
itl = 1 ./ 1.3.^(0:9);
rej = 0.5 * -diff(itl);
new = SwapScheduler.communication_barrier_ladder(itl, rej);
tc.verifyEqual(new(1), itl(1));
tc.verifyEqual(new(end), itl(end));
tc.verifyTrue(all(diff(new) < 0));
gaps = -diff(new);
tc.verifyEqual(gaps, repmat(mean(gaps), size(gaps)), 'AbsTol', 1e-12);
end

function testSchedulerAdaptationSchedule(tc)
before = 1 ./ 1.3.^(0:5);
s = SwapScheduler(before, 'deo', true, 1000);
changed = [];
for m = 1:1999
    sw = s.pairs();
    % rejection grows towards the cold end
    la = log(1 - 0.1 * (numel(before) - sw(:,1) + 1) / numel(before));
    s.record(sw, la, false(size(sw,1),1));
    if s.maybe_adapt(m)
        changed(end+1) = m; %#ok<AGROW>
    end
end
tc.verifyNotEmpty(changed);
tc.verifyLessThanOrEqual(max(changed), 1000);
tc.verifyEqual(s.itl(1), before(1));
tc.verifyEqual(s.itl(end), before(end), 'AbsTol', 1e-12);
tc.verifyGreaterThan(max(abs(s.itl - before)), 1e-6);
end

function testPooledGaussianPosterior(tc)
% Cold chain recovers the analytic posterior mean and sd for each scheme.
cases = {'deo', false; 'deo', true; 'random', false};
for c = 1:size(cases, 1)
    rng(3);
    nobs = 30;
    truth = [0.3 0.6];
    y = repelem(truth, nobs) + 0.05*randn(1, 2*nobs);
    f = @(X) repelem(X, 1, nobs);
    bounds = struct('a', [0 1], 'b', [0 1]);
    s = CalibSetup(bounds);
    s.addVecExperiments(y, ModelF(f, {'a','b'}, NaN, 'fix'), 0.05, 0, ones(1, 2*nobs));
    s.setTemperatureLadder(1.5.^(0:7), 200, cases{c,1}, cases{c,2});
    s.setMCMC(4000, 1500, 1, 100);
    evalc('o = calibPool(s);');
    post_mean = mean(reshape(y, nobs, 2), 1);
    post_sd = 0.05 / sqrt(nobs);
    th = reshape(o.theta(1501:end, 1, :), [], 2);
    lbl = sprintf('%s adapt=%d', cases{c,1}, cases{c,2});
    tc.verifyLessThan(abs(mean(th, 1) - post_mean), 0.25 * post_sd * [1 1], lbl);
    tc.verifyLessThan(abs(std(th, 0, 1) / post_sd - 1), 0.25 * [1 1], lbl);
    tc.verifySize(o.temperature_ladder, [1 8], lbl);
    tc.verifyEqual(o.temperature_ladder(1), 1, lbl);
    tc.verifySize(o.swap_rejection, [1 7], lbl);
    tc.verifyGreaterThanOrEqual(o.round_trips, 0, lbl);
    if cases{c,2}
        tc.verifyGreaterThan(max(abs(o.temperature_ladder - 1 ./ 1.5.^(0:7))), 1e-6, lbl);
    else
        tc.verifyEqual(o.temperature_ladder, 1 ./ 1.5.^(0:7), 'AbsTol', 1e-12, lbl);
    end
end
end

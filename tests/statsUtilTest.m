function tests = statsUtilTest
% Tests for statistical utilities: iqrange, plims, acf, autocorrelation.
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

% ---- iqrange -------------------------------------------------------------

function testIqrangeColumnVector(tc)
% For a column, iqrange returns a scalar interquartile-range estimate.
x = (1:100)';
y = iqrange(x);
% This estimator uses linear interpolation of order statistics; just
% verify it is positive and within the full data range.
tc.verifyGreaterThan(y, 0);
tc.verifyLessThanOrEqual(y, max(x) - min(x));
end

function testIqrangeRowVectorTransposed(tc)
% Row input is transposed internally and returns a scalar.
x = 1:100;
y = iqrange(x);
tc.verifySize(y, [1 1]);
tc.verifyGreaterThan(y, 0);
end

function testIqrangePerColumn(tc)
% Two independent columns -> two IQR values.
x = [(1:100)', (1:100)' * 2];
y = iqrange(x);
tc.verifySize(y, [1 2]);
% Second column is scaled by 2, so its spread is ~2x the first.
tc.verifyEqual(y(2), 2 * y(1), 'RelTol', 1e-10);
end

% ---- plims ---------------------------------------------------------------

function testPlimsMedian(tc)
x = (1:101)';
y = plims(x, 0.5);
tc.verifyEqual(y, 51, 'AbsTol', 1e-10);
end

function testPlimsDefaultQuartiles(tc)
x = (1:101)';
y = plims(x);      % default [0.25 0.5 0.75]
tc.verifyEqual(y(2), 51, 'AbsTol', 1e-10);
tc.verifyTrue(y(1) < y(2) && y(2) < y(3));
end

function testPlimsEndpoints(tc)
x = (0:10)';
tc.verifyEqual(plims(x, 0), 0, 'AbsTol', 1e-10);
tc.verifyEqual(plims(x, 1), 10, 'AbsTol', 1e-10);
end

% ---- acf / autocorrelation ----------------------------------------------

function testAcfLagZeroIsOne(tc)
% ACF is normalized so lag-0 autocorrelation equals 1.
rng(11);
x = randn(200, 1);
y = acf(x, 20);
tc.verifyEqual(y(1), 1, 'AbsTol', 1e-12);
end

function testAcfLength(tc)
x = randn(500, 1);
y = acf(x, 30);
tc.verifyEqual(numel(y), 31);   % lags 0..30
end

function testAcfWhiteNoiseDecays(tc)
% White noise has near-zero autocorrelation beyond lag 0.
rng(123);
x = randn(5000, 1);
y = acf(x, 10);
tc.verifyLessThan(max(abs(y(2:end))), 0.1);
end

function testAutocorrelationLagZeroNearOne(tc)
% autocorrelation normalizes by N*var(d), and var() uses the (N-1)
% denominator, so lag-0 equals (N-1)/N -- close to but not exactly 1.
rng(42);
n = 300;
x = randn(n, 1);
[ac, lag] = autocorrelation(x);
tc.verifyEqual(ac(1), (n - 1) / n, 'AbsTol', 1e-10);
tc.verifyEqual(lag(1), 0);
end

function testAutocorrelationLagVector(tc)
x = randn(100, 1);
[ac, lag] = autocorrelation(x);
tc.verifyEqual(numel(ac), numel(lag));
tc.verifyEqual(lag, 0:(numel(ac) - 1));
end

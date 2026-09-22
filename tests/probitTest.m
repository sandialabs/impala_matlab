function tests = probitTest
% Tests for probit / invprobit transforms (inverse relationship & known values).
tests = functiontests(localfunctions);
end

function setup(tc)
addProjectRootToPath(tc);
end

function testProbitAtHalfIsZero(tc)
% probit(0.5) = 0.
tc.verifyEqual(probit(0.5), 0, 'AbsTol', 1e-12);
end

function testInvprobitAtZeroIsHalf(tc)
tc.verifyEqual(invprobit(0), 0.5, 'AbsTol', 1e-12);
end

function testProbitInvprobitRoundTrip(tc)
% invprobit(probit(x)) == x for x in (0,1).
x = [0.01 0.1 0.25 0.5 0.75 0.9 0.99];
tc.verifyEqual(invprobit(probit(x)), x, 'AbsTol', 1e-10);
end

function testInvprobitProbitRoundTrip(tc)
% probit(invprobit(y)) == y for y on the real line.
y = [-3 -1 -0.5 0 0.5 1 3];
tc.verifyEqual(probit(invprobit(y)), y, 'AbsTol', 1e-10);
end

function testProbitMonotoneIncreasing(tc)
x = linspace(0.001, 0.999, 50);
p = probit(x);
tc.verifyTrue(all(diff(p) > 0));
end

function testInvprobitRangeIsUnitInterval(tc)
y = linspace(-6, 6, 100);
p = invprobit(y);
tc.verifyTrue(all(p > 0 & p < 1));
end

function testInvprobitSymmetry(tc)
% invprobit(-y) == 1 - invprobit(y).
y = [0.3 1 2.5];
tc.verifyEqual(invprobit(-y), 1 - invprobit(y), 'AbsTol', 1e-12);
end

function good = cf_bounds(x, bounds)
k = fieldnames(bounds);
good = x.(k{1}) < bounds.(k{1})(2);
for i = 1:numel(k)
    good = good .* (x.(k{i}) < bounds.(k{i})(2)) .* (x.(k{i}) > bounds.(k{i})(1));
end
good = logical(good);

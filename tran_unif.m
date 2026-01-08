function out = tran_unif(th, bounds, names)
out = struct();
tbounds = unnormalize(th,bounds)';
for i = 1:length(names)
    out.(names{i}) = tbounds(i,:);
end

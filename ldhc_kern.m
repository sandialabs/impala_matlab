function out = ldhc_kern(x, ~, ~)
% log kernel of a half-Cauchy prior on the error variance (used when
% s2_df == 0); the extra arguments match the signature of ldig_kern
out = -log(x+1);

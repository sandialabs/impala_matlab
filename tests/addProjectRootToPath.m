function fixture = addProjectRootToPath(tc)
%ADDPROJECTROOTTOPATH Put the repository root at the FRONT of the path.
%   fixture = addProjectRootToPath(tc) adds the directory that contains this
%   tests/ folder (i.e. the toolbox source root) to the top of the MATLAB
%   path for the duration of the test, and registers automatic teardown so
%   the original path is restored afterwards.
%
%   Prepending (rather than appending) guarantees the toolbox's own
%   functions win over any shadowing copies elsewhere on the path -- e.g.
%   MATLAB's built-in normalize()/acf(), or a sibling checkout such as
%   impala_matlab_snl.
%
%   Usage inside a function-based test file's setup:
%       function setup(tc)
%       addProjectRootToPath(tc);
%       end

thisDir  = fileparts(mfilename('fullpath'));   % .../tests
rootDir  = fileparts(thisDir);                 % repository root

oldPath = path();
addpath(rootDir);                              % prepend to the front
tc.addTeardown(@() path(oldPath));

fixture = rootDir;
end

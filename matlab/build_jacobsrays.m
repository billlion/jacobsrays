function build_jacobsrays
%BUILD_JACOBSRAYS Compile the shared C++ kernel using a configured MEX compiler.
% Run mex -setup C++ first if needed. Requires a C++11-capable compiler.
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
args = {'-R2018a', ['-I' fullfile(root,'include')], ...
    fullfile(here,'jacobsrays_mex.cpp'), '-outdir', here};
if ~ispc
    args = [args, {'CXXFLAGS=$CXXFLAGS -std=c++11 -O3'}];
end
mex(args{:});
end

function paths = jacobsrays(shape, starts, ends, spacing, origin)
%JACOBSRAYS Ordered intersections of finite segments with a Cartesian grid.
% P = JACOBSRAYS([nx ny nz], STARTS, ENDS, SPACING, ORIGIN)
% STARTS and ENDS are R-by-3 xyz arrays (R-by-2 for a 2D grid).
% SPACING defaults to ones; ORIGIN defaults to zeros and is the lower corner.
% P.offsets and P.indices are one-based int64 arrays. For ray r:
%   k = P.offsets(r):P.offsets(r+1)-1;
%   P.indices(k) gives voxels, in start-to-end order.
%   P.lengths(k) gives physical lengths, not fractions.
%   P.t_enter(k), P.t_exit(k) parameterise STARTS+t*(ENDS-STARTS).
% Linear indices address an [nx ny nz] MATLAB array via image(:).
% Parallel rays on internal planes belong to the higher-index cell;
% the lower outer face is included and the upper outer face excluded.
% Misses, tangencies and zero-length rays produce empty intervals.
if nargin<4 || isempty(spacing), spacing=ones(size(shape)); end
if nargin<5 || isempty(origin), origin=zeros(size(shape)); end
[paths.offsets, paths.indices, paths.lengths, paths.t_enter, paths.t_exit] = ...
    jacobsrays_mex(shape, starts, ends, spacing, origin);
paths.shape = shape(:).';
end

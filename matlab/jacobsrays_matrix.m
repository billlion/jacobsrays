function A = jacobsrays_matrix(paths)
%JACOBSRAYS_MATRIX Sparse scalar projector from ordered paths.
% A*image(:) projects; A'*data is its Euclidean adjoint.
% Sparse storage does not retain ray travel order: use paths for ODEs.
nr = numel(paths.offsets)-1;
nv = prod(paths.shape);
if nv>flintmax || nr>flintmax
    error('jacobsrays:tooLarge','Sparse double indices exceed exact integer range.');
end
% With no rays, diff(offsets) is empty and is not a valid repelem count.
% Also avoid constructing row indices when every ray misses the grid.
if isempty(paths.indices)
    A = sparse(nr, nv);
    return;
end
rays = repelem((1:nr)', double(diff(paths.offsets)));
A = sparse(rays, double(paths.indices), paths.lengths, nr, nv);
end

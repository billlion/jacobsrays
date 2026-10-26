function result = mod_jacobs_ray3d_multiple(shape, px, py, pz, qx, qy, qz, spacing, origin, output)
%MOD_JACOBS_RAY3D_MULTIPLE Compatibility entry point for the old wrapper.
% The old implementation used origin as LOWER CORNER despite its help text.
% Lengths are physical lengths. Coordinates may be row or column vectors.
% output='internal' gives ordered [ray, voxel, length] triples; the default
% is a sparse R-by-prod(shape) matrix. Use jacobsrays for interval parameters.
if nargin<8 || isempty(spacing), spacing=[1 1 1]; end
if nargin<9 || isempty(origin), origin=[0 0 0]; end
if nargin<10, output='sparse'; end
if isscalar(shape), shape=[shape shape shape]; end
if ~any(strcmp(output, {'sparse','internal'}))
    error('jacobsrays:output','output must be sparse or internal');
end
parts={px,py,pz,qx,qy,qz};
for k=1:6
    if ~(isvector(parts{k}) || isempty(parts{k})) || numel(parts{k})~=numel(px)
        error('jacobsrays:endpoints','Coordinate vectors must have the same length');
    end
end
paths=jacobsrays(shape,[px(:),py(:),pz(:)],[qx(:),qy(:),qz(:)],spacing,origin);
if strcmp(output,'internal')
    if prod(shape)>flintmax, error('jacobsrays:tooLarge','Indices exceed flintmax'); end
    if isempty(paths.indices)
        result=zeros(0,3);
        return;
    end
    rays=repelem((1:numel(px))',double(diff(paths.offsets)));
    % For one input ray, repelem treats its scalar input as a row vector.
    result=[rays(:),double(paths.indices),paths.lengths];
else
    result=jacobsrays_matrix(paths);
end
end

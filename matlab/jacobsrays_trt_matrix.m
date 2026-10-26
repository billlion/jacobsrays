function [T, paths] = jacobsrays_trt_matrix(shape, starts, ends, spacing, origin)
%JACOBSRAYS_TRT_MATRIX Full rank-two transverse ray transform, in 2D or 3D.
% T = JACOBSRAYS_TRT_MATRIX(SHAPE, STARTS, ENDS, SPACING, ORIGIN)
% integrates P*f*P, where P=eye(dim)-u*u' and u is the unit ray direction.
% SPACING and lower-corner ORIGIN have the same defaults as JACOBSRAYS.
%
% Store the tensor as F, an nv-by-nc array, one voxel per row. Components:
% 3D: [xx sqrt(2)*xy sqrt(2)*xz yy sqrt(2)*yz zz], nc=6;
% 2D: [xx sqrt(2)*xy yy], nc=3. Voxel order is MATLAB image(:).
% Y=reshape(T*F(:),nr,nc) contains the same Cartesian components per ray.
% The physical output tensor annihilates u. In 3D its six entries encode
% three independent components in the transverse plane; its trace is kept.
% Fadj=reshape(T'*Y(:),nv,nc) is the Frobenius/Euclidean discrete adjoint.
% Both F(:) and Y(:) are component-major (MATLAB column-major flattening).
%
% This small-problem example assembles the full sparse tensor matrix, which
% can have up to nc^2 times as many nonzeros as the scalar matrix. Python's
% TransverseRayTransform avoids that expansion. No MEX rebuild is required.
if nargin<4, spacing=[]; end
if nargin<5, origin=[]; end
paths=jacobsrays(shape,starts,ends,spacing,origin);
A=jacobsrays_matrix(paths);
[nr,nv]=size(A);
dim=numel(paths.shape); nc=dim*(dim+1)/2;
if nc*nv>flintmax || nc*nr>flintmax
    error('jacobsrays:tooLarge','Tensor sparse indices exceed exact integer range.');
end
if nnz(A)==0
    T=sparse(nc*nr,nc*nv);
    return;
end
B=zeros(dim,dim,nc); k=0;
for i=1:dim
    for j=i:dim
        k=k+1;
        if i==j, B(i,j,k)=1;
        else, B(i,j,k)=1/sqrt(2); B(j,i,k)=1/sqrt(2); end
    end
end
weights=zeros(nr,nc,nc);
for r=1:nr
    d=(ends(r,:)-starts(r,:))';
    length_ray=norm(d);
    if length_ray==0, continue; end
    u=d/length_ray; P=eye(dim)-u*u';
    for b=1:nc
        projected=P*B(:,:,b)*P;
        for a=1:nc
            weights(r,a,b)=sum(sum(B(:,:,a).*projected));
        end
    end
end
blocks=cell(nc,nc);
for a=1:nc
    for b=1:nc
        blocks{a,b}=spdiags(weights(:,a,b),0,nr,nr)*A;
    end
end
T=cell2mat(blocks);
end

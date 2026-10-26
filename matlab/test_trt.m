function test_trt
%TEST_TRT Physical-matrix oracle, Frobenius adjoint and edge cases.
rng(213);
for dim=[2 3]
    shape=[3 4 2]; shape=shape(1:dim);
    spacing=[.3 .7 1.1]; spacing=spacing(1:dim);
    origin=[-.2 -1.3 .5]; origin=origin(1:dim);
    nr=23; nv=prod(shape); nc=dim*(dim+1)/2;
    lo=origin-1; hi=origin+shape.*spacing+1;
    starts=lo+rand(nr,dim).*(hi-lo); ends=lo+rand(nr,dim).*(hi-lo);
    T=jacobsrays_trt_matrix(shape,starts,ends,spacing,origin);
    X=randn(nv,nc); Y=randn(nr,nc);
    expected=zeros(nr,nc);
    for r=1:nr
        d=ends(r,:)-starts(r,:); u=d'/norm(d); P=eye(dim)-u*u';
        total=zeros(dim);
        % Independent geometry: clip this ray against each voxel box.
        for v=1:nv
            rem_index=v-1; cell_index=zeros(1,dim);
            for j=1:dim
                cell_index(j)=mod(rem_index,shape(j));
                rem_index=floor(rem_index/shape(j));
            end
            low=origin+cell_index.*spacing; high=low+spacing;
            ts=[(low-starts(r,:))./d; (high-starts(r,:))./d];
            len=max(0,min(1,min(max(ts,[],1)))-max(0,max(min(ts,[],1))))*norm(d);
            total=total+len*P*unpack_tensor(X(v,:),dim)*P;
        end
        expected(r,:)=pack_tensor(total);
    end
    actual=reshape(T*X(:),nr,nc);
    assert(norm(actual-expected,'fro')<1e-11);
    assert(abs((T*X(:))'*Y(:)-X(:)'*(T'*Y(:)))<1e-11);
    reverse=jacobsrays_trt_matrix(shape,ends,starts,spacing,origin);
    assert(norm(T-reverse,'fro')<1e-11);
    empty=jacobsrays_trt_matrix(shape,zeros(0,dim),zeros(0,dim),spacing,origin);
    assert(issparse(empty) && isequal(size(empty),[0 nc*nv]));
    misses=jacobsrays_trt_matrix(shape,100*ones(2,dim),101*ones(2,dim),spacing,origin);
    assert(isequal(size(misses),[2*nc nc*nv]) && nnz(misses)==0);
    zero=jacobsrays_trt_matrix(shape,origin,origin,spacing,origin);
    assert(isequal(size(zero),[nc nc*nv]) && nnz(zero)==0);
end
% Exact chord length and full transverse trace, with nonzero off-diagonals.
F=[2 .3 -.7; .3 -1 .4; -.7 .4 3];
T=jacobsrays_trt_matrix([2 3 4],[-1 -1 -1],[2 2 2],[.5 1/3 .25]);
X=repmat(pack_tensor(F),24,1);
u=ones(3,1)/sqrt(3); P=eye(3)-u*u';
assert(norm(reshape(T*X(:),1,6)-sqrt(3)*pack_tensor(P*F*P))<1e-12);
X=repmat(pack_tensor(eye(3)),24,1);
assert(norm(reshape(T*X(:),1,6)-sqrt(3)*pack_tensor(P))<1e-12);
X=repmat(pack_tensor(u*u'),24,1);
assert(norm(T*X(:))<1e-12);
disp('jacobsrays TRT MATLAB checks passed');
end

function F=unpack_tensor(x,dim)
F=zeros(dim); k=0;
for i=1:dim
    for j=i:dim
        k=k+1; a=x(k);
        if i~=j, a=a/sqrt(2); end
        F(i,j)=a; F(j,i)=a;
    end
end
end

function x=pack_tensor(F)
dim=size(F,1); x=zeros(1,dim*(dim+1)/2); k=0;
for i=1:dim
    for j=i:dim
        k=k+1; x(k)=F(i,j);
        if i~=j, x(k)=sqrt(2)*x(k); end
    end
end
end

function test_jacobsrays
%TEST_JACOBSRAYS Run after build_jacobsrays; all assertions should pass.
p=jacobsrays([3 2],[-1 .5; 3 .5; .2 1.2],[4 .5; 0 .5; .8 1.2]);
assert(isequal(p.indices,int64([1;2;3;3;2;1;4])));
assert(max(abs(p.lengths-[1;1;1;1;1;1;.6]))<1e-14);
A=jacobsrays_matrix(p);
assert(isequal(size(A),[3 6]));
x=reshape(1:6,[3 2]); y=[2;3;5];
assert(abs((A*x(:))'*y-x(:)'*(A'*y))<1e-12);
p=jacobsrays([3 3 3],[-1 -1 -1],[4 4 4]);
assert(isequal(p.indices,int64([1;14;27])));
assert(max(abs(p.lengths-sqrt(3)))<1e-14);
% Regression: sparse sizing must not overwrite the final coefficient.
A=mod_jacobs_ray3d_multiple([2 2 1],0,1.5,.5,2,1.5,.5,[1 1 1],[0 0 0]);
assert(A(1,4)==1);
% Misses, empty batches and half-open outer boundary.
p=jacobsrays([2 2],[-1 2],[3 2]); assert(isempty(p.indices));
A=jacobsrays_matrix(p);
assert(issparse(A) && isequal(size(A),[1 4]) && nnz(A)==0);
p=jacobsrays([2 2],zeros(0,2),zeros(0,2));
assert(isequal(p.offsets,int64(1)));
A=jacobsrays_matrix(p);
assert(issparse(A) && isequal(size(A),[0 4]) && nnz(A)==0);
% More than one missing ray must retain all rows.
p=jacobsrays([2 2],[-1 2; -1 3],[3 2; 3 3]);
A=jacobsrays_matrix(p);
assert(issparse(A) && isequal(size(A),[2 4]) && nnz(A)==0);
% Mixed misses and hits must retain their original row positions.
p=jacobsrays([2 2],[-1 2; -1 .5; -1 3],[3 2; 3 .5; 3 3]);
assert(isequal(full(jacobsrays_matrix(p)),[0 0 0 0; 1 1 0 0; 0 0 0 0]));
% The old call signature must support empty batches in both output modes.
empty=zeros(0,1);
A=mod_jacobs_ray3d_multiple([2 2 1],empty,empty,empty,empty,empty,empty);
assert(issparse(A) && isequal(size(A),[0 4]) && nnz(A)==0);
T=mod_jacobs_ray3d_multiple([2 2 1],empty,empty,empty,empty,empty,empty, ...
    [1 1 1],[0 0 0],'internal');
assert(isa(T,'double') && isequal(size(T),[0 3]));
T=mod_jacobs_ray3d_multiple([2 2 1],-1,3,.5,3,3,.5, ...
    [1 1 1],[0 0 0],'internal');
assert(isequal(size(T),[0 3]));
% Single-ray triples are N-by-3, even though repelem of a scalar is a row.
T=mod_jacobs_ray3d_multiple([2 2 1],0,1.5,.5,2,1.5,.5, ...
    [1 1 1],[0 0 0],'internal');
assert(isequal(T,[1 3 1; 1 4 1]));
% Invalid MEX input must fail safely.
failed=false;
try, jacobsrays([2.5 2],[0 0],[1 1]); catch, failed=true; end
assert(failed);
disp('jacobsrays MATLAB checks passed');
end

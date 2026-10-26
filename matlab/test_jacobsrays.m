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
p=jacobsrays([2 2],zeros(0,2),zeros(0,2));
assert(isequal(size(jacobsrays_matrix(p)),[0 4]));
% Invalid MEX input must fail safely.
failed=false;
try, jacobsrays([2.5 2],[0 0],[1 1]); catch, failed=true; end
assert(failed);
disp('jacobsrays MATLAB checks passed');
end

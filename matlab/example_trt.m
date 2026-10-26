function example_trt
%EXAMPLE_TRT Full 3D rank-two TRT on a constant symmetric tensor.
% Run build_jacobsrays once, then example_trt. See docs/trt.md.
shape=[8 7 6]; spacing=1./shape;
starts=[-1 .5 .5; .5 -1 .5; .5 .5 -1; -1 -1 -1];
ends=[2 .5 .5; .5 2 .5; .5 .5 2; 2 2 2];
T=jacobsrays_trt_matrix(shape,starts,ends,spacing,[0 0 0]);
F=[2 .3 -.7; .3 -1 .4; -.7 .4 3];
packed=[F(1,1),sqrt(2)*F(1,2),sqrt(2)*F(1,3),F(2,2),sqrt(2)*F(2,3),F(3,3)];
field=repmat(packed,prod(shape),1);
Y=reshape(T*field(:),size(starts,1),6);
disp('TRT: rows are rays; columns are xx,sqrt(2)xy,sqrt(2)xz,yy,sqrt(2)yz,zz');
disp(Y);
% Read the three independent detector entries for the diagonal ray.
u=ones(3,1)/sqrt(3); a=[1;-1;0]/sqrt(2); b=cross(u,a); Q=[a b];
y=Y(4,:);
G=[y(1) y(2)/sqrt(2) y(3)/sqrt(2); y(2)/sqrt(2) y(4) y(5)/sqrt(2); ...
   y(3)/sqrt(2) y(5)/sqrt(2) y(6)];
H=Q'*G*Q;
disp('Detector components: aa,sqrt(2)ab,bb');
disp([H(1,1) sqrt(2)*H(1,2) H(2,2)]);
assert(norm(H-sqrt(3)*Q'*F*Q,'fro')<1e-12);
% T' acts on arbitrary packed ray tensors, including noisy/non-transverse ones.
rng(4); x=randn(size(T,2),1); y=randn(size(T,1),1);
fprintf('Adjoint error: %.3g\n',abs((T*x)'*y-x'*(T'*y)));
end

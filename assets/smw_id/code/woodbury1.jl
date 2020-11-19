# This file was generated, do not modify it. # hide
using WoodburyMatrices, LinearAlgebra
n = 5000;
k = 10;
A = randn(n,n) + I
U = randn(n,k);
C = randn(k,k)
V = randn(n,k);
Woodbury_struct = Woodbury(A, U, C, V')
Woodbury_dense = A + U*C*V'
@time Woodbury_struct\ones(n)
@time Woodbury_dense\ones(n)
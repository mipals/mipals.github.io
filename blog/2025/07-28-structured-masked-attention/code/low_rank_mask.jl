using LinearAlgebra
using Random
using Test
Random.seed!(42)
n, p = 10, 2
U, V = randn(n,p), randn(n,p)
B = rand(n,n)
M = B .* (U*V')
x = randn(n)
@test M*x ≈ sum(i -> Diagonal(U[:,i])*(B*(Diagonal(V[:,i])) * x), 1:p)
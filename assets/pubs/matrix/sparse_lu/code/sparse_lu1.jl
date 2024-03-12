# This file was generated, do not modify it. # hide
using LinearAlgebra
using SparseArrays
n = 5
u = rand(n)
d = n./rand(n)
D = sparse(1:n,1:n,d)

Al = [1 u'; u D]
Ar = [D u; u' 1]

# Checking if Al and Ar are positive definite
u'*(D\u)
println(u'*(D\u)) #hide
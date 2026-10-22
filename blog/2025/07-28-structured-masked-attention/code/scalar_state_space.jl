using SymSemiseparableMatrices
using LinearAlgebra
using Random
using Test
Random.seed!(42)

T = 10          # Sequence length
n = 6           # State size

# Here we treat the matrices in terms of their states and not sequence lengths
a_blks = rand(T-1)
a = Bidiagonal(ones(T), -a_blks, :L)
A = kron(I(n),a)        # Kronecker product with identity

# The blocks are now size equal to the sequence and a block for each state!
B_blks = [rand(T) for _ in 1:n]
C_blks = [rand(T) for _ in 1:n]  
# Collecting the blocks into the B and C
B = vcat(Diagonal.(B_blks)...)
C = vcat(Diagonal.(C_blks)...) 

# We want to see if the full matrix M = C'*(A\B) can be written as
# structured masked attention, i.e.,
# M = (CB')\circ a^{-1}. For this we start by computing a^{-1}
ai = inv(a) # We ignore here that inv(a) is semiseparable
@test C'*(A\B) ≈ sum(i-> Diagonal(C_blks[i])*ai*Diagonal(B_blks[i]), 1:n)
@test C'*(A\B) ≈ sum(i->(C_blks[i]*B_blks[i]') .* ai, 1:n)

# We can collect the terms and write it as Structured Masked Attention!
Cn = hcat(C_blks...)
Bn = hcat(B_blks...)
@test C'*(A\B) ≈ (Cn*Bn') .* ai

# We can apply the semiseparable structure of the inverse when multiplying!
ai_blks = [prod(a_blks[1:i-1],init=1.0) for i in 1:T]
u = vcat(ai_blks...)
v = vcat(inv.(ai_blks)...)
# "SymSemiseparableCholesky" represents the matrix A^{-1} = tril(UV') 
ais = SymSemiseparableCholesky(u',v')
# Efficient products using the structure of "ais" and diagonal B and C
x = randn(T)
@test C'*(A\(B*x)) ≈ sum(i-> C_blks[i].*(ais*(B_blks[i].*x)), 1:n)
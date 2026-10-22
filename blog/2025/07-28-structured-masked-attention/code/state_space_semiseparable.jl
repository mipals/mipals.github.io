using SymSemiseparableMatrices
using BlockBandedMatrices
using LinearAlgebra
using Random
using Test
Random.seed!(42)

T = 10          # Sequence length
n = 6           # State size
input_dim = 1   # Dimension of forcing term

A_blks = [rand()*I(n) for _ in 1:T-1]       # Diagonal dynamics in Mamba 2
D_blks = [I(n)/1.0 for i in 1:T]            # Diagonal blocks are identity
C_blks = [rand(n,1) for _ in 1:T]           # Measurements are scalars
B_blks = [rand(n,input_dim) for _ in 1:T]   # Inputs are scalars
# Defining zeros blocks for the matrices
A_zero_blks = [zeros(n,n) for b in A_blks]
C_zero_blks = [zeros(n,1) for _ in 1:T-1]
B_zero_blks = [zeros(n,input_dim) for _ in 1:T-1]   
# Defining the block matrices
A = BlockTridiagonal(-A_blks,D_blks,A_zero_blks)
C = BlockTridiagonal(C_zero_blks,C_blks,C_zero_blks)
B = BlockTridiagonal(B_zero_blks,B_blks,B_zero_blks)

# Computing states by iterating forward
x_blks = [randn(input_dim) for _ in 0:T-1]  # input data
h_blks = [B_blks[1]*x_blks[1]]              # initial hidden state
for i in 2:T
    push!(h_blks, A_blks[i-1]*h_blks[i-1] + B_blks[i]*x_blks[i])
end
y_blks = [C'*h for (C,h) in zip(C_blks,h_blks)] 

# Computing states using semiseparable matrices
Ai_blks = [prod(reverse(A_blks[1:i-1]),init=1.0*I(n)) for i in 1:T]
U = vcat(Ai_blks...)
V = vcat(inv.(transpose.(Ai_blks))...)
# "SymSemiseparableCholesky" represents the matrix A^{-1} = tril(UV') 
Ai = SymSemiseparableCholesky(U', V')
x = vcat(x_blks...) # Collecting input data

@test Ai*(B*x) ≈ vcat(h_blks...)        # Checking hidden states
@test C'*(Ai*(B*x)) ≈ vcat(y_blks...)   # Checking measurement
## Structured Masked Attention
using LinearAlgebra, Test
n, p = 10, 2
U, V = randn(n,p), randn(n,p)
B = rand(n,n)
M = B .* (U*V')
x = randn(n)
@test M*x ≈ sum(i -> Diagonal(U[:,i])*(B*(Diagonal(V[:,i])) * x),1:p)


## State-space models as structured matrices
using LinearAlgebra, BlockBandedMatrices, Test, SymSemiseparableMatrices

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
Ai_blks = [prod(A_blks[1:i-1],init=1.0*I(n)) for i in 1:T]
U = vcat(Ai_blks...)
V = vcat(inv.(Ai_blks)...)
# "SymSemiseparableCholesky" represents the matrix A^{-1} = tril(UV') 
Ai = SymSemiseparableCholesky(U',V')
x = vcat(x_blks...) # Collecting input data

@test Ai*(B*x) ≈ vcat(h_blks...)        # Checking hidden states
@test C'*(Ai*(B*x)) ≈ vcat(y_blks...)   # Checking measurement





##### State-space models as SMAs
using LinearAlgebra, Test, SymSemiseparableMatrices

T = 10          # Sequence length
n = 6           # State size

# Here we treat the matrices in terms of their states and not sequence lengths
a_blks = rand(T-1)
a = Bidiagonal(ones(T),-a_blks,:L)
A = kron(I(n),a)        # Kronecker product with identity

# The blocks are now size equal to the sequence and a block for each state!
B_blks = [rand(T) for _ in 1:n]
C_blks = [rand(T) for _ in 1:n]  
# Collecting the blocks into the B and C
B = vcat(Diagonal.(B_blks)...)
C = vcat(Diagonal.(C_blks)...) 

# We want to see if the full matrix M = C'*(A\B) can be written as
# structured masked attention ie. 
# M = (CB')\circ a^{-1}. For this we start by computing a^{-1}
ai = inv(a) # We ignore here that inv(a) is semiseparable
@test C'*(A\B) ≈ sum(i-> Diagonal(C_blks[i])*ai*Diagonal(B_blks[i]),1:n)
@test C'*(A\B) ≈ sum(i->(C_blks[i]*B_blks[i]') .* ai , 1:n)

# We can collect the terms and write is as Structured Masked Attention!
Cn = hcat(C_blks...)
Bn = hcat(B_blks...)
@test C'*(A\B) ≈ (Cn*Bn').*ai

# We can apply the semiseparable structure of the inverse when multiplying!
ai_blks = [prod(a_blks[1:i-1],init=1.0) for i in 1:T]
u = vcat(ai_blks...)
v = vcat(inv.(ai_blks)...)
# "SymSemiseparableCholesky" represents the matrix A^{-1} = tril(UV') 
ais = SymSemiseparableCholesky(u',v')
# Efficient products using the structure of "ais" and diagonal B and C
x = randn(T)
@test C'*(A\(B*x)) ≈ sum(i-> C_blks[i].*(ais*(B_blks[i].*x)),1:n)
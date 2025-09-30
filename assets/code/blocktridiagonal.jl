using LinearAlgebra, FillArrays, LinearAlgebra, BlockBandedMatrices, Test

T = 5 # Sequence length
n = 3 # blk sizes

A_blks = [i == 0 ? zeros(n,n) : exp(rand(n,n)) for i in 1:T] 
D_blks = [Matrix(1.0*I(n)) for i in 1:T]
A_blks_transposed = [A_blk' for A_blk in A_blks]
zero_blks = [zeros(n,n) for i in 1:T]

K = BlockTridiagonal(-A_blks[2:end], D_blks, -A_blks_transposed[2:end])
Δ_blks = copy(D_blks)
Σ_blks = copy(D_blks)
for (i,j) in zip(2:length(D_blks),length(D_blks):-1:2)
    Δ_blks[i]   = D_blks[i] - A_blks[i]*(Δ_blks[i-1]\A_blks[i]')
    Σ_blks[j-1] = D_blks[j] - A_blks[j]'*(Σ_blks[j]\A_blks[j])
end

L = BlockTridiagonal(-A_blks[2:end], zero_blks, zero_blks[1:end-1])
Δ = BlockTridiagonal(zero_blks[1:end-1], Δ_blks, zero_blks[1:end-1])
Σ = BlockTridiagonal(zero_blks[1:end-1], Σ_blks, zero_blks[1:end-1])

@test (Δ + L)*(Δ\(Δ + L')) ≈ K
@test (Σ + L')*(Σ\(Σ + L)) ≈ K

# Creating the semiseperable form
V_blk = [Matrix(1.0*I(n)) for _ in 1:T]
U_blk = [Matrix(1.0*I(n)) for _ in 1:T]
U_blk[1] = inv(Σ_blks[1])
for i in 2:T
    U_blk[i] = Σ_blks[i]'\A_blks[i]*U_blk[i-1]
    V_blk[i] = A_blks[i]'\(Δ_blks[i-1]*V_blk[i-1])
end
U = vcat(U_blk...)
V = vcat(V_blk...)
@test tril(U*V') ≈ tril(inv(K))
@test triu(V*U') ≈ triu(inv(K))

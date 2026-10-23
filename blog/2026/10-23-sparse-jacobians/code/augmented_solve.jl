# Solve J Δ = b with the sparse augmented matrix A = [L -Mᵀ; P 0] instead
# of J = P L⁻¹ Mᵀ. Its first block row gives z = L⁻¹ Mᵀ Δ, so the second
# one, P z = b, is J Δ = b.
using LinearAlgebra, SparseArrays

function augmented_solve(L, Mt, P, b)
    d, n = size(Mt)
    A = [L -Mt; P spzeros(size(P, 1), n)]
    return (A \ [zeros(d); b])[d+1:end]
end

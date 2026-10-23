using SparseSparse # Required for \ with a sparse right-hand side
include("tape.jl")

# u''(xᵢ) by central differences, with u = 0 at both ends
function cdiff2(x, i, h)
    left = i > 1 ? x[i-1] : 0.0
    right = i < length(x) ? x[i+1] : 0.0
    return (left - 2x[i] + right) / h^2
end

# Bratu, u'' + λ exp(u) = 0: each output reads three inputs
function bratu(x; λ=1.0)
    h = 1 / (length(x) + 1)
    return [cdiff2(x, i, h) + λ * exp(x[i]) for i in eachindex(x)]
end

# Nonlocal Ohmic heating, u'' + λ exp(-u) / (∫₀¹ exp(-u) dx)² = 0, of the
# next section: every output reads the sum s, so every input
function heating(x; λ=2.0)
    h = 1 / (length(x) + 1)
    s = sum(exp(-xj) for xj in x) # h * s is the integral
    scaling = λ / (h * s)^2
    return [cdiff2(x, i, h) + scaling * exp(-x[i]) for i in eachindex(x)]
end

x = 0.1 .* rand(1000)
_, L, Mt, P = trace(bratu, x) # nnz(L) = 13000, nnz(M) = 3998
J = P * (L \ Mt) # tridiagonal: nnz(J) = 2998
_, L, Mt, P = trace(heating, x) # nnz(L) = 27997, nnz(M) = 4998
J = P * (L \ Mt) # dense: nnz(J) = 1000000

# Discretized differential equations as functions of their unknowns:
# two boundary value problems on (0, 1) with u = 0 at both ends, as
# residuals F(x) = 0 on the n interior points.

# u''(xᵢ) by central differences
function cdiff2(x, i, h)
    left = i > 1 ? x[i-1] : 0.0
    right = i < length(x) ? x[i+1] : 0.0
    return (left - 2x[i] + right) / h^2
end

# Bratu, u'' + λ exp(u) = 0: each output reads three inputs
function bratu(x; λ=1.0)
    h = 1 / (length(x) + 1)
    return [cdiff2(x, i, h) + λ * exp(x[i])
            for i in eachindex(x)]
end

# Nonlocal Ohmic heating, u'' + λ exp(-u) / (∫₀¹ exp(-u) dx)² = 0:
# every output reads the sum s, so every output reads every input
function heating(x; λ=2.0)
    h = 1 / (length(x) + 1)
    s = exp(-x[1]) # h * s is the integral
    for j in 2:length(x)
        s += exp(-x[j])
    end
    return [cdiff2(x, i, h) + λ * exp(-x[i]) / (h * s)^2
            for i in eachindex(x)]
end

# The heating residual lifted with the partial sums sₖ = sₖ₋₁ + exp(-xₖ)
# as unknowns, z = [x; s], so that every equation reads only a few unknowns
function heating_lifted_partial_sums(z; λ=2.0)
    n = length(z) ÷ 2
    x, s = view(z, 1:n), view(z, n+1:2n)
    h = 1 / (n + 1)
    F = [cdiff2(x, i, h) + λ * exp(-x[i]) / (h * s[n])^2
         for i in 1:n]
    S = [s[k] - (k > 1 ? s[k-1] : 0.0) - exp(-x[k]) for k in 1:n]
    return [F; S]
end

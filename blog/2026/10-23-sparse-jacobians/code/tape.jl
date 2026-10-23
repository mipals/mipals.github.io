# A scalar tape. Every operation on a Tracked number is one node Φᵢ
# of the forward relation u = Φ(u, x), and it stores its partial
# derivatives, the entries L[i, j] = -∂Φᵢ/∂uⱼ and Mᵀ[i, k] = ∂Φᵢ/∂xₖ.
using LinearAlgebra, SparseArrays

# The tape: rows, columns and values of the entries of L - I and Mᵀ,
# and the number of nodes
const L_entries = (Int[], Int[], Float64[])
const Mt_entries = (Int[], Int[], Float64[])
const nodes = Ref(0)

# index > 0 is the node uᵢ, index < 0 is the input x₋ᵢ
struct Tracked
    value::Float64
    index::Int
end

# Append the node with the given value and (argument => partial) pairs
function record!(value, args...)
    i = nodes[] += 1
    for (a, ∂a) in args
        rows, cols, vals = a.index > 0 ? L_entries : Mt_entries
        push!(rows, i)
        push!(cols, abs(a.index))
        push!(vals, a.index > 0 ? -∂a : ∂a)
    end
    return Tracked(value, i)
end

Base.:+(a::Tracked, b::Tracked) = record!(a.value + b.value, a => 1.0, b => 1.0)
Base.:-(a::Tracked, b::Tracked) = record!(a.value - b.value, a => 1.0, b => -1.0)
Base.:*(a::Tracked, b::Tracked) = record!(a.value * b.value, a => b.value, b => a.value)
Base.:/(a::Tracked, b::Tracked) =
    record!(a.value / b.value, a => 1 / b.value, b => -a.value / b.value^2)
Base.:+(a::Tracked, c::Real) = record!(a.value + c, a => 1.0)
Base.:+(c::Real, a::Tracked) = a + c
Base.:-(a::Tracked, c::Real) = a + (-c)
Base.:-(c::Real, a::Tracked) = record!(c - a.value, a => -1.0)
Base.:-(a::Tracked) = record!(-a.value, a => -1.0)
Base.:*(a::Tracked, c::Real) = record!(a.value * c, a => Float64(c))
Base.:*(c::Real, a::Tracked) = a * c
Base.:/(a::Tracked, c::Real) = a * inv(c)
Base.:/(c::Real, a::Tracked) = record!(c / a.value, a => -c / a.value^2)
Base.exp(a::Tracked) = (v = exp(a.value); record!(v, a => v))
Base.sin(a::Tracked) = record!(sin(a.value), a => cos(a.value))
Base.:^(a::Tracked, p::Integer) = record!(a.value^p, a => p * a.value^(p - 1))

# Run f on tracked inputs and return f(x) with the sparse matrices
# L = I - ∂Φ/∂u, Mᵀ = ∂Φ/∂x and the selection P of the outputs, so
# that the Jacobian of f at x is P L⁻¹ Mᵀ.
function trace(f, x)
    foreach(empty!, (L_entries..., Mt_entries...))
    nodes[] = 0
    y = f([Tracked(xk, -k) for (k, xk) in enumerate(x)])
    d, n, m = nodes[], length(x), length(y)
    L = I + sparse(L_entries..., d, d) # sums the entries of x * x
    Mt = sparse(Mt_entries..., d, n)
    P = sparse(1:m, [yi.index for yi in y], ones(m), m, d)
    return [yi.value for yi in y], L, Mt, P
end

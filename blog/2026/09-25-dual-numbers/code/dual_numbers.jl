# A small, deliberately incomplete forward-mode AD implementation.
struct Dual{T<:Real}
    value::T
    deriv::T
end
# Promote mixed inputs such as Dual(2, 1.0) to a common type
Dual(value::Real, deriv::Real) = Dual(promote(value, deriv)...)

# Product rule: (f * g)' = f' * g + f * g'
Base.:*(f::Dual, g::Dual) = Dual(
    f.value * g.value,
    f.deriv * g.value + f.value * g.deriv,
)
Base.:*(f::Dual, g::Real) = f * Dual(g, zero(g))
Base.:*(f::Real, g::Dual) = Dual(f, zero(f)) * g

# Quotient rule: (f / g)' = (f' * g - f * g') / g^2
Base.:/(f::Dual, g::Dual) = Dual(
    f.value / g.value,
    (f.deriv * g.value - f.value * g.deriv) / g.value^2,
)

# Composition rule: (f ∘ g)' = (f' ∘ g) * g'
Base.sin(f::Dual) = Dual(sin(f.value), cos(f.value) * f.deriv)
Base.exp(f::Dual) = Dual(exp(f.value), exp(f.value) * f.deriv)

# Addition rules: (f + g)' = f' + g', (f + c)' = f', (c + f)' = f'
Base.:+(f::Dual, g::Dual) = Dual(f.value + g.value, f.deriv + g.deriv)
Base.:+(f::Dual, c::Real) = f + Dual(c, zero(c))
Base.:+(c::Real, g::Dual) = Dual(c, zero(c)) + g

# Subtraction rules: (-f)' = -f', (f - g)' = f' - g'
Base.:-(f::Dual) = Dual(-f.value, -f.deriv)
Base.:-(f::Dual, g::Dual) = f + (-g)

# The derivative is the epsilon coefficient of f(x + 1*epsilon)
derivative(f, x) = f(Dual(x, one(x))).deriv

# Dual is not a Number, so Base's generic ^ does not apply to it and
# f_scalar(x) = sin(x^2) + x would throw a MethodError
f_scalar(x) = sin(x * x) + x
@assert isapprox(derivative(f_scalar, 2.0), 4cos(4.0) + 1)

println("f'(2) = ", derivative(f_scalar, 2.0))
@isdefined(Dual) || include("dual_numbers.jl")

function jvp(f, x, v)
    @assert length(x) == length(v)
    seeded = [Dual(xi, vi) for (xi, vi) in zip(x, v)]
    result = f(seeded)
    return map(z -> z.value, result), map(z -> z.deriv, result)
end

f_vector(x) = (x[1] * x[2] + sin(x[1]), exp(x[2]))
x, v = [2.0, 3.0], [1.0, -1.0]
value, deriv = jvp(f_vector, x, v)
@assert all(isapprox.(value, (6 + sin(2.0), exp(3.0))))
@assert all(isapprox.(deriv, (1 + cos(2.0), -exp(3.0))))

println("f($x) = ", value)
println("Jf($x)$v = ", deriv)
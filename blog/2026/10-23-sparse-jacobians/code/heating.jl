# Newton's method for nonlocal Ohmic heating,
# u'' + λ exp(-u) / (∫₀¹ exp(-u) dx)² = 0, whose Jacobian is dense,
# with the sparse augmented matrix A = [L -Mᵀ; P 0] instead of J.
include("tape.jl")
include("augmented_solve.jl")
include("problems.jl")

# A Newton step: trace the program at x and solve with A
function newton_step(f, x)
    y, L, Mt, P = trace(f, x)
    return augmented_solve(L, Mt, P, y)
end

function newton(f, x; tol=1e-9, maxiter=20)
    for k in 0:maxiter
        r = norm(f(x))
        println("  iteration $k: ‖F(x)‖ = $r")
        r < tol * (1 + norm(x)) && break
        x -= newton_step(f, x)
    end
    return x
end

println("Newton with the augmented system, n = 1000")
u = newton(heating, zeros(1000))
println("  maximum temperature $(maximum(u))")

# Seconds, the best of three runs after compilation
best(f) = (f(); minimum(@elapsed(f()) for _ in 1:3))
for n in (500, 1000, 2000, 4000, 8000)
    x = 0.1 .* rand(n)
    _, L, Mt, P = trace(heating, x)
    println("n = $n: nnz(A) = $(nnz(L) + nnz(Mt) + nnz(P)), ",
        "step $(best(() -> newton_step(heating, x))) s")
end

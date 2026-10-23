# Estimating λ in the heating problem from noisy temperatures with an
# NLP solver (Uno), with the integral inside every equation and lifted,
# with the sum s as an extra variable and s = ∑ⱼ exp(-xⱼ) as an extra
# equation.
include("tape.jl")
include("augmented_solve.jl")
include("problems.jl")
using JuMP, UnoSolver, Random

# Temperatures for λ = 2 by ten Newton steps with the augmented system,
# with noise of standard deviation σ
function measurements(n; λ=2.0, σ=0.01, rng=Xoshiro(1))
    x = zeros(n)
    for _ in 1:10
        y, L, Mt, P = trace(x -> heating(x; λ), x)
        x -= augmented_solve(L, Mt, P, y)
    end
    return x .+ σ .* randn(rng, n)
end

function estimation_model(data; lifted)
    n = length(data)
    model = Model(UnoSolver.Optimizer)
    set_silent(model)
    @variable(model, x[1:n], start = 0.0)
    @variable(model, λ >= 0, start = 1.0)
    if lifted
        @variable(model, s, start = n) # sum(exp(-x[j])) = n at the start x = 0
        @constraint(model, s == sum(exp(-x[j]) for j in 1:n))
    else
        s = @expression(model, sum(exp(-x[j]) for j in 1:n))
    end
    left(i) = i > 1 ? x[i-1] : 0.0
    right(i) = i < n ? x[i+1] : 0.0
    # The residual times h², which keeps the equations of order one
    @constraint(model, [i in 1:n],
        left(i) - 2x[i] + right(i) + λ * exp(-x[i]) / s^2 == 0)
    @objective(model, Min, sum((x[i] - data[i])^2 for i in 1:n))
    return model
end

# The best time of the given number of runs, each with a new model
function estimate(data; lifted, runs)
    results = map(1:runs) do _
        model = estimation_model(data; lifted)
        bytes = @allocated (t = @elapsed optimize!(model))
        (; λ=value(model[:λ]), t, bytes)
    end
    return argmin(r -> r.t, results)
end
report(name, r) =
    println("  $name: λ = $(r.λ), $(r.t) s, $(r.bytes / 2^20) MiB")

# Compile on a small problem, then time each size. The inline
# formulation allocates memory of order n³, 4 GB for n = 200, so it
# runs once; the lifted one is timed as the best of three runs. Only
# when run as a script, so that figures.jl can include the functions.
if abspath(PROGRAM_FILE) == @__FILE__
    estimate(measurements(10); lifted=true, runs=1)
    estimate(measurements(10); lifted=false, runs=1)
    for n in (100, 200, 1000, 2000, 4000)
        data = measurements(n)
        println("n = $n")
        report("lifted", estimate(data; lifted=true, runs=3))
        n <= 200 &&
            report("inline integral", estimate(data; lifted=false, runs=1))
    end
end

# Newton steps for the heating problem with the sparse backend of
# DifferentiationInterface.jl, on the residual lifted with every partial
# sum of the integral as an unknown.
include("problems.jl")
using DifferentiationInterface, ForwardDiff, SparseConnectivityTracer
using SparseMatrixColorings, LinearAlgebra, SparseArrays

backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
G = heating_lifted_partial_sums
lifted_step(z, prep) = jacobian(G, prep, backend, z) \ G(z)
lift(x) = [x; cumsum(exp.(-x))] # the partial sums that belong to x

# Newton from x = 0
let z = lift(zeros(1000))
    prep = prepare_jacobian(G, backend, z)
    for k in 0:20
        println("  iteration $k: ‖G(z)‖ = $(norm(G(z)))")
        norm(G(z)) < 1e-9 * (1 + norm(z)) && break
        z -= lifted_step(z, prep)
    end
end

# Seconds, the best of three runs after compilation
best(f) = (f(); minimum(@elapsed(f()) for _ in 1:3))
prepare_jacobian(G, backend, lift(rand(10))) # compile
for n in (500, 1000, 2000, 4000, 8000)
    z, z_new = lift(0.1 .* rand(n)), lift(0.1 .* rand(n))
    # The preparation is timed once
    t = @elapsed (prep = prepare_jacobian(G, backend, z))
    println("n = $n: $(ncolors(prep)) colors, ",
        "nnz $(nnz(jacobian(G, prep, backend, z))), prepare $t s, ",
        "step $(best(() -> lifted_step(z_new, prep))) s")
end

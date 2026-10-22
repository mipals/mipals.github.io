using Convex, Clarabel, LinearAlgebra

# The sharp operator of a norm from its definition,
#   g♯ = -argmin_h ⟨g, h⟩ + ½‖h‖²,
# solved as a small conic program.
function sharp(G, norm)
    H = Variable(size(G)...)
    t = Variable()                 # t ≥ ‖H‖, so ½t² is the penalty
    J = dot(vec(G), vec(H)) + square(t) / 2
    problem = minimize(J, [norm(H) <= t])
    solver() = Clarabel.Optimizer(chordal_decomposition_enable = false)
    solve!(problem, solver; silent = true)
    return -evaluate(H)
end

G = [3.0 -1.0; 4.0 1.0]            # the derivative, as a matrix
a = [1.0 2.0; 4.0 8.0]             # weights of a diagonal inner product
W = Diagonal(sqrt.(vec(a)))        # so that ‖H‖² = Σ aᵢⱼHᵢⱼ² = ‖W vec(H)‖²
F = svd(G)

# Each norm, with the closed form of g♯ from the post
norms = [
    "Frobenius" => (H -> norm(vec(H), 2), G),
    "Diagonal"  => (H -> norm(W * vec(H), 2), G ./ a),
    "Max-norm"  => (H -> norm(vec(H), Inf), norm(G, 1) * sign.(G)),
    "Spectral"  => (H -> opnorm(H, 2), sum(F.S) * F.U * F.Vt),
]

for (name, (nrm, closed_form)) in norms
    println(name)
    display(round.(sharp(G, nrm); digits = 4))
    display(closed_form)
end

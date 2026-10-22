using LinearAlgebra, Random, SparseArrays
Random.seed!(42)

# A linear map stored together with its adjoint
struct Operator
    operator
    adjoint
    sym::String
end
Base.adjoint(A::Operator) = Operator(A.adjoint, A.operator, "$(A.sym)′")
Base.:*(A::Operator, X) = A.operator(X)
Base.show(io::IO, A::Operator) = print(io, A.sym)
Base.zero(::Type{Operator}) = Operator(zero, zero, "𝒪") # For printing
const ℐ = Operator(identity, identity, "ℐ")

# Adjoints of sparse matrices whose entries are operators.
const AdjointOperators = Adjoint{<:Any, <:SparseMatrixCSC{Operator}}

# Back substitution with ℒ*: Λ_j = g_j - ∑_{i>j} ℒ_ij* Λ_i.
function Base.:\(A::AdjointOperators, g::Vector)
    ℒ = parent(A) # Get the non-adjoint, but apply adjoint later
    rows, blocks = rowvals(ℒ), nonzeros(ℒ) # Column j: blocks[r], rows[r]
    Λ = copy(g)
    for j in length(g):-1:1, r in nzrange(ℒ, j)
        i = rows[r]
        if i > j
            Λ[j] -= blocks[r]' * Λ[i]
        end
    end
    return Λ
end

# ℳ = (ℳ′)*: entry k sums the contributions of the layers that use θ_k.
function Base.:*(A::AdjointOperators, Λ::Vector)
    ℳ′ = parent(A) # Get the non-adjoint, but apply adjoint later
    rows, blocks = rowvals(ℳ′), nonzeros(ℳ′)
    return [sum(blocks[r]' * Λ[rows[r]] for r in nzrange(ℳ′, k))
            for k in axes(ℳ′, 2)]
end

# Defining the activation function and its derivative
σ(x) = exp(-x)
σ′(x) = -exp(-x)

# Layer i reads the state U_j through the weight W_ij = θ_k for every
# j => k in W[i], and adds the bias b_i = θ_b[i]. Layers 3 and 4 share
# both their weight θ₃ and their bias θ₁₁.
d = [5, 4, 4, 4, 4, 3, 1] # Feature sizes of U₀, …, U₆
W = [[0 => 1], [1 => 2], [2 => 3, 1 => 6], [3 => 3], [4 => 4, 2 => 7],
     [5 => 5, 3 => 8]]
b = [9, 10, 11, 11, 12, 13]
N, K = 6, 13 # Layers, parameter blocks

function network(θ)
    U, Δ = Matrix{Float64}[], Matrix{Float64}[]
    state(j) = j == 0 ? U₀ : U[j]
    for i in 1:N
        Z = sum(θ[k] * state(j) for (j, k) in W[i]) .+ θ[b[i]]
        push!(U, σ.(Z))
        push!(Δ, σ′.(Z))
    end
    return U, Δ
end

# The sparse matrices ℒ = I - D_U Φ and ℳ′ = D_θ Φ, block by block.
function operators(θ, U, Δ)
    state(j) = j == 0 ? U₀ : U[j]
    ℒ, ℳ′ = spzeros(Operator, N, N), spzeros(Operator, N, K)
    for i in 1:N
        ℒ[i, i] = ℐ
        for (j, k) in W[i]
            Wij, Uj = θ[k], state(j)
            if j > 0
                ℒ[i, j] = Operator(E -> -(Δ[i] .* (Wij * E)),
                                   Λ -> -(Wij' * (Δ[i] .* Λ)), "ℒ")
            end
            ℳ′[i, k] = Operator(h -> Δ[i] .* (h * Uj),
                                Λ -> (Δ[i] .* Λ) * Uj', "ℳ′")
        end
        ℳ′[i, b[i]] = Operator(h -> Δ[i] .* h,
                               Λ -> vec(sum(Δ[i] .* Λ, dims=2)), "ℳ′")
    end
    return ℒ, ℳ′
end

# The data and the parameters.
m = 10 # Data points
α = 1e-2 # Weight of the regularization α/2 ⟨θ, θ⟩
init(dims...) = 0.01 * randn(dims...)
U₀, Y = init(d[1], m), init(d[end], m)
θ = Vector{Array{Float64}}(undef, K)
for i in 1:N
    for (j, k) in W[i]
        θ[k] = init(d[i+1], d[j+1])
    end
    θ[b[i]] = init(d[i+1])
end

# The gradient of J(θ) = ‖U_N - Y‖²/(2m) + α/2 ⟨θ, θ⟩.
J(θ) = norm(network(θ)[1][N] - Y)^2 / (2m) + α / 2 * dot(θ, θ)
U, Δ = network(θ)
ℒ, ℳ′ = operators(θ, U, Δ)
∇Uℓ = [i == N ? (U[N] - Y) / m : zero(U[i]) for i in 1:N] # 𝒫*(∇f)
Λ = ℒ' \ ∇Uℓ
∇J = ℳ′' * Λ + α * θ # ℳ(Λ) + αθ
display(ℒ)
display(ℳ′)

# Central difference of the loss along a random direction.
h = [randn(size(p)...) for p in θ]
ε = 1e-6
directional = (J(θ + ε * h) - J(θ - ε * h)) / (2ε)
versus_finite_difference = abs(directional - dot(∇J, h)) / abs(dot(∇J, h))
@show versus_finite_difference
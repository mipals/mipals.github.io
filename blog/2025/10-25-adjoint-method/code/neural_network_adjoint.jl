using BlockDiagonalMatrices
using LinearAlgebra
using Kronecker
using Random
using Test

Random.seed!(42)

h(x)  =  exp(-x) # activation function
∇h(x) = -exp(-x) # derivative of activation function

# Forward pass including the derivatives
function forward_pass(u0, layers)
    u = u0
    _, b1 = first(layers)
    derivs = typeof(b1)[]
    kron_blocks = typeof([u; 1]' ⊗ I(length(b1)))[]
    for (W, b) in layers
        push!(kron_blocks,[u; 1]' ⊗ I(length(b))) # Lazy Kronecker product
        z = W*u + b
        u = h.(z)
        push!(derivs, ∇h.(z))
    end
    return kron_blocks, derivs, u
end
# Block back substitution: solves L^T y = b, i.e. y = L^(-T) b
function backsub(derivs, weights, b)
    y = Matrix(b)
    stop = length(y)
    @views for (d,Wt) in zip(reverse(derivs), reverse(transpose.(weights)))
        nprev, n = size(Wt)
        src = stop-n+1:stop
        dst = stop-n-nprev+1:stop-n
        tmp = d .* y[src]
        # Equivalent to: y[dst] .+= Wt * (d .* y[src])
        mul!(y[dst], Wt, tmp, 1, 1)
        stop -= n
    end
    return y
end
# Helper function that unpacks a vector θ into Ws and bs
function unpack_θ(θ, layer_sizes)
    Ws, bs = Matrix{eltype(θ)}[], Vector{eltype(θ)}[]
    i = 1
    for (n_in,n_out) in zip(layer_sizes[1:end-1], layer_sizes[2:end])
        j = i + n_out*n_in
        push!(Ws, reshape(θ[i:j-1], n_out, n_in))
        i = j + n_out
        push!(bs, θ[j:i-1])
    end
    return Ws, bs
end
# Evaluate the network output using the forward pass
function network_output(θ, layer_sizes, u0)
    Ws, bs = unpack_θ(θ, layer_sizes)
    _, _, uN = forward_pass(u0, zip(Ws,bs))
    return only(uN)
end
# Gradient computation using the adjoint method
function ∇L(θ, layer_sizes, u0; y=0.0)
    # First unpack the parameter vector
    Ws, bs = unpack_θ(θ, layer_sizes)
    # Forward pass includes derivative information
    kron_blocks, derivs, uN = forward_pass(u0, zip(Ws, bs))
    # S has subdiagonal blocks W_2,...,W_N; L = I - D*S and M^T = D*K
    D = Diagonal(vcat(derivs...))
    K = BlockDiagonal(kron_blocks)
    g = zeros(eltype(θ), sum(layer_sizes[2:end]))
    g[end] = 2*(only(uN) - y)
    # The final step is to evaluate the gradient from the right
    grad_adjoint = (backsub(derivs[2:end],Ws[2:end],g')*D)*K
    return grad_adjoint'
end
# Forward difference to test the adjoint gradient implementation
function fd(θ, layer_sizes, u0, i; e=1e-6, y=2.0)
    uN0 = network_output(θ, layer_sizes, u0)
    θ[i] += e
    uN1 = network_output(θ, layer_sizes, u0)
    θ[i] -= e
    return ((uN1 - y)^2 - (uN0 - y)^2)/e
end

# Setting up parameters

layer_sizes = [50,40,30,20,10,1]
N = length(layer_sizes) - 1
init(sizes...) = 0.01*randn(sizes...)
Ws = [init(layer_sizes[i+1], layer_sizes[i]) for i=1:N]
bs = [init(layer_sizes[i+1]) for i = 1:N]
u0 = init(layer_sizes[1])
layers = zip(Ws, bs)

# First we compute the forward pass
θvec = vcat([[W[:]; b] for (W, b) in layers]...)

## Testing the gradient
y = 3.0 # We aim to have the final output be 3.0
grad_adjoint = ∇L(θvec, layer_sizes, u0; y=y)
idx = 4000
grad_fd = fd(θvec, layer_sizes, u0, idx; e=1e-5, y=y)
@test grad_fd ≈ grad_adjoint[idx] atol=1e-6

# Optimizing to get the desired output y
for _ = 1:1000
    grad = ∇L(θvec, layer_sizes, u0; y=y)
    θvec .-= 0.001 .* grad
end
@test network_output(θvec, layer_sizes, u0) ≈ y
final_gradient = ∇L(θvec, layer_sizes, u0; y=y)
@test final_gradient ≈ zeros(length(θvec)) atol=1e-10
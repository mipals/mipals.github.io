#### Example: (Structured) Linear constraints
using LinearAlgebra 
n = 1000        # Number of linear constraints
c = rand(n)     # Objective vector
b = rand(n)     # Linear equality vector
θ = rand(2n-1)  # Parameters of the equality constraint

# Parametrizing linear equality constraint matrix
A(θ) = SymTridiagonal(θ[1:n], θ[n+1:end]) 

# Objective function
f(θ) = (c' * (A(θ) \ b))^2 

# Partial derivatives
partial(i,n,λ,u) = i <= n ?  λ[i]*u[i] : λ[i+1-n]*u[i-n] + λ[i-n]*u[i+1-n]

# Gradient computation
function ∇f(θ)
    n = length(b)
    M = A(θ)                # Defining constraint matrix
    u = M \ b               # Computing solution
    λ = M \ (-2*c*(c'*u))   # Adjoint variables
    return [partial(i,n,λ,u) for i = 1:2n-1]
end

# Testing against ForwardDiff
using Test, ForwardDiff
@test ForwardDiff.gradient(f,θ) ≈ ∇f(θ)



#### Example: Neural Network
# add https://github.com/mipals/BlockDiagonalMatrices.jl.git # For BlockDiagonalMatrices
using Test, ForwardDiff, LinearAlgebra, BlockBandedMatrices, SparseArrays, BlockDiagonalMatrices, Kronecker
h(x)  =  exp(-x) # activation function
∇h(x) = -exp(-x) # derivative of activation function

# Forward pass including the derivatives
function forward_pass(u0,θ)
    x0 = u0
    diags = empty([first(θ)[2]])
    krons = empty([first(θ)[2]' ⊗ I(2) ])
    for (W,b) in θ
        push!(krons,[x0; 1]' ⊗ I(length(b))) # Lazy Kronecker producect using Kronecker.jl
        tmp = W*x0 + b  # Can be used for both forward pass and derivative pass
        x0 = h.(tmp)
        push!(diags, ∇h.(tmp))
    end
    return krons, diags, x0
end
# Block backsubstitution: Solving L^(-T)y = b
function backsub(dblks,wblks,b)
    y  = convert.(eltype(wblks[1]), copy(b))
    j0 = length(b)
    i0 = length(b) - size(wblks[end],1)
    @views for (D,blk) in (zip(reverse(dblks),reverse(Transpose.(wblks))))
        i1,j1 = size(blk)
        tmp = D .* y[j0-j1+1:j0]
        mul!(y[i0-i1+1:i0], blk, tmp, 1, 1)
        j0 -= j1
        i0 -= i1
    end
    return y
end
# Helper function thats packs a vector θ into Ws and bs
function pack_θ(θ, Ws_sizes, bs_sizes)
    We = empty([ones(eltype(θ),1,1)])
    be = empty([ones(eltype(θ),1)])
    i = 1
    for (W_size,b_size) in zip(Ws_sizes, bs_sizes)
        j = i+prod(W_size)
        push!(We, reshape(θ[i:j-1], W_size...))
        i = j + b_size
        push!(be, θ[j:i-1])
    end
    return We, be
end
# Evaluating f using the forward pass
function eval_f(θ, Ws_sizes, bs_sizes, u0)
    We,be = pack_θ(θ, Ws_sizes, bs_sizes)
    _,_,uN = forward_pass(u0, zip(We,be))
    return uN
end
# Gradient computation using the adjoint method
function ∇f(θ, Ws_sizes, bs_sizes, u0; y=0.0)
    # First pack vector parameters to matrices
    We,be = pack_θ(θ, Ws_sizes, bs_sizes)
    # Forward parse includes derivative information
    krons, ddiags, uN = forward_pass(u0, zip(We,be))
    # We here use that L' = I - blkdiag(W_1,..., W_N)D' and M^T = D*K 
    D = Diagonal(vcat(ddiags...))
    K = BlockDiagonal(krons)
    g = zeros(eltype(θ), sum(w_sizes -> w_sizes[1], Ws_sizes))
    g[end] = 2*(uN[1] - y) # uN is a 1x1 matrix so extract the scalar
    # The final step is to evaluate the gradient from the right
    grad_adjoint = (backsub(ddiags,We[2:end],g')*D)*K
    return grad_adjoint'
end
# Forward difference to test the adjoint gradient implementation
function fd(θ, Ws_sizes,bs_sizes,u0,i;e=1e-6,y=2.0)
    f0 = eval_f(θ, Ws_sizes, bs_sizes, u0)
    θ[i] += e
    f1 = eval_f(θ, Ws_sizes, bs_sizes, u0)
    θ[i] -= e
    return sum(((f1[1] - y)^2 - (f0[1] - y)^2)/e)
end

# Setting up parameters 
layer_sizes = [50,40,30,20,10,1]
N = length(layer_sizes) - 1
init(sizes...) = 0.01*randn(sizes...)
Ws = [init(layer_sizes[i+1],layer_sizes[i]) for i=1:N]
bs = [init(layer_sizes[i+1]) for i = 1:N]
u0 = init(layer_sizes[1],1)[:]
θ  = zip(Ws,bs)

Ws_sizes = size.(Ws)
bs_sizes = length.(bs)

# First we compute the forward pass
θvec = vcat([[W[:]; b] for (W,b) in θ]...)

## Testing the gradient
y = 3.0 # We aim to have the final output be 3.0
grad_adjoint = ∇f(θvec, Ws_sizes, bs_sizes, u0;y=y)
idx = 4000
@test fd(θvec, Ws_sizes,bs_sizes,u0,idx;e=1e-5,y=y) ≈ grad_adjoint[idx] atol=1e-6

# Optimizing to get the output y
for iter = 1:1000
    grad = ∇f(θvec, Ws_sizes, bs_sizes, u0; y=y) # Y is the output value we want
    θvec -= 0.001*grad
end
@test eval_f(θvec,Ws_sizes,bs_sizes,u0)[1] ≈ y
@test ∇f(θvec, Ws_sizes, bs_sizes, u0;y=y) ≈ zeros(length(θvec)) atol=1e-10


# Jacobian vs. JVP
jac = ForwardDiff.jacobian(x -> ∇f(x, Ws, bs, u0), θvec)
col_no = 2000
ei = zeros(length(θvec)); ei[col_no] = 1
e = 1e-5
jaci = (∇f(θvec + e*ei, Ws, bs, u0) - ∇f(θvec, Ws, bs, u0))/e

jaci ./ jac[:,col_no]


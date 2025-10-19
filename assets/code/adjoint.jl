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
using Test, ForwardDiff, LinearAlgebra, BlockBandedMatrices, SparseArrays, BlockDiagonalMatrices, Kronecker
h(x) =   exp(-x) # sample activation function
∇h(x) = -exp(-x)

n = [50,40,30,20,10,1]  ## this contains [n₀...n_N]
k = 10 # batchsize
N = length(n)-1
init(sizes...) = 0.01randn(sizes...)
Ws = [init(n[i+1],n[i])  for i=1:N]
bs = [init(n[i+1]) for i = 1:N]
y  = init(n[end],k); #  y is what we will compare X_N agains
u0 = init(n[1],1)[:]
θ = zip(Ws,bs)

function fd(Ws,bs,u0,i;e=1e-6,y=2.0)
    Ws_sizes = prod.(size.(Ws)) # Number of W-parameters by layer
    cumulative_no_params_pr_layer = cumsum(Ws_sizes + length.(bs))
    idx = searchsortedfirst(cumulative_no_params_pr_layer,i) # Find layer for parameter "i"
    v = [0; cumulative_no_params_pr_layer]  # Parameter offset pr. layer
    We, be = deepcopy(Ws), deepcopy(bs)     # Copying input parameters
    # Add small change ("e") to either W or b
    if i - v[idx] <= Ws_sizes[idx]
        We[idx][i- v[idx]] += e
    else 
        be[idx][i - v[idx] - Ws_sizes[idx]] += e
    end
    # Initialize both forward passes
    x0, xe = u0, u0
    for (W,b,Wd,be) in zip(Ws,bs,We,be)
        x0, xe = h.(W*x0 + b), h.(Wd*xe + be) # Forward Pass
    end
    return sum(((xe[1] - y)^2 - (x0[1] - y)^2)/e)
end
function forward_pass(u0,θ)
    x0 = u0
    diags = empty([first(θ)[2]])
    krons = empty([first(θ)[2]' ⊗ I(2) ])
    # krons = Vector{Any}()
    for (W,b) in θ
        push!(krons,[x0; 1]' ⊗ I(length(b))) # Lazy Kronecker
        # push!(krons,kron([x0; 1]', I(length(b)))) # Dense Kronecker
        tmp = W*x0 + b  # Can be used for both forward pass and derivative
        x0 = h.(tmp)
        push!(diags, ∇h.(tmp))
    end
    return krons, diags, x0
end
function backsub(dblks,wblks,b)
    y  = convert.(eltype(wblks[1]), copy(b))
    j0 = length(b)
    i0 = length(b) - size(wblks[end],1)
    @views for (D,blk) in (zip(reverse(dblks),reverse(Transpose.(wblks))))
        i1,j1 = size(blk)
        tmp = D .* y[j0-j1+1:j0]
        mul!(y[i0-i1+1:i0], blk, tmp, 1, 1) # We have to use y here
        # y[i0-i1+1:i0] += blk*tmp
        j0 -= j1
        i0 -= i1
    end
    return y
end
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
function eval_f(θ, Ws, bs, u0)
    We,be = pack_θ(θ,Ws,bs)
    _,_,uN = forward_pass(u0, zip(We,be))
    return uN
end
function ∇f(θ, Ws_sizes, bs_sizes, u0; y=2.0)
    We,be = pack_θ(θ, Ws_sizes, bs_sizes)
    krons, ddiags, uN = forward_pass(u0, zip(We,be))
    D = Diagonal(vcat(ddiags...))
    # println(eltype(krons))
    K = BlockDiagonal(krons)
    # g have the size to all the combined size of all output states
    g = zeros(eltype(θ), sum(n[2:end]))
    g[end] = 2*(uN[1] - y) # Final layer is the scaler we're after
    # # We can now compute the gradient using the adjoint method
    grad_adjoint = (backsub(ddiags,We[2:end],g')*D)*K
    return grad_adjoint'
end

Ws_sizes = size.(Ws) # Number of W-parameters by layer
bs_sizes = length.(bs)

# First we compute the forward pass
θvec = vcat([[W[:]; b] for (W,b) in θ]...)
eval_f(θvec,Ws_sizes,bs_sizes,u0)

## Testing the gradient
y = 3.0
grad_adjoint = ∇f(θvec, Ws_sizes, bs_sizes, u0;y=y)
idx = 4000
@test fd(Ws,bs,u0,idx;e=1e-5,y=y) ≈ grad_adjoint[idx] atol=1e-6


# Optimizing to get the output y
for iter = 1:1000
    grad = ∇f(θvec, Ws_sizes, bs_sizes, u0; y=y) # Y is the output value we want
    θvec -= 0.001*grad
end
eval_f(θvec,Ws_sizes,bs_sizes,u0)
∇f(θvec, Ws_sizes, bs_sizes, u0;y=y) # Is the gradient close to 0?


# Jacobian vs. JVP
jac = ForwardDiff.jacobian(x -> ∇f(x, Ws, bs, u0), θvec)
col_no = 2000
ei = zeros(length(θvec)); ei[col_no] = 1
e = 1e-5
jaci = (∇f(θvec + e*ei, Ws, bs, u0) - ∇f(θvec, Ws, bs, u0))/e

jaci ./ jac[:,col_no]


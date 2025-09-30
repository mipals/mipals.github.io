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
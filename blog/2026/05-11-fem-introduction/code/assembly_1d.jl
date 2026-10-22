### Importing relevant packages
using FastGaussQuadrature
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using ForwardDiff
using Random
Random.seed!(42)
## Defining geometry.
# Impedance tube of diameter 10 cm and length 1 m (used later)
D = 0.1             # 100 mm diameter
L = 10*D            # Length of the cavity
ne = 600            # Number of quadratic elements
nnt = 2*ne+1        # Total number of nodes
h = L/ne            # Length of the elements
x = Vector(0:h/2:L) # Coordinates table

## Computing the element matrices
# Local basis functions (gradient by ForwardDiff: slow but easy)
Tᵉ(u)  = [u .* (u .- 1)/2; 1 .- u .^2; u .* (u .+ 1)/2]'
∇Tᵉ(u) = hcat(ForwardDiff.derivative.(Tᵉ,u)...)
# All elements are equal, so the Jacobian is the same for each.
# It maps [-1,1] onto [x_i,x_{i+1}], i.e. length 2 onto length h.
jacobian(u) = h/2
# In 1D the Jacobian function and matrix are equal (not so in 2D).
J(u) = h/2
# The element matrices, the same for all elements of equal size.
Q = 3  # Number of Gauss points
u,w = gausslegendre(Q)
Me = sum(i -> w[i]*Tᵉ(u[i])'*Tᵉ(u[i])*jacobian(u[i]),1:Q)
Ke = sum(i -> w[i]*∇Tᵉ(u[i])'*J(u[i])^(-1)*J(u[i])^(-1)*∇Tᵉ(u[i])*
              jacobian(u[i]),1:Q)

## Assembly 1: with the localization matrices (never do this!)
function assembly1(Me, Ke, nnt, ne)
    K = zeros(nnt, nnt)  # Dense matrix! Not ideal!
    M = zeros(nnt, nnt)  # Dense matrix! Not ideal!
    for ie = 1:ne
        Le = zeros(3, nnt)
        Le[:, ie*2-1:ie*2+1] = Diagonal(ones(3))
        K += Le'*Ke*Le
        M += Le'*Me*Le
    end
    return K, M
end
@btime K, M = assembly1(Me, Ke, nnt, ne)

## Assembly 2: with indexing instead of localization matrices
function assembly2(Me, Ke, nnt, ne)
    K = zeros(nnt, nnt)  # Dense matrix! Not ideal!
    M = zeros(nnt, nnt)  # Dense matrix! Not ideal!
    for ie = 1:ne
        K[ie*2-1:ie*2+1, ie*2-1:ie*2+1] += Ke
        M[ie*2-1:ie*2+1, ie*2-1:ie*2+1] += Me
    end
    return K, M
end
@btime K, M = assembly2(Me, Ke, nnt, ne)

## Assembly 3: sparse, using the compact support of the elements
function assembly3(Me, Ke, nnt, ne)
    I = zeros(Int64, 4nnt - 3)
    J = zeros(Int64, 4nnt - 3)
    Kd = zeros(ComplexF64, length(I))
    Md = zeros(ComplexF64, length(I))
    for ie=1:ne
        for i = 1:3
            r = (8*(ie-1)+1 + 3*(i-1)):(8*(ie-1) + 3*i)
            I[r]  .= ie*2-1:ie*2-1+2
            J[r]  .= (ie-1)*2 + i
            Kd[r] += Ke[:, i]
            Md[r] += Me[:, i]
        end
    end
    K = sparse(I, J, Kd)
    M = sparse(I, J, Md)
    return K, M
end
@btime K, M = assembly3(Me, Ke, nnt, ne)
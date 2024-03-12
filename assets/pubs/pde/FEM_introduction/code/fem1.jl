# This file was generated, do not modify it. # hide
### Importing relevant packages
using ForwardDiff
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FastGaussQuadrature
## Defining geometry.
# Models Impedance Tube of 10cm diameter and 1m in length (used later)
D = 0.1             # 100 mm diameter
L = 10*D            # Length of the cavity
ne = 600            # Number of quadratic elements
nnt = 2*ne+1        # Total number of nodes
h = L/ne            # Length of the elements
x = Vector(0:h/2:L) # Coordinates table

## Computing the element matrices
# Defining local basis functions (and gradient using ForwardDiff - This is inefficient but easy)
Tᵉ(u)  = [u .* (u .- 1)/2; 1 .- u .^2; u .* (u .+ 1)/2]'
∇Tᵉ(u) = hcat(ForwardDiff.derivative.(Tᵉ,u)...)
# Every element is the same, so the Jacobian does not depend on the element in this case. 
# Furthermore we map from [-1,1] onto [x_i,x_{i+1}]. Meaning from length 2 to length h.
jacobian(u) = h/2 
# In the 1D case the Jacobian function and matrix are equal. This is not true in higher dimensions.
J(u) = h/2
# Defining the local element matrices. Since the elements are the same size its constant.
Q = 3  # Number of Gaussian points used in the integration. 
u,w = gausslegendre(Q)
Me = sum(i -> w[i]*Tᵉ(u[i])'*Tᵉ(u[i])*jacobian(u[i]),1:Q)
Ke = sum(i -> w[i]*∇Tᵉ(u[i])'*J(u[i])^(-1)*J(u[i])^(-1)*∇Tᵉ(u[i])*jacobian(u[i]),1:Q)

## Assembly 1: Simple (using the element localization matrices. Never do this!)
function assembly1(Me,Ke,nnt,ne)
    K = zeros(nnt,nnt)  # Dense matrix! Not ideal!
    M = zeros(nnt,nnt)  # Dense matrix! Not ideal!
    for ie = 1:ne
        Le = zeros(3,nnt)
        Le[:,ie*2-1:ie*2+1] = Diagonal(ones(3))

        K += Le'*Ke*Le
        M += Le'*Me*Le
    end
    return K,M
end
@btime K,M = assembly1(Me,Ke,nnt,ne)

## Assembly 2: Intermediate (using indexing instead of the element localization matrices)
function assembly2(Me,Ke,nnt,ne)
    K = zeros(nnt,nnt)  # Dense matrix! Not ideal!
    M = zeros(nnt,nnt)  # Dense matrix! Not ideal!
    for ie = 1:ne
        K[ie*2-1:ie*2+1,ie*2-1:ie*2+1] += Ke
        M[ie*2-1:ie*2+1,ie*2-1:ie*2+1] += Me
    end
    return K,M
end
@btime K,M = assembly2(Me,Ke,nnt,ne) # Note that the matrices are here still dense. 

## Assembly 3: Advanced (Sparse assembly using the compact support of the elements.)
function assembly3(Me,Ke,nnt,ne)
    I = zeros(Int64,4nnt-3)
    J = zeros(Int64,4nnt-3)
    Kd = zeros(ComplexF64,length(I))
    Md = zeros(ComplexF64,length(I))
    for ie=1:ne
        for i = 1:3
            I[(8*(ie-1)+1 + 3*(i-1)):(8*(ie-1) + 3*i)]  .= ie*2-1:ie*2-1+2
            J[(8*(ie-1)+1 + 3*(i-1)):(8*(ie-1) + 3*i)]  .= (ie-1)*2 + i
            Kd[(8*(ie-1)+1 + 3*(i-1)):(8*(ie-1) + 3*i)] += Ke[:,i]
            Md[(8*(ie-1)+1 + 3*(i-1)):(8*(ie-1) + 3*i)] += Me[:,i]
        end
    end
    K = sparse(I,J,Kd)
    M = sparse(I,J,Md)
    return K,M
end
@btime K,M = assembly3(Me,Ke,nnt,ne)
### Importing relevant packages
using Plots
using ForwardDiff
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using FastGaussQuadrature
## Defining geometry.
Lx = 10
Ly = 4
n = 40+1
m = 16+1
ne = (n-1)*(m-1)    # Number of linear elements
nnt = n*m           # Total number of nodes
x = Vector(0:Lx/(n-1):Lx)
y = Vector(0:Ly/(m-1):Ly)
X = kron(x,ones(m))
Y = kron(ones(n),y)
scatter(X,Y,aspect_ratio=1)
## Creating the topology
T = reshape(1:nnt,m,n)
topology = zeros(Int,4,ne)
topology[1,:] = T[1:end-1,1:end-1][:]
topology[2,:] = T[2:end,1:end-1][:]
topology[3,:] = T[2:end,2:end][:]
topology[4,:] = T[1:end-1,2:end][:]

## Step 2: Computing Element matrices
# Defining local basis functions (and gradient using ForwardDiff - This is inefficient but easy)
Tᵉ(u)  = [(1-u[1])*(1-u[2]);(1+u[1])*(1-u[2]);(1+u[1])*(1+u[2]);(1-u[1])*(1+u[2])]'/4
∇Tᵉ(u) = hcat(ForwardDiff.jacobian(N,u))'
# Every element is the same, so the Jacobian does not depend on the element in this case.
# Furthermore we map from [-1,1] onto [x_i,x_{i+1}]. Meaning from length 2 to length h.
jacobian(u) = (x[2]-x[1])/2*(y[2]-y[1])/2
# In the 1D case the Jacobian function and matrix are equal. This is not true in higher dimensions.
J(u) = Diagonal([x[2]-x[1];y[2]-y[1]])/2
# Defining the local element matrices. Since the elements are the same size its constant.
Q = 2  # Number of Gaussian points used in the integration.
u,wu = gausslegendre(Q)
v,wv = gausslegendre(Q)
U = kron(u,ones(Q))
V = kron(ones(Q),v)
W = kron(wu,wv)
P = [U';V']

Me = sum(i -> W[i]*jacobian(P[:,i])*Tᵉ(P[:,i])'*Tᵉ(P[:,i]),1:2Q)
Ke = sum(i -> W[i]*jacobian(P[:,i])*∇Tᵉ(P[:,i])'*J(P[:,i])^(-1)*J(P[:,i])^(-1)'*∇Tᵉ(P[:,i]),1:2Q)

## Assembly: Advanced (Sparse assembly using the compact support of the elements.)
function connected_topology(topology,nnt,ne)
    source_connections = [zeros(Int,0) for _ in 1:nnt]
    for element = 1:ne
        for i = 1:4
            append!(source_connections[topology[i,element]],topology[:,element])
        end
    end
    sort!.(unique.(source_connections))
end
function create_I_J(connections)
    I = zeros(Int,sum(length.(connections)))
    J = zeros(Int,sum(length.(connections)))
    lower = 1
    for (idx,con) in enumerate(connections)
        upper = lower + length(con)
        I[lower:upper-1]  = con
        J[lower:upper-1] .= idx
        lower = upper
    end
    return I,J
end
function assembly(Me,Ke,topology,nnt)
    ne = size(topology,2)
    connections = connected_topology(topology,nnt,ne)
    I,J = create_I_J(connections)
    S = sparse(I,J,1:length(I)) # Sparse matrix representing the indices of Kd and Md
    Kd = zeros(length(I))
    Md = zeros(length(I))
    for ie=1:ne
        top = topology[:,ie]
        Kd[S[top,top]] += Ke
        Md[S[top,top]] += Me
    end
    K = sparse(I,J,Kd)
    M = sparse(I,J,Md)
    return K,M
end
@btime K,M = assembly(Me,Ke,topology,nnt);
K,M = assembly(Me,Ke,topology,nnt);


using Arpack
c₀ = 343.0
F = eigs(c₀^2*K,M,which=:SM,nev=10)
freq = sort(sqrt.(abs.(F[1]))/2/π)
plot((contourf(x,y,reshape(F[2][:,id],m,n),linewidth=0,levels=1000,legend=false,axis=false,title="$(round(freq[id],digits=2)) Hz") for id in 2:10)..., layout = (3, 3))

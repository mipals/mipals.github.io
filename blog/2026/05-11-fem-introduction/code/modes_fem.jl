include("modes_geometry.jl")
using ForwardDiff, SparseArrays, LinearAlgebra
using FastGaussQuadrature, Arpack
c₀ = 342.2
## Creating the topology
T = reshape(1:nnt,m,n)      # Numbering all nodes on the grid
topology = zeros(Int,4,ne)
topology[1,:] = T[1:end-1,1:end-1][:]
topology[2,:] = T[1:end-1,2:end][:]
topology[3,:] = T[2:end,2:end][:]
topology[4,:] = T[2:end,1:end-1][:]
## Computing Element matrices
# Local basis functions (gradient by ForwardDiff: slow but easy)
Tᵉ(u)  = [(1-u[1])*(1-u[2]); (1+u[1])*(1-u[2]);
          (1+u[1])*(1+u[2]); (1-u[1])*(1+u[2])]'/4
∇Tᵉ(u) = hcat(ForwardDiff.jacobian(Tᵉ,u))'
# All elements are equal, so the Jacobian is the same for each.
jacobian(u) = (x[2]-x[1])/2*(y[2]-y[1])/2
J(u) = Diagonal([x[2]-x[1];y[2]-y[1]])/2
# Defining the local element matrices.
Q = 2
u,wu = gausslegendre(Q)
v,wv = gausslegendre(Q)
U = kron(u,ones(Q))
V = kron(ones(Q),v)
W = kron(wu,wv)
P = [U';V']

Me = sum(i -> W[i]*jacobian(P[:,i])*Tᵉ(P[:,i])'*Tᵉ(P[:,i]),1:Q^2)
Ke = sum(i -> W[i]*jacobian(P[:,i])*∇Tᵉ(P[:,i])'*
              J(P[:,i])'^(-1)*J(P[:,i])^(-1)*∇Tᵉ(P[:,i]),1:Q^2)

## Assembly
function connected_topology(topology,nnt,ne)
    source_connections = [zeros(Int,0) for _ in 1:nnt]
    for element = 1:ne
        for i = 1:4
            node = topology[i,element]
            append!(source_connections[node],topology[:,element])
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
        I[lower:upper-1] .= con
        J[lower:upper-1] .= idx
        lower = upper
    end
    return I,J
end
function assembly(Me,Ke,topology,nnt)
    ne = size(topology,2)
    connections = connected_topology(topology,nnt,ne)
    I,J = create_I_J(connections)
    S = sparse(I,J,1:length(I))
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
K,M = assembly(Me,Ke,topology,nnt);

using Arpack
F = eigs(c₀^2*K,M,which=:SM,nev=10);
freq = sort(sqrt.(abs.(F[1]))/2/π);

## The analytical frequencies, for comparison
nx, ny = 0:4, 0:4
freq_analytical = c₀/2*sqrt.((nx/Lx).^2 .+ (ny'/Ly).^2)
analytical_freqs = sort(freq_analytical[:])
println("FEM:        ", round.(freq[2:10],digits=2))
println("Analytical: ", round.(analytical_freqs[2:10],digits=2))
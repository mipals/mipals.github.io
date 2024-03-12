#===========================================================================================
                            Importing Relvant Packages
===========================================================================================#
using LegendrePolynomials
using LinearAlgebra
using FastGaussQuadrature
using SparseArrays
using Plots
#===========================================================================================
                            Computing Quadrature Points
===========================================================================================#
nodes,weights = gausslegendre(100)
nodes   = [-1.0; nodes; 1.0]
weights = [0.0; weights; 0.0]
#===========================================================================================
            Evaluating the p+1 first Legendre polynomials at the quadrature points
===========================================================================================#
pmax = 10
tmp = collectPl.(nodes,lmax=pmax)
M = hcat(tmp...)
TMP = hcat(collectPl.(nodes,lmax=pmax)...)
#===========================================================================================
            Evaluating inner products of the Legendre Polynomials (Similar to Figure 1)
===========================================================================================#
N = zeros(pmax+1,pmax+1)
for (i,x) in enumerate(eachrow(M))
    for (j,y) in enumerate(eachrow(M))
        N[i,j] = dot(x .* y, weights)
    end
end
N[abs.(N) .<= 1e-14] .= 0.0;
sparse(N)
Diagonal(2.0 ./ (2.0 .* collect(0:pmax) .+ 1.0))
# collectPl.(-1.0,lmax=pmax)
# collectPl.(+1.0,lmax=pmax)
n_max = maximum(N)
heatmap(N, color = reverse(cgrad(:greys)),clim=(0,n_max),
        aspect_ratio=1,ticks=nothing,border=:none,legend=false,yflip=true,size=(600,600))
#===========================================================================================
            Evaluating inner products of the modified Legendre Polynomials (Figure 1a)
===========================================================================================#
Mmod = deepcopy(M)
Nmod = deepcopy(N)
# NOTE: We use 1-indexing and not zero indexing!
Mmod[3:2:end,:] = M[3:2:end,:] .- 1.0
Mmod[4:2:end,:] = M[4:2:end,:] .- nodes'
Mmod[1,:] = 1.0 .- nodes'
Mmod[2,:] = 1.0 .+ nodes'
for (i,x) in enumerate(eachrow(Mmod))
    for (j,y) in enumerate(eachrow(Mmod))
        Nmod[i,j] = dot(x .* y, weights)
    end
end
Nmod[abs.(Nmod) .<= 1e-14] .= 0.0
sparse(Nmod)
mod_max = maximum(abs.(Nmod))
heatmap(abs.(Nmod), color = reverse(cgrad(:greys)),clim=(0,mod_max),
        aspect_ratio=1,ticks=nothing,border=:none,legend=false,yflip=true,size=(600,600))
#===========================================================================================
            Evaluating inner products of the modified Legendre Polynomials (Figure 1c)
===========================================================================================#
Mmod2 = deepcopy(M)
Nmod2 = deepcopy(N)
# NOTE: We use 1-indexing and not zero indexing!
Mmod2[3:2:end,:] = M[3:2:end,:] .- M[1:2:end-2,:]
Mmod2[4:2:end,:] = M[4:2:end,:] .- M[2:2:end-2,:]
Mmod2[1,:] = 1.0 .- nodes'
Mmod2[2,:] = 1.0 .+ nodes'
for (i,x) in enumerate(eachrow(Mmod2))
    for (j,y) in enumerate(eachrow(Mmod2))
        Nmod2[i,j] = dot(x .* y, weights)
    end
end
Nmod2[abs.(Nmod2) .<= 1e-14] .= 0.0
sparse(Nmod2)
mod2_max = maximum(abs.(Nmod2))
heatmap(abs.(Nmod2), color = reverse(cgrad(:greys)),clim=(0,mod2_max),
        aspect_ratio=1,ticks=nothing,border=:none,legend=false,yflip=true,size=(600,600))
Mmod2
#===========================================================================================
            Evaluating inner products of regular linear and quadratic elements
===========================================================================================#
linear(ξ)    = [0.5*(1.0 .- ξ); 0.5*(1.0 .+ ξ)]
Mlin = linear(nodes')
Nlin = zeros(2,2)
for (i,x) in enumerate(eachrow(Mlin))
    for (j,y) in enumerate(eachrow(Mlin))
        Nlin[i,j] = dot(x .* y, weights)
    end
end
Nlin
Mlin
lin_max = maximum(abs.(Nlin))
heatmap(abs.(Nlin), color = reverse(cgrad(:greys)),clim=(0,lin_max),
        aspect_ratio=1,ticks=nothing,border=:none,legend=false,yflip=true,size=(600,600))

quadratic(ξ) = [0.5 * ξ .* (ξ .- 1.0); 1.0 .- ξ .^2; 0.5 * ξ .* (ξ .+ 1.0)]
Mquad = quadratic(nodes')
Nquad = zeros(3,3)
for (i,x) in enumerate(eachrow(Mquad))
    for (j,y) in enumerate(eachrow(Mquad))
        Nquad[i,j] = dot(x .* y, weights)
    end
end
Nquad
Mquad

quad_max = maximum(abs.(Nquad))
heatmap(abs.(Nquad), color = reverse(cgrad(:greys)),clim=(0,quad_max),
        aspect_ratio=1,ticks=nothing,border=:none,legend=false,yflip=true,size=(600,600))
#===========================================================================================
                                    Scalings
===========================================================================================#
ms = collect(2:pmax)
Cm = [sqrt(3)/4;sqrt(3)/4;0.5*sqrt.((2.0*ms .- 3.0).*(2.0*ms .+ 1.0)./(2.0*ms .- 1.0))]
Cn = sqrt.(ms .+ 0.5)

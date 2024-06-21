# This file was generated, do not modify it. # hide
using LinearAlgebra, StatsBase, Plots, Random, InvertedIndices
Random.seed!(1234)

n = 20  # Number of data points
k = 5   # Rank of approximation
dim = 1 # Dimension of data
l = 2.0 # Lengthscale parameter

X = [rand(floor(Int,n/3),dim) .- 2; rand(floor(Int,n/3),dim); rand(floor(Int,n/3),dim)/2 .+ 1.5] # Three groups
n = length(X) # Number of data points
X = sort(X[:]) # Soring the data - Not strictly necessary, but makes the discrete form nicer
G(x,y,l=l) = exp(-norm(x-y)^2/(2*l))

# The actual view is in 2D
o = ones(n)
Xx = kron(X,o)
Xy = kron(o,X)
Gk = [G(x,y) for (x,y) in zip(Xx,Xy)]
plot_matrix = heatmap(sort(X[:]),sort(X[:]),reshape(Gk,n,n),aspect_ratio=:equal, title="Discrete")
scatter!(plot_matrix,Xx,Xy, label=false)
xlabel!(plot_matrix,"x"); ylabel!(plot_matrix,"y"); yflip!(true)

Xc = range(-2,2,300)
plot_smooth = contour(Xc,Xc, (x,y) -> G(x,y), fill=true,aspect_ratio=:equal,clim=(0,1))
xlabel!(plot_smooth,"x"); ylabel!(plot_smooth,"y"); yflip!(true)
scatter!(plot_smooth,Xx,Xy, label=false,title="Continuous")
scatter!(plot_smooth,X,X, label=false, color=:black)
plot(plot_matrix,plot_smooth, layour=(1,2),dpi=300)
savefig(joinpath(@OUTPUT, "initial_scatter.png")) # hide
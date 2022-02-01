# This file was generated, do not modify it. # hide
using GaussQuadrature, LinearAlgebra, Plots
N = 50
a = 0 
b = 2*pi
xi,wi = legendre(N)
f(x) = abs.(sin.(x))
xbar = 0.5*(b-a)*xi .+ 0.5*(b+a)
wbar = 0.5*(b-a)*wi
wbar'*f(xbar)
println(wbar'*f.(xbar)) # hide
plot(a:0.05:b,f.(a:0.05:b),label=L"|\sin(x)|") # hide
scatter!(xbar,f.(xbar),label=L"(\overline{x}_i, f(\overline{x}_i))") # hide
ylims!(-0.1,1.2) # hide
savefig(joinpath(@OUTPUT, "f_func.svg")) # hide
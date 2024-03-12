# This file was generated, do not modify it. # hide
nx, ny = 0:4, 0:4
freq_analytical = c₀/2*sqrt.((nx/Lx).^2 .+ (ny'/Ly).^2)
analytical_freqs = sort(freq_analytical[:])
print(round.(analytical_freqs[2:10],digits=2))
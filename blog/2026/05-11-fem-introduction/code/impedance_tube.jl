include("assembly_1d.jl")
using CairoMakie
## Recomputing FEM matrices
K,M = assembly3(Me,Ke,nnt,ne)
## Setup
ρ₀ = 1.2   # Fluid density
c₀ = 342.2 # Speed of sound
U_L = 1    # Piston displacement
D = 0.1    # Diameter of Impedance Tube
s = 0.5*D  # Microphone spacing
d = D/2    # Distance between mic 2 and sample
## Frequency domain
fc = floor(1.84*c₀/D/pi) # Cut off frequency
freq = Vector(100:2:fc)
ω = 2*pi*freq
k₀ = ω/c₀
## Impedance properties
Z₀ = ρ₀ * c₀
thickness = 0.02  # thickness of the material
σ = 10000 # flow resistivity
X = ρ₀*freq/σ
Zc = Z₀  *(1 .+ 0.0570*X.^(-0.754) - im*0.087.*X.^(-0.732))
kc = k₀ .*(1 .+ 0.0978*X.^(-0.700) - im*0.189.*X.^(-0.595))
Z = -im.*Zc.*cot.(kc*thickness) / Z₀
beta = 1.0 ./ Z   # convert to admittance
# Finding nodes located on the microphones
in2 = findall(x->x,abs.(x .- d) .< 1e-6)[1]
in1 = findall(x->x,abs.(x .- (d+s)) .< 1e-6)[1]
# Correcting the s and d distances to fit the numerical values
s = abs(x[in1] - x[in2])
d = x[in2]

## Output
ndof = nnt
nfreqs = length(ω)
P_mic1 = zeros(ComplexF64,nfreqs)
P_mic2 = zeros(ComplexF64,nfreqs)
A = Diagonal(zeros(ComplexF64,nnt))
F = zeros(ComplexF64,ndof)

## Frequency sweep
for i in eachindex(ω)
    A[1,1] = im*k₀[i]*beta[i]
    F[end,1] = ρ₀*ω[i]^2*U_L
    S = K - k₀[i]^2*M + A
    p = S\F
    P_mic2[i] = p[in2]
    P_mic1[i] = p[in1]
end

# Calculate the normalized impedance
H₁₂ = P_mic2 ./ P_mic1
R = (H₁₂ - exp.(-im*k₀*s))./(exp.(im*k₀*s) - H₁₂) .*
    exp.(im*2*k₀ * (d + s))
Z_num = (1 .+ R)./(1 .- R)
## Comparison with the exact solution
fig = Figure(size=(800,320))
ax_re = Axis(fig[1,1],xlabel="Frequency (Hz)",ylabel="Real part")
lines!(ax_re,freq,real.(Z),label="Analytical")
lines!(ax_re,freq,real.(Z_num),label="FEM",linestyle=:dash)
ax_im = Axis(fig[1,2],xlabel="Frequency (Hz)",ylabel="Imaginary part")
lines!(ax_im,freq,imag.(Z))
lines!(ax_im,freq,imag.(Z_num),linestyle=:dash)
axislegend(ax_re)
fig
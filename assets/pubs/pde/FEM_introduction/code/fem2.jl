# This file was generated, do not modify it. # hide
using Plots
## Recomputing FEM matrices
K,M = assembly3(Me,Ke,nnt,ne) # (using @btime earlier means not storing the results)
## Setup
ρ₀ = 1.2   # Fluid density
c₀ = 342.2 # Speed of sound
Uₙ = 1     # Piston displacement
D = 0.1    # Diameter of Impedance Tube
s = 0.5*D  # Microphone spacing
d = D/2    # Distance between mic 2 and sample
## Frequency domain
fc = floor(1.84*c₀/D/pi) # Cut off frequency
freq = Vector(100:2:fc)  # Choose correctly the lowest frequency ( a funtion of mics spacing)
ω = 2*pi*freq
k₀ = ω/c₀
## Impedance properties
Z₀ = ρ₀ * c₀
h = 0.02  # thickness of the material
σ = 10000 # flow resitivity
X = ρ₀*freq/σ
Zc = Z₀*(1 .+ 0.057*X.^(-0.754)-im*0.087.*X.^(-0.732))
k = k₀ .*(1 .+0.0978*X.^(-0.700)-im*0.189.*X.^(-0.595))
Z = -im.*Zc.*cot.(k*h) / Z₀
beta= 1.0 ./ Z # convert to admittance
# Finding nodes located on the microphones. 
in2=findall(x->x,abs.(x .-d) .< 1e-6)[1]        # Location of mic 2
in1=findall(x->x,abs.(x .-(d+s)) .<1e-6)[1]     # Location of mic 1
# Correcting the s and d distances to fit the numerical values.
s=abs(x[in1] - x[in2])  # Recalculate microphones separation
d=x[in2]                # Recalculate the distance between microphone 2 and the sample

## Output
ndof = nnt
nfreqs = length(ω)
P_mic1 = zeros(ComplexF64,nfreqs)
P_mic2 = zeros(ComplexF64,nfreqs)
A = Diagonal(zeros(ComplexF64,nnt))
F = zeros(ComplexF64,ndof) # Initialize force vector
k = ω/c₀

## Frequency sweep
for i in eachindex(ω)
    A[1,1] = im*k[i]*beta[i]
    F[end,1] = ρ₀*ω[i]^2*Uₙ
    S = K - k[i]^2*M + A
    p = S\F
    P_mic2[i] = p[in2]
    P_mic1[i] = p[in1]
end

# Calculate the normalized impedance
H₁₂ = P_mic2./P_mic1
R = (H₁₂ - exp.(-im*k*s))./(exp.(im*k*s)-H₁₂) .*exp.(im*2*k * (d + s))
Z_num = (1 .+ R)./(1 .- R)
## Comparison with the exact solution
plot(freq,real.(Z),label="Analytical",linewidth=2)
plot!(freq,real.(Z_num),label="FEM",linestyle=:dash,linewidth=2)
xlims!((100, 2000))
xlabel!("Frequency (Hz)")
ylabel!("Normalized Impedance - Real part")
savefig(joinpath(@OUTPUT, "fem_fig.svg")) # hide
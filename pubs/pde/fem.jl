using Plots
using SparseArrays
using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature

# Defining geometry. Models Impedance Tube of 10cm diameter and 1m in length
D = 0.1             # 100 mm diameter
L = 10 *D           # Length of the cavity
ne = 600            # Number of quadratic elements
nnt = 2*ne+1        # Total number of nodes
h=L/ne              # Length of the elements
x=Vector(0:h/2:L)   # Coordinates table

## Step 2: Computing Elementa matrices
# Defining
Tᵉ(u)  = [u .* (u .- 1)/2; 1 .- u .^2; u .* (u .+ 1)/2]'
∇Tᵉ(u) = hcat(ForwardDiff.derivative.(T,u)...)
G(u)  = Tᵉ(u)'*Tᵉ(u)*(h/2)
dG(u) = ∇Tᵉ(u)'*∇Tᵉ(u)*(2/h)
n = 3
u,w = gausslegendre(n)
Me = sum(i -> w[i]*G(u[i]),1:n)
Ke = sum(i -> w[i]*dG(u[i]),1:n)

## Assembly: Simple
@time begin
K = zeros(nnt,nnt)
M = zeros(nnt,nnt)
for ie = 1:ne
    Le = zeros(3,nnt)
    Le[:,ie*2-1:ie*2+1] = Diagonal(ones(3))

    K += Le'*Ke*Le
    M += Le'*Me*Le
end
end;

@time begin
K = zeros(nnt,nnt)
M = zeros(nnt,nnt)
for ie = 1:ne
    K[ie*2-1:ie*2+1,ie*2-1:ie*2+1] += Ke
    M[ie*2-1:ie*2+1,ie*2-1:ie*2+1] += Me
end
end;


## Step 3: Optimized
@time begin
I = zeros(Int64,4nnt-3)
J = zeros(Int64,4nnt-3)
Kd = zeros(ComplexF64,length(I))
Md = zeros(ComplexF64,length(I))
for ie=1:ne
    for i = 1:3
        I[(8*(ie-1)+1 + 3*(i-1)):(8*(ie-1) + 3*i)]   = ie*2-1:ie*2-1+2
        J[(8*(ie-1)+1 + 3*(i-1)):(8*(ie-1) + 3*i)]  .= (ie-1)*2 + i
        Kd[(8*(ie-1)+1 + 3*(i-1)):(8*(ie-1) + 3*i)] += Ke[:,i]
        Md[(8*(ie-1)+1 + 3*(i-1)):(8*(ie-1) + 3*i)] += Me[:,i]
    end
end
K = sparse(I,J,Kd)
M = sparse(I,J,Md)
end;


ρ₀ = 1.2   # Fluid density
c₀ = 342.2 # Speed of sound
U₀ = 1     # Piston displacement
s = 0.5*D  # microphone spacing
d = D/2    # distance between mic 2 and sample
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
## Step 1: Mesh
in2=findall(x->x,abs.(x .-d) .< 1e-6)[1]        # Location of mic 2
in1=findall(x->x,abs.(x .-(d+s)) .<1e-6)[1]     # Location of mic 1
s=abs(x[in1] - x[in2])  # Recalculate microphones separation
d=x[in2]                # and distance between microphone 2 and the sample



## Step 4 & 5: specify frquency dependent impedance and Solve the system with the Force vector : Piston at a x=L
ndof = nnt
nfreqs = length(ω)
P_mic1 = zeros(ComplexF64,nfreqs)
P_mic2 = zeros(ComplexF64,nfreqs)
A = Diagonal(zeros(ComplexF64,nnt))
F = zeros(ComplexF64,ndof) # Initialize force vector
k = ω/c₀

## ------Write script here------
for i in eachindex(ω)
    A[1,1] = im*k[i]*beta[i]
    F[end,1] = ρ₀*ω[i]^2*U₀
    S = K - k[i]^2*M + A
    P = S\F
    P_mic2[i] = P[in2]
    P_mic1[i] = P[in1]
end


# calculate the normalized impedance
H₁₂ = P_mic2./P_mic1
R = (H₁₂ - exp.(-im*k*s))./(exp.(im*k*s)-H₁₂) .*exp.(im*2*k * (d + s))
Z_num = (1 .+ R)./(1 .- R)
## Step 6: Comparison with the exact solution
plot(freq,real.(Z),label="Analytical",linewidth=2)
plot!(freq,real.(Z_num),label="FEM",linestyle=:dash,linewidth=2)
xlims!((100, 2000))
xlabel!("Frequency (Hz)")
ylabel!(" Normalized Impedance - Real part")
plot(freq,imag.(Z),label="Analytical",linewidth=2)
plot!(freq,imag.(Z_num),label="FEM",linestyle=:dash,linewidth=2)
xlims!((100, 2000))
xlabel!("Frequency (Hz)")
ylabel!(" Normalized Impedance - Imaginary part")
R_theo=(Z .- 1)./(Z .+ 1)
alpha_theo= 1 .- abs.(R_theo).^2
R_num = (Z_num .- 1)./(Z .+ 1)
alpha_num = 1 .- abs.(R_num).^2
plot(freq,alpha_theo,label="Analytical",linewidth=2)
plot!(freq,alpha_num,label="FEM",linestyle=:dash,linewidth=2)
xlims!((100, 2000))
xlabel!("Frequency (Hz)")
ylabel!(" Absorption coefficient")



using LinearAlgebra
using Plots
n = 500
A = randn(n, n) / sqrt(n)
eigvals_a = eigvals(A)
scatter(real.(eigvals_a),imag.(eigvals_a))

Ashift = A + 5*I
eigvals_shift = eigvals(Ashift)
scatter!(real.(eigvals_shift),imag.(eigvals_shift))

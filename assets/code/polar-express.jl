using LinearAlgebra

poly_coeffs = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
safe_poly_coeffs = [coeffs ./ (1.01, 1.01^3, 1.01^5) for coeffs in poly_coeffs[1:end-1]]

psd_coeffs = [
        ( 8.3119043343,  -23.0739115930,  16.4664144722 ),
        ( 4.1439360087,   -2.9176674704,   0.5246212487 ),
        ( 4.0257813209,   -2.9025002398,   0.5334261214 ),
        ( 3.5118574347,   -2.5740236523,   0.5050097282 ),
        ( 2.4398158400,   -1.7586675341,   0.4191290613 ),
        ( 1.9779835097,   -1.3337358510,   0.3772169049 ),
        ( 1.9559726949,   -1.3091355170,   0.3746734515 ),
        ( 1.9282822454,   -1.2823649693,   0.3704626545 ),
        ( 1.9220135179,   -1.2812524618,   0.3707011753 ),
        ( 1.8942192942,   -1.2613293407,   0.3676616051 )
]

function polynomial_approxmation(G, method=:msign; steps=10)
    # Tranpose G if required
    if_tranpose = >(size(G)...) 
    G = if_tranpose ? G : G'
    # Normalize (Shuld also push to float16)
    X = G/norm(G)
    # Getting element types of X
    elt = eltype(X)
    # Polynomial composition (not at the end the same polynomial is repeatedly applied)
    for step in 1:steps
        # This step forces to F64
        if method == :msign
            a1,a3,a5 = elt.(step <= length(safe_poly_coeffs) ? safe_poly_coeffs[step] : poly_coeffs[end])
        elseif method == :psd_projection
            a1,a3,a5 = elt.(step <= length(psd_coeffs) ? psd_coeffs[step] : psd_coeffs[end])
        end
        # Evaluating the polynomial using Horners method x*(a1 + x^2*(a3 + a5*x^2))
        X2 = X * X'      # Defining the square variable
        Y = a5*X2 + a3*I # Inner parentheses
        Y = Y*X2  + a1*I # Middle parentheses
        X = Y*X          # Outer parentheses
    end
    # If PSD projection is sought remember last step
    G = method == :msign ?  X : G*(I + X)/2
    # Transpose back if needed
    G = if_tranpose ? G' : G
    return G
end

function polar(G, steps=10)
    return polynomial_approxmation(G, :msign, steps=steps)
end

function relu(G, steps=10)
    return polynomial_approxmation(G, :psd_projection, steps=steps)
end


####
using LinearAlgebra
N = 5
A = rand(N,N)
A = A + A'

relu(x::Number) = x < 0 ? zero(x) : x


A_eigvals = eigvals(A)
Aeig = eigen(A)
Asvd = svd(A)


Aeig_sign = Aeig.vectors*Diagonal(sign.(Aeig.values))*Aeig.vectors'
Asvd_sign = Asvd.U*Diagonal(sign.(Asvd.S))*Asvd.Vt
Asvd_iter = msign(A,20)
eigvals(Aeig_sign) ./ sign.(A_eigvals)
eigvals(Asvd_sign) ./ sign.(A_eigvals)
eigvals(Asvd_iter) ./ sign.(A_eigvals)


Aeig_relu = Aeig.vectors*Diagonal(relu.(Aeig.values))*Aeig.vectors'
Asvd_relu = Asvd.U*Diagonal(relu.(Asvd.S))*Asvd.Vt # No signs so cant be used!
Asvd_iter = relu(A)

eigvals(Aeig_relu) ./ A_eigvals
eigvals(Asvd_relu) ./ A_eigvals
eigvals(Asvd_iter) ./ A_eigvals





function ppolar(x,steps=10; method=:psign)
    y = x
    elt = eltype(x)
    for step in 1:steps
        if method == :psign
            a1,a3,a5 = elt.(step <= length(safe_poly_coeffs) ? safe_poly_coeffs[step] : poly_coeffs[end])
        else
            a1,a3,a5 = elt.(step <= length(psd_coeffs) ? psd_coeffs[step] : psd_coeffs[end])
        end
        y = a1*y + a3*y^3 + a5*y^5
    end
    return method == :psign ? y : x * (1 + y)/2
end

function psign(x,steps=10)
    return ppolar(x,steps; method=:psign)
end
function ppsd(x,steps=10)
    return ppolar(x,steps; method=:psd)
end


## Sign
using Plots, LinearAlgebra
function NewtonSchulz(x, steps=10; order=5)
    p(x) = order == 5 ? (15*x - 10*x^3 + 3*x^5)/8 :  3/2*x - x^3/2
    for step in 1:steps
        x = p(x)
    end
    return x
end

n = 1000
x = sort(2*rand(n)) .- 1
p1 = plot(x, sign.(x), linewidth=3, legend=:topleft, label="sign(x)")
for i in 1:3:10
    plot!(p1,x, NewtonSchulz.(x,i), linewidth=2, label="NewtonShulz$(i)", color=:green)
    plot!(p1,x, psign.(x,i), linewidth=2, label="Polar$(i)", linestyle=:dash,color=:red)
end
display(p1)


p2 = plot()
for i in 1:2:10
    plot!(p2,x, ppsd.(x,i))
end
display(p2)


### WHY?
p(x) = I + x^2
p(A)

Aeig.vectors*Diagonal(p.(Aeig.values))*Aeig.vectors'



i = 5
plot(x,abs.(sign.(x) - psign.(x,i)) .+ eps(Float64),yscale=:log10)
plot!(x,abs.(sign.(x) - NewtonSchulz.(x,i)) .+ eps(Float64),yscale=:log10)


plot(x, sign.(x), label="sign",legend=:topleft)
plot!(x, psign.(x,i), label="psign")
plot!(x, NewtonSchulz.(x,i), label="NewtonSchulz")



using WGLMakie

x = LinRange(0, 2π, 200)
fig = Figure(resolution=(800,400))
ax = Axis(fig[1,1])
lines!(ax, x, sin.(x), label="sin")
lines!(ax, x, cos.(x), label="cos")
axislegend(ax)  # or Legend(fig[...] , ... ) and place where desired

# show in browser (starts a local server)
display(fig)



using Test, LinearAlgebra
A = rand(2,2)
A = Diagonal([0.5, -0.5])
S = svd(A)
@test S.U * Diagonal(sin.(S.S)) * S.Vt ≈ sin(A)
@test S.U * Diagonal(cos.(S.S)) * S.Vt ≈ cos(A) # Now f is not odd


fsin(A,k=3) = sum(n -> (-1)^n*A^(2n+1)/factorial(2n+1), 0:k)
fcos(A,k=3) = sum(n -> (-1)^n*A^(2n)/factorial(2n), 0:k)


Podd(x) = x - x^3/6 + x^5/120 - x^7/5040
@test S.U * Diagonal(Podd.(S.S)) * S.Vt ≈ Podd(A)
Peven(x) = x - x^2/2 + x^4/24 - x^6/720
@test S.U * Diagonal(Peven.(S.S)) * S.Vt ≈ Peven(A)



# Newton Schulz
P = randn(100,100)/25
S = svd(P)
Polar = S.U*S.Vt
p(x,order=5) = order == 5 ? (15*x - 10*(x*x')*x + 3*(x*x')^2*x)/8 :  3/2*x - (x*x')*x/2
for i = 1:20
    P = p(P)
    println("iteration $i: ||P - UV'||_F = ", norm(P - Polar))
end
function Φblocks(u,θ;u0=u0)
    U = [u0,u...]
    return [h.(W*ui + b) for (ui,(W,b)) in zip(U,θ)]
end 
function f(u,θ;u0=u0)
    nsplits = cumsum(n) .- (n[1] - 1)
    U = [u[low:up-1] for (low,up) in zip(nsplits[1:end-1],nsplits[2:end])]
    Φblks = Φblocks(U,θ;u0=u0)
    return u - vcat(Φblks...)
end

u0 = rand(n[1])
uinit = [randn(i) for i in n[2:end]]
f(vcat(uinit...),θ)


ForwardDiff.jacobian(x -> f(x,zip(Ws,bs)),vcat(uinit...))


function compute_∇Φblks(u,θ;u0=u0)
    nsplits = cumsum(n) .- (n[1] - 1)
    U = [u[low:up-1] for (low,up) in zip(nsplits[1:end-1],nsplits[2:end])]
    U = [u0,U...]
    return [∇h.(W*ui + b) .* W for (ui,(W,b)) in zip(U,θ)]
end


function ∂f∂u(u,θ;u0=u0)
    ∇Φblks = compute_∇Φblks(u,θ;u0=u0)
    zeroblocks = [zeros(i,i) for i in n[2:end]]
    
    return I - Matrix(BlockBidiagonal(zeroblocks, ∇Φblks[2:end],:L))
end

@test ∂f∂u(vcat(uinit...),θ;u0=u0) ≈ ForwardDiff.jacobian(x -> f(x,zip(Ws,bs),u0=u0),vcat(uinit...))

function ∂f∂θ(u,θ;u0=u0)
    nsplits = cumsum(n) .- (n[1] - 1)
    U = [u[low:up-1] for (low,up) in zip(nsplits[1:end-1],nsplits[2:end])]
    U = [u0,U...]
    
    ∇θblks = [Diagonal(∇h.(W*ui + b)[:])*kron([ui; 1]',I(length(b)))   for (ui,(W,b)) in zip(U,θ)]

    return dropzeros!(sparse(BlockDiagonal(∇θblks)))
end

function Mblocks(u,θ;u0=u0)
    nsplits = cumsum(n) .- (n[1] - 1)
    U = [u[low:up-1] for (low,up) in zip(nsplits[1:end-1],nsplits[2:end])]
    U = [u0,U...]
    
    uis = [vcat(ui, 1) for ui in U]
    krons = [ui' ⊗ I(length(b)) for (ui,(W,b)) in zip(uis,θ)] # Lazy Kronecker
    diags = [∇h.(W*ui + b)  for (ui,(W,b)) in zip(U,θ)]

    return diags, krons, uis
end


x0 = u0
u = []
for layer in [x -> h.(W*x + b) for (W,b) in θ]
    x0 = layer(x0)
    push!(u,x0)
end
u = vcat(u...)



gh = ∇h.(Ws[1]*u0 + bs[1])
G = gh*u0'

FwdW = ForwardDiff.jacobian(W -> h.(W*u0 + bs[1]), Ws[1]) 
Fwdb = ForwardDiff.jacobian(b -> h.(Ws[1]*u0 + b), bs[1])

# 
dW = Diagonal(gh[:])*kron(u0',I(n[2])) 
@test FwdW ≈ dW

# FD
@test Diagonal(gh[:]) ≈ Fwdb

# Everyhing together
F = hcat([Diagonal(gh[:]*u) for u in [u0; 1]]...)
@test [FwdW Fwdb] ≈ F


nsplits = cumsum(n) .- (n[1] - 1)
U = [u[low:up-1] for (low,up) in zip(nsplits[1:end-1],nsplits[2:end])]
U = [u0,U...]

i = 2
FwdWi = ForwardDiff.jacobian(W -> h.(W*U[i] + bs[i]), Ws[i]) 
Fwdbi = ForwardDiff.jacobian(b -> h.(Ws[i]*U[i] + b), bs[i])
∇θblks = [Diagonal(∇h.(W*ui + b)[:])*kron([ui; 1]',I(length(b)))   for (ui,(W,b)) in zip(U,θ)]
@test [FwdWi Fwdbi] ≈ ∇θblks[i]

##
M = ∂f∂θ(u,θ;u0=u0)
Mblks = Mblocks(u,θ;u0=u0)
ImL = ∂f∂u(vcat(uinit...),θ;u0=u0) # This is BlockBidiagonal. Inverse is semiseparable!

ndigs = Diagonal(vcat(Mblks[1]...))
kdigs = BlockDiagonal(Mblks[2])
krons = Mblks[2]
uis = Mblks[3]

v = randn(size(M,2))
@test M*v ≈ ndigs*(kdigs*v)

# Exploiting the structure
cuts = [(length(d),length(ui)) for (d,ui) in zip(Mblks[1],Mblks[3])]
cut_prods = [0;cumsum(prod.(cuts))]
v_blks = [reshape(v[cut_prods[i]+1:cut_prods[i+1]],cut[1],cut[2]) for (i,cut) in enumerate(cuts)]
@test M*v ≈ ndigs*(BlockDiagonal(v_blks)*vcat(uis...))


function apply_inverse(∇Φblks,b)
    y = copy(b)
    i0, j0 = size(∇Φblks[1])
    j0 = 0
    @views for (i,blk) in enumerate(∇Φblks[2:end])
        i1,j1 = size(blk)
        mul!(y[i0+1:i0+i1], blk, y[j0+1:j0+j1], 1, 1) # We have to use y here
        j0 += j1
        i0 += i1
    end
    return y
end
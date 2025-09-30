# Butterfly matrices
N = 1000
dmax = floor(Int,log2(N))
Si = [ I + [(i == j + 2^d) for i in 1:N, j in 1:N] for d in 0:dmax]
using SparseArrays
Id = sparse(I,N,N)

Is = [collect(2^d+1:N) for d in 0:dmax]
Js = [collect(1:length(Is[d+1])) for d in 0:dmax]
Vs = [ones(length(j)) for j in Js]
Si = [I + sparse(Is[i], Js[i], Vs[i],N,N) for i in 1:dmax+1]
prod(Si)
sum(nnz, Si)
(N*(N+1))/2



N = 7
Npad = 8
Cpad = [i >= j ? 1.0 : 0.0 for i in 1:Npad, j in 1:Npad]

using SparseArrays

function butterfly_stage(N, stride)
    B = spdiagm(0 => ones(N))
    for i = stride+1:N
        B[i, i-stride] = 1.0
    end
    return B
end

N = 2^13
Npad = N
stages = []
k = Int(floor(log2(Npad)))
for d = 0:k-1
    push!(stages, butterfly_stage(Npad, 2^d))
end

function mul_stages(stages,x)
    y = deepcopy(x)
    for stage in stages
        mul!(y,stage,y)
        # y = stage*y
    end
    return y
end
v = rand(N)
T = tril(ones(N,N))
@time mul_stages(stages,v);
@time cumsum(v);
@time T*v;


k = Int(floor(log2(Npad)))
v = rand(Npad)
@time begin
y =  deepcopy(v)
for d = 0:k-1
    ystride = 2^d
    @. y[1+ystride:end] += y[1:end-ystride]
end
end
@test y ≈ cumsum(v)

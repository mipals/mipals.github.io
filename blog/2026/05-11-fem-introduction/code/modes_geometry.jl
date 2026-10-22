using LinearAlgebra, CairoMakie
## Defining geometry.
Lx = 10
Ly = 4
n = 20+1
m = 8+1
ne = (n-1)*(m-1)    # Number of linear elements
nnt = n*m           # Total number of nodes
## Plotting the elements
x = Vector(0:Lx/(n-1):Lx)
y = Vector(0:Ly/(m-1):Ly)
X = kron(x,ones(m))
Y = kron(ones(n),y)
scatter(X,Y,axis=(aspect=DataAspect(),))
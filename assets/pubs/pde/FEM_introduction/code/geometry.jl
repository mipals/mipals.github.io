# This file was generated, do not modify it. # hide
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
scatter(X,Y,aspect_ratio=1,legend=false,gridlinewidth=1,gridalpha=1,alpha=0,background=:gray,background_color_outside=:white)
xticks!(x); xlims!((0,Lx)); xlabel!("x"); 
yticks!(y); ylims!((0,Ly)); ylabel!("y");
savefig(joinpath(@OUTPUT, "2d_mesh.svg")) # hide
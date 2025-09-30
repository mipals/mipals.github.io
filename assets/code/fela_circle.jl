using LinearAlgebra
using SparseArrays
using BlockArrays
using FerriteGmsh
using FerriteViz #do/ferrite-1.0-prep
using Ferrite
using Test
import GLMakie
import Ferrite: n_components, UpdateFlags, CellIterator, celldofs

function create_disk(refinements;order=1,a=1.0,R=3.0)
    @assert a < R && (a > 0) && (R > 0) && (order < 3) && (order > 0)
    gmsh.initialize()
    gmsh.model.occ.addDisk(0.0,0.0,0.0,a,a,1)
    gmsh.model.occ.addDisk(0.0,0.0,0.0,R,R,2)
    gmsh.model.occ.cut([(2,2)],[(2,1)])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    # To get good solution quality refine the elements several times
    for _ in 1:refinements
        gmsh.model.mesh.refine()
    end
    gmsh.model.mesh.setOrder(order)
    nodes = tonodes()
    elements, _ = toelements(2)
    gmsh.finalize()
    grid = Grid(elements, nodes);
    addnodeset!(grid, "inner", x -> abs.(norm(x)-a) < 1e-6)
    addnodeset!(grid, "outer", x -> abs.(norm(x)-R) < 1e-6)
    return grid
end

### Creating disk mesh
R = 1.5 # [m] outer radius
a = 1.0 # [m] inner radius
# Radii fraction ("Lower bound limit analysis of cohesive-frictional materials using SOCP")
@test R/a == 1.5 

# Refinements for convergence. And geometric order
refinements = 3
geom_order = 1
grid = create_disk(refinements;order=geom_order,a=a,R=R)
FerriteViz.wireframe(grid)

# Lower bound element 
u_order = 1
s_order = u_order - 1
ipu = Lagrange{RefTriangle, u_order}()^2  # Interpolation order of u
ips = DiscontinuousLagrange{RefTriangle, s_order}()^3  # Interpolation order of sigma (as a vector)
ipg = Lagrange{RefTriangle, geom_order}()    # Geometric interpolation
qr = QuadratureRule{RefTriangle}(u_order+1) # Quadrature points
cvu = CellValues(qr, ipu, ipg)
cvs = CellValues(qr, ips, ipg)

# Creating the DofHandler and getting the pressure
dh = DofHandler(grid)
add!(dh, :u, ipu)
add!(dh, :sigma, ips)
close!(dh)
renumber!(dh, DofOrder.FieldWise()) # Renumber so that we get desired block structure

function assemble(B, dh, cvu, cvs)
    # This dummy rhs should not be necessary
    dummy_rhs = zeros(ndofs(dh))
    fe = zeros(ndofs_per_cell(dh)) # Dummy rhs
    assembler = start_assemble(B, dummy_rhs)
    Be = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        # Reset local matrix
        fill!(Be, 0)
        # Update cell values for current cell
        reinit!(cvu, cell)
        reinit!(cvs, cell)
        # Local dof ranges
        ur = dof_range(dh, :u)
        sr = dof_range(dh, :sigma)
        # Cell assembly
        for qp in 1:getnquadpoints(cvs)
            # Quadrature weight
            dΩ = getdetJdV(cvs, qp)
            # Assemble parts with y-test function
            # Assemble su block
            for (j, J) in pairs(ur)
                sym_grad_u = shape_symmetric_gradient(cvu, qp, j)
                for (i, I) in pairs(sr)
                    # Ni = fromvoigt(SymmetricTensor{2,2},Vector(shape_value(cvs, qp, i)))
                    Ni_tensor = SymmetricTensor{2,2,Float64}(shape_value(cvs, qp, i))
                    Be[I, J] += (sym_grad_u ⊡  Ni_tensor) * dΩ
                end
            end
            # Assemble element contribution to global matrix
        end
        # We have only assembled the off-diagonal block so the opposite block is easy to get.
        # Note this does not work we also want the diagonal blocks.
        Be = (Be + Be') 
        dofs = celldofs(cell)
        assemble!(assembler, dofs, Be, fe)
    end
    return B
end

# Getting dof for blocks
order = DofOrder.ComponentWise([1;1;2;2;2])
field_dims = map(fieldname -> n_components(dh, fieldname), dh.field_names)
target_blocks = order.target_blocks
# Collect all dofs into the corresponding block according to target_blocks
nblocks = maximum(target_blocks)
dofs_for_blocks = [Set{Int}() for _ in 1:nblocks]
component_offsets = pushfirst!(cumsum(field_dims), 0)
flags = UpdateFlags(nodes=false, coords=false, dofs=true)
for sdh in dh.subdofhandlers
    dof_ranges = [dof_range(sdh, f) for f in eachindex(sdh.field_names)]
    global_idxs = [findfirst(x -> x === f, dh.field_names) for f in sdh.field_names]
    for cell in CellIterator(dh, sdh.cellset, flags)
        cdofs = celldofs(cell)
        for (local_idx, global_idx) in pairs(global_idxs)
            rng = dof_ranges[local_idx]
            fdim = field_dims[global_idx]
            component_offset = component_offsets[global_idx]
            for (j, J) in pairs(rng)
                comp = mod1(j, fdim) + component_offset
                block = target_blocks[comp]
                push!(dofs_for_blocks[block], cdofs[J])
            end
        end
    end
end
@assert sum(length, dofs_for_blocks) == ndofs(dh)

# This should work also for continuous elements.
ndofs_u = length(dofs_for_blocks[1]) 
ndofs_s = length(dofs_for_blocks[2]) 
global_dofs_u = 1:ndofs_u
global_dofs_s = (ndofs_u + 1):ndofs(dh)

# Getting sizes of cells
n_cells = getncells(grid)
n_func_s = Ferrite.getnbasefunctions(cvs)
n_func_u = Ferrite.getnbasefunctions(cvu)

# Allocating stiffness and mass matrix
# coupling = zeros(Bool, n_func_s + n_func_u, n_func_s + n_func_u)
# pattern = ones(Bool, n_func_u, n_func_s)
# pattern[[2;4;6],1] .= false
# pattern[[1;3;5],3] .= false
# coupling[1:n_func_u,n_func_u+1:end] .= pattern
# coupling[n_func_u+1:end,1:n_func_u] .= pattern'
sp = BlockSparsityPattern([ndofs_u, ndofs_s])
add_sparsity_entries!(sp, dh)
# add_sparsity_entries!(sp, dh;coupling=coupling)
B = allocate_matrix(BlockMatrix, sp)


# Do the assembly
K = allocate_matrix(BlockMatrix, sp)
K = assemble(K, dh, cvu, cvs);
Bt = K[Block(1,2)]


################################
### Defining constraints
################################
Φ = 0.100   #
fc = 30.0   # [MPa] - Material parameter
p_ast = 6.0 # [MPa] - Optimal value (alpha? Forcing magnitude?)

# Setting up external load - Radial outwards from the "inner" circle
# Currently only workds if order of IPU is lower than order of the geometry.
# For linear geometry and quadratic ipu we can find the point by just averaging the corners?
p0 = zeros(ndofs_u) # A load case per diplacement variable
inner_set = [id for id in getnodeset(grid,"inner")]
nodes = hcat([node.x for node in grid.nodes]...)
x = nodes[1,inner_set]
y = nodes[2,inner_set]

plt = FerriteViz.wireframe(grid,linewidth=0.5,markersize=4)
plt.axis.aspect = 1
GLMakie.scatter!(x,y)
scaling = 1/(length(inner_set)) * 2π
GLMakie.arrows!(x,y,x*scaling,y*scaling,color=:red,linewidth=2)

##### Setting up optimization problem ####
using JuMP, Clarabel, MosekTools
# Shared nodes have been counted twice. Remove them
ncoords = length(unique(coords))
# We have "ncoords - 1" "lengths" between the nodes. Total length is 2πa
scaling = 1/(ncoords - 1) * 2π*a
p_average = p0 * scaling # Getting the average scaling
# Define optimization problem
model = Model(Clarabel.Optimizer)
# set_attribute(model, "direct_solve_method", :panua)
# model = Model(Mosek.Optimizer)
# set_attribute(model, "MSK_IPAR_PRESOLVE_LINDEP_USE", 0)
# Freed constraints equal to (div(n_func_s*getncells(grid),3))*2?
# Bt.nzval[abs.(Bt.nzval) .<= 5e-16] .= 0.0
# dropzeros!(Bt)
@variable(model, α)                     # Adding α as a variable
@variable(model, σ[1:ndofs_s])          # Adding σ as a variable
@objective(model, MAX_SENSE, α)         # We aim to maximize
@constraint(model, Bt*σ == p_average*α) # Loading constraint
@constraint(model, α >= 0) # Loading constraint

# For von Mises material model from Martin
θ = 30.0
c = 1.0
A = [-sind(θ) 0 -sind(θ); 1 0 -1; 0 2 0] # "Tensors" notation
b = [2*c*cosd(θ); 0; 0]
for i in 1:(div(n_func_s*getncells(grid),3))
    idx = (3(i-1)+1):3i
    @constraint(model, A*σ[idx] + b in SecondOrderCone())
end
optimize!(model)
value(α)

# Number of freed constraints:
# Seems as one can get rid of two constraints pr. element
2*getncells(grid)
# nnz before factor is a lot lower than nnz(Bt) <- Must remove constraints?


# Analytical solution
ϕ = deg2rad(θ)
ξ = tan(π/4 + ϕ/2)^2
true_sol = cot(ϕ) * ((R/a)^((ξ - 1)/ξ) - 1)
relative_error = (value(α) - true_sol)/true_sol * 100

### TODO
# - Add support for all element orders (both geoemtry and non-constant sigma)
T = [1/sqrt(2) 1/sqrt(2) 0;1/sqrt(2) -1/sqrt(2) 0; 0 0 1]



### Possible post-processing
σv = value.(σ)
tmp = zeros(size(K,1))
tmp[end-length(σv)+1:end] .= σv
symmatrix(s) = [s[1] s[2]; s[2] s[3]] 

# Computing range of eigenvalues
Sigma = empty([rand(2,2)])
for i in 1:(div(n_func_s*getncells(grid),3))
    idx = (3(i-1)+1):3i
    x = σv[idx]
    push!(Sigma,symmatrix(x))
end
emin = minimum(eigmin.(Sigma))
emax = maximum(eigmax.(Sigma))

plotter = MakiePlotter(dh, tmp)
cmap = :jet
f = GLMakie.Figure()
axs = [GLMakie.Axis(f[1, 1], title="σx"),GLMakie.Axis(f[1, 2], title="σy"),GLMakie.Axis(f[1, 3], title="τxy"),
       GLMakie.Axis(f[3, 1], title="von Mises"),GLMakie.Axis(f[3, 2], title="σ2"),GLMakie.Axis(f[3, 3], title="σ1")]
p1 = FerriteViz.solutionplot!(axs[1], plotter, process=x->x[1], colormap=cmap, field=:sigma)
p2 = FerriteViz.solutionplot!(axs[2], plotter, process=x->x[3], colormap=cmap, field=:sigma)
p3 = FerriteViz.solutionplot!(axs[3], plotter, process=x->x[2], colormap=cmap, field=:sigma)
f[2,1] = GLMakie.Colorbar(f[1,1], p1, vertical=false)
f[2,2] = GLMakie.Colorbar(f[1,2], p2, vertical=false)
f[2,3] = GLMakie.Colorbar(f[1,3], p3, vertical=false)

p4 = FerriteViz.solutionplot!(axs[4], plotter, process=x->sqrt((x[1]-x[3])^2 + 4*x[2]^2) + (x[1] + x[3])*sind(θ) - 2*c*cosd(θ), colormap=cmap, field=:sigma)
p5 = FerriteViz.solutionplot!(axs[5], plotter, process=x->eigmin(symmatrix(x)), colormap=cmap, field=:sigma,colorrange=(emin,emax))
p6 = FerriteViz.solutionplot!(axs[6], plotter, process=x->eigmax(symmatrix(x)), colormap=cmap, field=:sigma,colorrange=(emin,emax))
f[4,1] = GLMakie.Colorbar(f[3,1], p4, vertical=false)
f[4,2] = GLMakie.Colorbar(f[3,2], p5, vertical=false)
f[4,3] = GLMakie.Colorbar(f[3,3], p6, vertical=false)


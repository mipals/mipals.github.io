using BoundaryIntegralEquations, LinearAlgebra
## Physical parameters. The sphere pulsates with the normal velocity v₀.
a = 1.0             # Radius of the sphere
ρ = 1.21            # Density of air
c = 343.0           # Speed of sound
v₀ = 1.0            # Normal velocity of the surface
s = im*ρ*c          # s(α) for the time dependency exp(-iωt), i.e. α = -1
## Mesh: 464 quadratic triangles with 930 nodes
pkg_dir = pkgdir(BoundaryIntegralEquations)
mesh_file = joinpath(pkg_dir, "examples", "meshes", "sphere_1m_coarse")
mesh = load3dTriangularComsolMesh(mesh_file)
vₙ = fill(v₀, size(mesh.sources, 2))
## Solving (diag(ζ) - F)p = -s k G vₙ for each wavenumber.
# The normals of the mesh point into the sphere, so the F of the package
# is -F, and the integral free term of the exterior problem is ζ = 1 - C.
ka = range(0.05, 5, length=100)
p_bem = map(ka) do kai
    k = kai/a
    F, G, C = assemble_parallel!(mesh, k, mesh.sources; progress=false)
    H = Diagonal(1 .- C) + F
    p = H \ (-s*k*G*vₙ)
    sum(p)/length(p)    # The surface pressure is constant; average it.
end
## The analytical surface pressure of the pulsating sphere.
p_analytical = @. ρ*c*v₀*(-im*ka)/(1 - im*ka)
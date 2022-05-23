using LinearAlgebra

using Revise
using VirtuE

##-------------- Level set stuff --------------#

radius = 1 / sqrt(11) # [1/sqrt(11) gets a REAALLY bad cut]
center = [0.5, 0.5]

levelset(x) = LinearAlgebra.norm(x - center) - radius

##-------------- Problem setup --------------#

k = 1 # Polynomial degree
μ(x) = (levelset(x) < 0) * 1 + (levelset(x) >= 0) * 1
source(x) = 40 * π^2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
p_bdry(x) = 0

# u_sol(x) = [p_bdry(x) for x in eachrow(mesh.node_coords)]
# u_sol(x) = [2*sin(2*pi*x[1])*sin(4*pi*x[2]) for x in eachrow(mesh.node_coords)]
u_sol(x) = 2 * sin(2 * π * x[1]) * sin(4 * π * x[2])


##-------------- Refinement tests, in a for loop --------------#

h = [];
errors = [];
cond_nbrs = [];

for N = 2 .^ (1:7) # size of mesh
    append!(h, 1 / N)
    mesh = Meshing.create_tri_mesh(N)
    # mesh = Meshing.create_rect_mesh(N)
    # mesh = Meshing.create_pentagon_mesh()

    mesh = Meshing.remesh(mesh, levelset)

    A_0, u = Primal.Darcy_setup(mesh, source, p_bdry, μ, k)

    u_error = Primal.norm_L2(mesh, k, u, u_sol)
    append!(errors, u_error)

    areas = mesh.cell_areas
    area_ratio = minimum(areas) / maximum(areas)
    # append!(cond_nbrs,cond(Array(A_0)))
    println("Mesh size ", N, " done.")
end
convgs = log.(errors[2:end] ./ errors[1:end-1]) ./ log.(h[2:end] ./ h[1:end-1])#log(evec[i]/evec[i-1])/log(h[i]/h[i-1])

display(convgs)
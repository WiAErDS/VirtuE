using LinearAlgebra

using Revise
using VirtuE

##-------------- Level set stuff --------------#

radius = 1 / sqrt(11) # [1/sqrt(11) gets a REAALLY bad cut]
center = [0.5, 0.5]

levelset(x) = LinearAlgebra.norm(x - center) - radius

levelset(x) = x[2] - (0.5 + 1e-7)

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

# for x2 = 
for N = [5 + i for i = 1:20]#2 .^ (1:7) # size of mesh
    append!(h, 1 / N)
    mesh = Meshing.create_tri_mesh(N)
    # mesh = Meshing.create_rect_mesh(N)
    # mesh = Meshing.create_pentagon_mesh()

    # center = [x2, 0.5]
    # levelset(x) = LinearAlgebra.norm(x - center) - radius

    mesh = Meshing.remesh(mesh, levelset)

    A_0, u = Primal.Darcy_setup(mesh, source, p_bdry, μ, k)

    # Find dofs related to essential BCs
    bdry_dofs = Meshing.get_bdry_dofs(mesh)[1]
    E_bdry = Primal.create_restriction(bdry_dofs)
    E_0 = Primal.create_restriction(.!bdry_dofs)

    # Restrict system to the actual dofs
    A_0 = E_0 * A * E_0'

    areas = mesh.cell_areas
    area_ratio = minimum(areas) / maximum(areas)
    println(area_ratio)

    append!(cond_nbrs, cond(Array(A_0)))

    # A_1 = Mixed.assemble_lhs(mesh, 0)
    # append!(cond_nbrs, cond(Array(A_1)))

    Meshing.draw_mesh(mesh)
    println("Mesh size ", N, " done.")
end
# convgs = log.(errors[2:end] ./ errors[1:end-1]) ./ log.(h[2:end] ./ h[1:end-1])#log(evec[i]/evec[i-1])/log(h[i]/h[i-1])
display(maximum(cond_nbrs))

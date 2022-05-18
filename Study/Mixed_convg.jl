using LinearAlgebra
using SparseArrays

using Revise
using VirtuE

##-------------- Refinement tests, in a for loop --------------#

h = [];
u_errors = [];
p_errors = [];
cond_nbrs = [];
k = 0 # Polynomial degree

radius = 1 / sqrt(11) # radius = 1/sqrt(2)
center = [0.5, 0.5] # center = [0, 0]

# <0:inside, >0:outside
levelset(x) = norm(x - center) - radius
levelset(x) = x[2] - 0.7501

μ(x) = [1 0; 0 1]

# source(x) = 0
# p_bdry(x) = 1 - x[1]

source_scalar(x) = 40 * π^2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
p_bdry(x) = 0
u_sol(x) = -[4π * cos(2π * x[1]) * sin(4π * x[2]), 8π * cos(4π * x[2]) * sin(2π * x[1])]
p_sol(x) = 2 * sin(2 * π * x[1]) * sin(4 * π * x[2])

# source_scalar(x) = 0
# p_sol(x) = -sin(x[1])*sinh(x[2]) - (cos(1) - 1)*(cosh(1) - 1)
# u_sol(x) = [cos(x[1])*sinh(x[2]), sin(x[1])*cosh(x[2])]
# p_bdry(x) = p_sol(x)

for N = 2 .^ (1:6) # size of mesh
    # for N = 10:1:29
    append!(h, 1 / N)
    mesh = Meshing.create_tri_mesh(N)
    mesh = Meshing.remesh(mesh, levelset)

    A, _, ξ = Mixed.Darcy_setup(mesh, k, source_scalar, p_bdry, μ)

    u = ξ[1:Meshing.get_num_faces(mesh)]
    p = ξ[Meshing.get_num_faces(mesh)+1:end]

    u_error = Mixed.norm_L2(mesh, k, u, u_sol)
    p_error = Mixed.norm_L2(mesh, k, p, p_sol)
    append!(u_errors, u_error)
    append!(p_errors, p_error)

    # areas = Meshing.get_cell_areas(mesh)
    # area_ratio = minimum(areas) / maximum(areas)
    # append!(cond_nbrs,cond(Array(A)))
    println("Mesh size ", N, " done.")
end
u_convgs = log.(u_errors[2:end] ./ u_errors[1:end-1]) ./ log.(h[2:end] ./ h[1:end-1])
p_convgs = log.(p_errors[2:end] ./ p_errors[1:end-1]) ./ log.(h[2:end] ./ h[1:end-1])

display(u_convgs)
display(p_convgs)
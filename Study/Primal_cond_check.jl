using Revise
using VirtuE
using LinearAlgebra

k = 1 # Polynomial degree
h = [];
cond_nbrs = [];

for x2 = 10.0 .^ [-5, -6, -7] # location of interface line
    N = 2^2
    append!(h, 1 / N)
    # mesh = Meshing.create_tri_mesh(N)
    mesh = Meshing.create_rect_mesh(N)
    # mesh = Meshing.create_pentagon_mesh()

    levelset(x) = x[2] - (0.5 + x2)

    mesh = Meshing.remesh(mesh, levelset)

    A = Primal.assemble_stiffness_matrix(mesh, k)

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

end
display(cond_nbrs)
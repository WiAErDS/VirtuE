using Revise
using VirtuE
using LinearAlgebra

k = 1 # Polynomial degree
mesh = 0

h = []
cond_nbrs = []
cond_nbrs_expanded = []
for x2 = 10.0 .^ [-4, -5, -6] # location of interface line
    N = 2^2
    append!(h, 1 / N)
    # mesh = Meshing.create_tri_mesh(N)
    mesh = Meshing.create_rect_mesh(N)
    # mesh = Meshing.create_pentagon_mesh()

    # levelset(x) = x[2] - (0.5 + x2)
    radius = 0.25 + x2
    center = [0.5, 0.5]
    levelset(x) = norm(x - center) - radius

    mesh, itf_faces = Meshing.remesh(mesh, levelset; get_itf_faces=true)
    Meshing.draw_mesh(mesh)

    A = Primal.assemble_stiffness_matrix(mesh, k)

    # Find dofs related to essential BCs
    bdry_dofs = Meshing.get_bdry_dofs(mesh)[1]
    E_bdry = Primal.create_restriction(bdry_dofs)
    E_0 = Primal.create_restriction(.!bdry_dofs)

    # Restrict system to the actual dofs
    A_0 = E_0 * A * E_0'

    areas = mesh.cell_areas
    area_ratio = minimum(areas) / maximum(areas)
    println("Worst area ratio before expansion: ", area_ratio)
    append!(cond_nbrs, cond(Array(A_0)))

    # ------------------- Expand cells --------------------------
    itf_nodes = Meshing.get_rows_from_cols(mesh.face_nodes, itf_faces)
    conn_nodes = Meshing.get_ngbr_nodes(mesh, itf_nodes; in_interior=true, ignored_faces=itf_faces)
    mesh = Meshing.expand_cells(mesh, conn_nodes)

    A = Primal.assemble_stiffness_matrix(mesh, k)

    # Find dofs related to essential BCs
    bdry_dofs = Meshing.get_bdry_dofs(mesh)[1]
    E_bdry = Primal.create_restriction(bdry_dofs)
    E_0 = Primal.create_restriction(.!bdry_dofs)

    # Restrict system to the actual dofs
    A_0 = E_0 * A * E_0'

    areas = mesh.cell_areas
    area_ratio = minimum(areas) / maximum(areas)
    println("Worst area ratio after expansion: ", area_ratio)
    append!(cond_nbrs_expanded, cond(Array(A_0)))
end
display(cond_nbrs)
display(cond_nbrs_expanded)

plt = Meshing.draw_mesh(mesh)
# using Plots
# Plots.savefig(plt, "test.svg")


# exterior_nodes = sparsevec(Meshing.get_bdry_dofs(mesh)[1])
# itf_nodes = (abs.(mesh.face_nodes) * itf_faces .!= 0)
# itf_nodes[exterior_nodes] *= 0
# dropzeros!(itf_nodes)

# conn_faces = (abs.(mesh.face_nodes') * itf_nodes .!= 0)
# conn_faces[itf_faces] *= 0
# dropzeros!(conn_faces)

# Meshing.draw_faces_on_mesh(plt, mesh, findnz(conn_faces)[1])

# conn_nodes = (abs.(mesh.face_nodes) * conn_faces .!= 0)
# conn_nodes[itf_nodes] *= 0
# dropzeros!(conn_nodes)

# function get_rows_from_cols(A::SparseMatrixCSC, cols::SparseVector; ignored_row_entries=1)
#     rows = (abs.(A) * cols .!= 0)
#     if ignored_row_entries != 1
#         @assert(typeof(ignored_row_entries) == SparseVector{Bool,Int64})
#         rows[ignored_row_entries] *= 0
#         dropzeros!(rows)
#     end
#     return rows
# end

# exterior_nodes = sparsevec(Meshing.get_bdry_dofs(mesh)[1])
# itf_nodes = get_rows_from_cols(mesh.face_nodes, itf_faces; ignored_row_entries=exterior_nodes)
# conn_faces = get_rows_from_cols(sparse(mesh.face_nodes'), itf_nodes; ignored_row_entries=itf_faces)
# # Meshing.draw_faces_on_mesh(plt, mesh, findnz(conn_faces)[1])
# conn_nodes = get_rows_from_cols(mesh.face_nodes, conn_faces; ignored_row_entries=itf_nodes)


# function get_ngbr_nodes(grid, nodes; in_interior=false, ignored_faces=1)
#     exterior_nodes = sparsevec(Meshing.get_bdry_dofs(mesh)[1])
#     if in_interior
#         nodes[exterior_nodes] *= 0
#         dropzeros!(nodes)
#     end

#     if ignored_faces != 1
#         @assert(typeof(ignored_faces) == SparseVector{Bool,Int64})

#         conn_faces = get_rows_from_cols(sparse(grid.face_nodes'), nodes; ignored_row_entries=ignored_faces)
#         if in_interior
#             ignored_nodes = (exterior_nodes + nodes .!= 0)
#             return get_rows_from_cols(grid.face_nodes, conn_faces; ignored_row_entries=ignored_nodes)
#         else
#             return get_rows_from_cols(grid.face_nodes, conn_faces; ignored_row_entries=nodes)
#         end
#     else
#         conn_faces = get_rows_from_cols(sparse(grid.face_nodes'), nodes)
#         if in_interior
#             ignored_nodes = (exterior_nodes + nodes .!= 0)
#             return get_rows_from_cols(grid.face_nodes, conn_faces; ignored_row_entries=ignored_nodes)
#         else
#             return get_rows_from_cols(grid.face_nodes, conn_faces; ignored_row_entries=nodes)
#         end
#     end
# end

# itf_nodes = get_rows_from_cols(mesh.face_nodes, itf_faces)
# conn_nodes = get_ngbr_nodes(mesh, itf_nodes; in_interior=true, ignored_faces=itf_faces)


# function expand_cells(grid, conn_nodes)
#     conn_nodes = findnz(conn_nodes)[1]
#     new_avgd_coords = mesh.node_coords[conn_nodes, :]
#     j = 1
#     for node in conn_nodes
#         faces_to_node = mesh.face_nodes[node, :]
#         ngbr_nodes = (abs.(mesh.face_nodes) * faces_to_node .!= 0)

#         ngbr_coords = mesh.node_coords[ngbr_nodes, :]
#         new_coord = [sum(ngbr_coords[:, 1]), sum(ngbr_coords[:, 2])] / length(ngbr_coords[:, 1])
#         new_avgd_coords[j, :] = new_coord
#         j = j + 1
#     end
#     grid.node_coords[conn_nodes, :] = new_avgd_coords
#     return Meshing.Mesh(grid.node_coords, grid.cell_faces, grid.face_nodes)
# end

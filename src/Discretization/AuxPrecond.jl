module AuxPrecond

using SparseArrays
using LinearAlgebra

import ..Primal
import ..Mixed
import ..Meshing

function assemble_vector_primal_stiffness_matrix(mesh, k, μ_inv)
    A = Primal.assemble_stiffness_matrix(mesh, k, μ_inv)
    zero_mat = zeros(size(A, 1), size(A, 2))
    return [A zero_mat; zero_mat A]
end

function assemble_vector_primal_mass_matrix(mesh, k)
    M = Primal.assemble_mass_matrix(mesh, k)
    zero_mat = zeros(size(M, 1), size(M, 2))
    return [M zero_mat; zero_mat M]
end

function assemble_vector_smoother(mesh, k, μ_inv)
    A = assemble_vector_primal_stiffness_matrix(mesh, k, μ_inv)
    M = assemble_vector_primal_mass_matrix(mesh, k)

    diag_vec = diag(M + A)
    return spdiagm(1 ./ diag_vec)
end

function assemble_div_smoother(mesh, k, μ_inv)
    M = Mixed.assemble_mass_matrix(mesh, k, μ_inv)

    A_div = spzeros(size(M))
    for cell = 1:Meshing.get_num_cells(mesh)
        area = mesh.cell_areas[cell]
        faces = Meshing.get_rowvals(mesh.cell_faces, cell)
        num_faces = length(faces)
        A_div[faces, faces] += 1 / area * ones(num_faces, num_faces)
    end

    diag_vec = diag(M + A_div)
    return spdiagm(1 ./ diag_vec)
end

"""
Maps (vector) nodes to faces globally. Transpose does the opposite.
Returns a matrix of size: num_faces x 2*num_nodes
"""
function curl(mesh)
    return [mesh.face_nodes' mesh.face_nodes']
end

function assemble_div_projector_matrix(mesh)
    num_faces = Meshing.get_num_faces(mesh)
    num_nodes = Meshing.get_num_nodes(mesh)

    Π = spzeros(num_faces, 2 * num_nodes)
    for face = 1:num_faces
        face_normal = Meshing.get_face_normals(mesh, face)
        nodes = Meshing.get_rowvals(mesh.face_nodes, face) # only if node is on the face, its dof is active

        for node in nodes
            Π[face, node] += 1 / 2 * face_normal[1] #(dot(p_x, face_normal))
            Π[face, num_nodes+node] += 1 / 2 * face_normal[2] #(dot(p_y, face_normal))
        end
    end
    return Π
end

function apply_aux_precond(ξ, mesh, k, μ_inv=x -> 1)
    A_vec = assemble_vector_primal_stiffness_matrix(mesh, k, μ_inv)
    M_vec = AuxPrecond.assemble_vector_primal_mass_matrix(mesh, k)
    AA_vec = A_vec + M_vec

    S_vec = assemble_vector_smoother(mesh, k, μ_inv)
    S = assemble_div_smoother(mesh, k - 1, μ_inv)
    C = curl(mesh) # faces -> nodes
    Π = assemble_div_projector_matrix(mesh) # faces -> nodes


    out = C * S_vec * C' * ξ
    out += S * ξ
    # + CA_invC'ξ
    out += C * (AA_vec \ (C' * ξ))
    # + Pi A_inv Pi^T 
    out += Π * (AA_vec \ (Π' * ξ))
    return out
end

end
module AuxPrecond

using SparseArrays
using LinearAlgebra

import ..Primal
# import ..Monomials
import ..Meshing
# import ..NumIntegrate

function assemble_vector_primal_stiffness_matrix(mesh, k, μ_inv)
    A_1 = Primal.assemble_stiffness_matrix(mesh, k, μ_inv)
    zero_mat = zeros(size(A_1, 1), size(A_1, 2))
    return [A_1 zero_mat; zero_mat A_1]
end

function assemble_vector_primal_mass_matrix(mesh, k)
    M_1 = Primal.assemble_mass_matrix(mesh, k)
    zero_mat = zeros(size(M_1, 1), size(M_1, 2))
    return [M_1 zero_mat; zero_mat M_1]
end

function assemble_vector_smoother(mesh, k, μ_inv)
    A = assemble_vector_primal_stiffness_matrix(mesh, k, μ_inv)
    M = assemble_vector_primal_mass_matrix(mesh, k)

    diag_vec = diag(M + A)
    return spdiagm(1 ./ diag_vec)
end

function assemble_smoother(mesh, k, μ_inv)
    A = Primal.assemble_stiffness_matrix(mesh, k, μ_inv)
    M = Primal.assemble_mass_matrix(mesh, k)

    diag_vec = diag(M + A)
    return spdiagm(1 ./ diag_vec)
end

function curl(mesh)
    return mesh.face_nodes'
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

end
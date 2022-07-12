module AuxPrecond

using SparseArrays
using LinearAlgebra

import ..Primal
import ..Mixed
import ..Meshing

"""
Creates smoother out of energy or mass matrix
- E some square matrix
returns inverse of diagonal of E
"""
function create_smoother(E)
    diag_vec = diag(E)
    return spdiagm(1 ./ diag_vec)
end

"""
These functions assemble energy matrices for primal, mixed and vector primal cases
"""
function assemble_primal_energy_matrix(mesh, k, μ_inv)
    A = Primal.assemble_stiffness_matrix(mesh, k, μ_inv)
    M = Primal.assemble_mass_matrix(mesh, k)
    return A + M
end
function assemble_mixed_energy_matrix(mesh, k, μ_inv)
    M = Mixed.assemble_mass_matrix(mesh, k, μ_inv)

    # A_div = spzeros(size(M))
    # for cell = 1:Meshing.get_num_cells(mesh)
    #     area = mesh.cell_areas[cell]
    #     faces = Meshing.get_rowvals(mesh.cell_faces, cell)
    #     num_faces = length(faces)
    #     A_div[faces, faces] += 1 / area * ones(num_faces, num_faces)
    # end

    B = mesh.cell_faces'
    M_0 = Primal.assemble_mass_matrix(mesh, k, μ_inv)
    A_div = B' * (M_0 \ B)
    return A_div + M
end
function assemble_vector_primal_energy_matrix(mesh, k, μ_inv)
    A = assemble_vector_primal_stiffness_matrix(mesh, k, μ_inv)
    M = assemble_vector_primal_mass_matrix(mesh, k)
    return A + M
end

"""
These functions assemble vector primal stiffness and mass matrices
"""
function assemble_vector_primal_stiffness_matrix(mesh, k, μ_inv)
    A = Primal.assemble_stiffness_matrix(mesh, k, μ_inv)
    zero_mat = spzeros(size(A))
    return [A zero_mat; zero_mat A]
end
function assemble_vector_primal_mass_matrix(mesh, k)
    M = Primal.assemble_mass_matrix(mesh, k)
    zero_mat = spzeros(size(M))
    return [M zero_mat; zero_mat M]
end


"""
Maps nodes to faces globally. Transpose does the opposite.
Returns a matrix of size: num_faces x num_nodes
"""
function curl(mesh)
    return mesh.face_nodes'
end

"""
Creates the projector Π : vector primal -> mixed with the property that the integral of the normal component along its edge 
is unchanged under the projection
"""
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

"""
Applies the preconditioner P to Mat such that P Mat v = λ v
"""
function apply_aux_precond(Mat, mesh, k=0, μ_inv=x -> 1) # not sure how to do this with I,J,V
    S_div = create_smoother(assemble_mixed_energy_matrix(mesh, k, μ_inv))
    Π = assemble_div_projector_matrix(mesh) # vector nodes -> faces
    E_vec = assemble_vector_primal_energy_matrix(mesh, k + 1, μ_inv)
    C = curl(mesh) # nodes -> faces
    E = assemble_primal_energy_matrix(mesh, k + 1, μ_inv)
    S = create_smoother(E)

    Mat_prec = spzeros(size(Mat))
    # colcount = 1
    for (colcount, col) in enumerate(eachcol(Mat))
        col_prec = apply_aux_precond_vec(col, S_div, Π, E_vec, C, S, E)
        Mat_prec[:, colcount] = col_prec
        # colcount += 1
    end
    return Mat_prec
end
function apply_aux_precond_vec(ξ, S_div, Π, E_vec, C, S, E)
    ξ_prec = S_div * ξ
    # collect: sparse -> dense, rhs b apparently needs to be dense for A \ b
    ξ_prec += Π * (E_vec \ collect(Π' * ξ))    # + Pi A_inv Pi^T ξ
    ξ_prec += C * S * C' * ξ
    return ξ_prec + C * (E \ collect(C' * ξ))    # + C A_inv C' ξ
end

function apply_Darcy_precond(M, B, b, mesh, k=0, μ_inv=x -> 1)
    # [P 0; 0 Q]*[M -B'; B 0] = [PM -PB'; QB 0]
    M_prec = apply_aux_precond(M, mesh, k, μ_inv)
    Bt_prec = apply_aux_precond(B', mesh, k, μ_inv)

    M_0 = Primal.assemble_mass_matrix(mesh, k, μ_inv)
    Q = create_smoother(M_0)

    zero_mat = zeros(size(B, 1), size(B, 1))

    num_faces = Meshing.get_num_faces(mesh)
    b_P = apply_aux_precond(b[1:num_faces], mesh, k, μ_inv)

    return [M_prec -Bt_prec; Q*B zero_mat], Array([b_P; Q * b[num_faces+1:end]])
end

end
module AuxPrecond

using SparseArrays
using LinearAlgebra

import ..Primal
import ..Mixed
import ..Meshing

struct AuxPreconditioner
    E_p::SparseMatrixCSC
    E_div::SparseMatrixCSC
    Π::SparseMatrixCSC
    C::SparseMatrixCSC
end

"""
Constructor
"""
function AuxPreconditioner(mesh, k=0, μ_inv=x -> 1)
    E_p = assemble_primal_energy_matrix(mesh, k + 1, μ_inv)
    E_div = assemble_mixed_energy_matrix(mesh, k, μ_inv)
    Π = assemble_div_projector_matrix(mesh)
    C = curl(mesh)

    return AuxPreconditioner(E_p, E_div, Π, C)
end

"""
Apply the auxiliary space preconditioner P to a vector v
"""
function apply_aux_precond(P::AuxPreconditioner, v)
    v_prec = apply_smoother(P.E_div, v)
    v_prec += P.Π * (vector_version(P.E_p) \ collect(P.Π' * v))    # + Pi A_inv Pi^T ξ
    v_prec += P.C * apply_smoother(P.E_p, P.C' * v)
    v_prec += P.C * (P.E_p \ collect(P.C' * v))
    return v_prec
end

"""
Applies smoother out of energy or mass matrix
- E some square matrix
- v some vector
returns inverse of diagonal of E times v
"""
function apply_smoother(E, v)
    return v ./ diag(E)
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
    A_div = Mixed.assemble_divdiv_matrix(mesh, k, μ_inv)
    return A_div + M
end


"""
These functions assemble vector primal stiffness and mass matrices
"""
function vector_version(M)
    return blockdiag(M, M)
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
    face_normals = Meshing.get_face_normals(mesh)
    n_x = spdiagm(face_normals[:, 1])
    n_y = spdiagm(face_normals[:, 2])
    return [n_x * mesh.face_nodes' n_y * mesh.face_nodes']
end

"""
Applies the preconditioner P to dense matrix Mat
"""
function apply_aux_precond_to_mat(P::AuxPreconditioner, Mat::Matrix)
    M_prec = zeros(size(Mat))
    for (i, col) in enumerate(eachcol(Mat))
        M_prec[:, i] = apply_aux_precond(P, col)
    end
    return M_prec
end


"""
Apply auxiliary space preconditioner P to a mixed Darcy system
"""
function apply_Darcy_precond(P, v, k)
    num_faces = Meshing.get_num_faces(mesh)
    M_0 = Primal.assemble_mass_matrix(mesh, k, μ_inv)

    v_prec = zeros(length(v))
    v_prec[1:num_faces] = apply_aux_precond(P, v[1:num_faces])
    v_prec[num_faces+1:end] = M_0 \ v[num_faces+1:end]

    return v_prec
end


end
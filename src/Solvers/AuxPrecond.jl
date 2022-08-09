module AuxPrecond

using SparseArrays
using LinearAlgebra
import Base.:\

import ..Primal
import ..Mixed
import ..Meshing

struct AuxPreconditioner
    E_p::SparseMatrixCSC
    E_div::SparseMatrixCSC
    Π::SparseMatrixCSC
    C::SparseMatrixCSC
    M_0::SparseMatrixCSC
end

"""
Constructor
"""
function AuxPreconditioner(mesh, k=0, μ_inv=x -> 1)
    E_p = assemble_primal_energy_matrix(mesh, k + 1, μ_inv)
    E_div = assemble_mixed_energy_matrix(mesh, k, μ_inv)
    Π = assemble_div_projector_matrix(mesh)
    C = curl(mesh)
    M_0 = Primal.assemble_mass_matrix(mesh, k, μ_inv)

    return AuxPreconditioner(E_p, E_div, Π, C, M_0)
end

"""
Apply the auxiliary space preconditioner P to a vector v
"""
function apply_aux_precond(P::AuxPreconditioner, v)
    v_prec = zeros(size(v))
    v_prec += apply_smoother(P.E_div, v)
    v_prec += P.Π * (vector_version(P.E_p) \ collect(P.Π' * v))    # + Pi A_inv Pi^T ξ
    v_prec += P.C * apply_smoother(P.E_p, P.C' * v)
    v_prec += P.C * (P.E_p \ collect(P.C' * v))
    return v_prec
end

function (\)(P::AuxPreconditioner, v::AbstractVector)
    return apply_aux_precond(P, v)
end

LinearAlgebra.ldiv!(P::AuxPreconditioner, v::AbstractVector) = v .= P \ v
LinearAlgebra.ldiv!(y, P::AuxPreconditioner, v::AbstractVector) = y .= P \ v


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
These functions assemble energy matrices for primal and mixed cases
"""
function assemble_primal_energy_matrix(mesh, k, μ_inv=x -> 1)
    A = Primal.assemble_stiffness_matrix(mesh, k, μ_inv)
    M = Primal.assemble_mass_matrix(mesh, k)
    return A + M
end
function assemble_mixed_energy_matrix(mesh, k, μ_inv=x -> 1)
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
    f_n = abs.(mesh.face_nodes') / 2
    return [n_x * f_n n_y * f_n]
end # dont understand this one!

"""
Applies the preconditioner P to dense matrix Mat that scales as divdiv
"""
function apply_aux_precond_to_mat(P::AuxPreconditioner, Mat::Matrix)
    M_prec = (collect(apply_aux_precond(P, col)) for col in eachcol(Mat))
    return hcat(M_prec...)
end


"""
Apply auxiliary space preconditioner P to a vector in a mixed Darcy system
"""
function apply_Darcy_precond(P::AuxPreconditioner, v)
    num_cells = size(P.M_0)[1]

    v_prec = zeros(length(v))
    v_prec[1:end-num_cells] = apply_aux_precond(P, v[1:end-num_cells])
    v_prec[end-num_cells+1:end] = P.M_0 \ v[end-num_cells+1:end]

    return v_prec
end

"""
Applies the preconditioner P to a mixed Darcy system
"""
function apply_Darcy_precond_to_mat(P::AuxPreconditioner, Mat)
    M_prec = (collect(apply_Darcy_precond(P, col)) for col in eachcol(Mat))
    return hcat(M_prec...)
end



"""
"""
function create_smoother(E)
    return spdiagm(1 ./ diag(E))
end

"""
"""
function get_mat_for_gmres_darcy(P::AuxPreconditioner)
    P_tmp = create_smoother(P.E_div)
    P_tmp += P.Π * (vector_version(P.E_p) \ collect(P.Π'))   # + Pi A_inv Pi^T ξ
    P_tmp += P.C * create_smoother(P.E_p) * P.C'
    P_tmp += P.C * (P.E_p \ collect(P.C'))
    P_inv = inv(collect(P_tmp))
    Q_inv = inv(collect(P.M_0))
    return blockdiag(sparse(P_inv), sparse(Q_inv))
end

end
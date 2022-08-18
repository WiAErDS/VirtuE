module AuxPrecond

using SparseArrays
using LinearAlgebra
import Base: \
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
Constructor s
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
function (\)(P::AuxPreconditioner, v::AbstractVector)
    v_prec = zeros(size(v))
    v_prec += apply_smoother(P.E_div, v)
    v_prec += P.Π * (vector_version(P.E_p) \ collect(P.Π' * v))    # + Pi A_inv Pi^T ξ
    v_prec += P.C * apply_smoother(P.E_p, P.C' * v)
    v_prec += P.C * (P.E_p \ collect(P.C' * v)) # P.E_p is not diagonal, this might be expensive
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
end

struct AuxPreconditioner_Darcy
    P::AuxPreconditioner
end

"""
Constructor
"""
function AuxPreconditioner_Darcy(mesh::Meshing.Mesh, k=0, μ_inv=x -> 1)
    P = AuxPreconditioner(mesh, k, μ_inv)

    return AuxPreconditioner_Darcy(P)
end

"""
Apply auxiliary space preconditioner P to a vector in a mixed Darcy system
"""
function (\)(D::AuxPreconditioner_Darcy, v::AbstractVector)
    num_cells = size(D.P.M_0)[1]

    v_prec = zeros(length(v))
    v_prec[1:end-num_cells] = D.P \ v[1:end-num_cells]
    v_prec[end-num_cells+1:end] = D.P.M_0 \ v[end-num_cells+1:end]

    return v_prec
end

LinearAlgebra.ldiv!(D::Union{AuxPreconditioner,AuxPreconditioner_Darcy}, v::AbstractVector) = v .= D \ v
LinearAlgebra.ldiv!(y, D::Union{AuxPreconditioner,AuxPreconditioner_Darcy}, v::AbstractVector) = y .= D \ v

"""
Applies the preconditioner P to dense matrix Mat that scales as divdiv, or a mixed Darcy system
"""
function apply_precond_to_mat(P::Union{AuxPreconditioner,AuxPreconditioner_Darcy}, Mat)
    M_prec = (collect(P \ col) for col in eachcol(Mat))
    return hcat(M_prec...)
end

end

module AuxPrecondMultiplicative

using SparseArrays
using LinearAlgebra
import Base: \
import ..Primal
import ..Mixed
import ..Meshing

struct AuxPreconditionerMultiplicative
    S::SparseMatrixCSC
    E_p::SparseMatrixCSC
    E_div::SparseMatrixCSC
    M::SparseMatrixCSC
    Π::SparseMatrixCSC
    C::SparseMatrixCSC
end

"""
Constructor s
"""
function AuxPreconditionerMultiplicative(smoother_choice, mesh, k=0, μ_inv=x -> 1, problem_choice="divdiv")
    E_p = assemble_primal_energy_matrix(mesh, k + 1, μ_inv) # (u,v) + (grad u, grad v)
    E_div,M = assemble_mixed_energy_matrix(mesh, k, μ_inv)    # (u,v) + (div u, div v)
    S = assemble_smoother(smoother_choice, mesh, E_div)
    Π = assemble_div_projector_matrix(mesh)                 # Π_h
    C = curl(mesh)                                          # curl

    if problem_choice == "divdiv"
        return AuxPreconditionerMultiplicative(S, E_p, E_div, Π, C)
    else
        return AuxPreconditionerMultiplicative(S, E_p, M, Π, C)
    end
end

"""
Apply the auxiliary space preconditioner P to a vector v
"""
function (\)(P::AuxPreconditionerMultiplicative, r::AbstractVector)
    """
    Multiplicative:
    """
    z = apply_smoother(P.S, r)
    r -= P.E_div * z 
    z += P.Π * (vector_version(P.E_p) \ collect(P.Π' * r))
    r -= P.E_div * z
    z += P.C * (P.E_p \ collect(P.C' * r))
    return z
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
Creates VEM smoother
"""
function assemble_smoother(smoother_choice, mesh, E_div::SparseMatrixCSC)
    if smoother_choice == "energy"
        return E_div
    else 
        # h = maximum(mesh.cell_diams)
        face_tangents = Meshing.get_tangents(mesh)
        hm2 = sparsevec([1/norm(face_tangents[i,:])^2 for i = 1:length(face_tangents[:,1])])
        M_F = spdiagm(ones(Meshing.get_num_faces(mesh))) # h (u*n,v*n)_F (look at action on basis functions)
        return M_F * hm2 # 1/h^2 * M_F
    end
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
    return A_div + M, M
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

struct AuxPreconditionerMult_Darcy
    P::AuxPreconditionerMultiplicative
    M_0::SparseMatrixCSC
end

"""
Constructor
"""
function AuxPreconditionerMult_Darcy(smoother_choice, mesh::Meshing.Mesh, k=0, μ_inv=x -> 1)
    P = AuxPreconditionerMultiplicative(smoother_choice, mesh, k, μ_inv, "darcy")
    M_0 = Primal.assemble_mass_matrix(mesh, k, μ_inv)

    return AuxPreconditionerMult_Darcy(P, M_0)
end

"""
Apply auxiliary space preconditioner P to a vector in a mixed Darcy system
"""
function (\)(D::AuxPreconditionerMult_Darcy, v::Union{AbstractVector,SparseVector})
    num_cells = size(D.M_0)[1]

    v_prec = zeros(length(v))
    v_prec[1:end-num_cells] = D.P \ v[1:end-num_cells]
    v_prec[end-num_cells+1:end] = D.M_0 \ v[end-num_cells+1:end]

    return v_prec
end

LinearAlgebra.ldiv!(D::Union{AuxPreconditionerMultiplicative,AuxPreconditionerMult_Darcy}, v::AbstractVector) = v .= D \ v
LinearAlgebra.ldiv!(y, D::Union{AuxPreconditionerMultiplicative,AuxPreconditionerMult_Darcy}, v::AbstractVector) = y .= D \ v

"""
Applies the preconditioner P to matrix Mat that scales as divdiv, or a mixed Darcy system
"""
function apply_precond_to_mat(P::Union{AuxPreconditionerMultiplicative,AuxPreconditionerMult_Darcy}, Mat)
    mat_size = size(Mat)
    M_prec = spzeros(mat_size)
    for i = 1:mat_size[2]
        M_prec[:,i] = P \ Mat[:,i]
    end
    return M_prec
end

end

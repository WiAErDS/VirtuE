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
    Π::SparseMatrixCSC
    C::SparseMatrixCSC
end

"""
Constructor s
"""
function AuxPreconditionerMultiplicative(smoother_choice, mesh, k=0, μ_inv=x -> 1)
    E_p = assemble_primal_energy_matrix(mesh, k + 1, μ_inv) # (u,v) + (grad u, grad v)
    E_div = assemble_mixed_energy_matrix(mesh, k, μ_inv)    # (u,v) + (div u, div v)
    S = assemble_smoother(smoother_choice, mesh, E_div)
    Π = assemble_div_projector_matrix(mesh)                 # Π_h
    C = curl(mesh)                                          # curl

    return AuxPreconditionerMultiplicative(S, E_p, E_div, Π, C)
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

"""
Applies the preconditioner P to dense matrix Mat that scales as divdiv
"""
function apply_precond_to_mat(P::AuxPreconditionerMultiplicative, Mat)
    M_prec = (collect(P \ col) for col in eachcol(Mat))
    return hcat(M_prec...)
end

LinearAlgebra.ldiv!(D::AuxPreconditionerMultiplicative, v::AbstractVector) = v .= D \ v
LinearAlgebra.ldiv!(y, D::AuxPreconditionerMultiplicative, v::AbstractVector) = y .= D \ v


end

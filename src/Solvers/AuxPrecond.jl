module AuxPrecond

using SparseArrays
using LinearAlgebra
import Base: \
import ..Primal
import ..Mixed
import ..Meshing

struct AuxPreconditioner
    S::SparseMatrixCSC
    E_p::SparseMatrixCSC
    E_div::SparseMatrixCSC
    Π::SparseMatrixCSC
    C::SparseMatrixCSC
end

"""
Constructor s
"""
function AuxPreconditioner(smoother_choice, mesh, k=0, μ_inv=x -> 1)
    E_p = assemble_primal_energy_matrix(mesh, k + 1, μ_inv) # (u,v) + (grad u, grad v)
    E_div,_ = assemble_mixed_energy_matrix(mesh, k, μ_inv)    # (u,v) + (div u, div v)
    S = assemble_smoother(smoother_choice, mesh, E_div)
    Π = assemble_div_projector_matrix(mesh)                 # Π_h
    C = curl(mesh)                                          # curl

    return AuxPreconditioner(S, E_p, E_div, Π, C)
end

"""
Apply the auxiliary space preconditioner P to a vector r
"""
function (\)(P::AuxPreconditioner, r::AbstractVector)
    z = apply_smoother(P.S, r)
    z += P.Π * (vector_version(P.E_p) \ collect(P.Π' * r))     # + Pi A_inv Pi^T ξ
    z += P.C * (P.E_p \ collect(P.C' * r))                     # (P.E_p is not diagonal, this might be expensive)
    # z += P.C * apply_smoother(P.E_p, P.C' * r)
    return z
end

"""
Applies smoother out of energy or mass matrix
- S some square matrix
- v some vector
returns inverse of diagonal of S times v
"""
function apply_smoother(S, v)
    return v ./ diag(S)
end

"""
Creates VEM smoother
"""
function assemble_smoother(smoother_choice, mesh, E_div::SparseMatrixCSC)
    if smoother_choice == "energy"
        return E_div
    else 
        # h = maximum(mesh.cell_diams)
        # I_F = spdiagm(ones(Meshing.get_num_faces(mesh))) # h (u*n,v*n)_F (look at action on basis functions)
        # return 1/h^2 * M_F

        face_tangents = Meshing.get_tangents(mesh)
        facenorm_sq_reciprocals = sparsevec([1/norm(face_tangents[i,:])^2 for i = 1:length(face_tangents[:,1])])
        return spdiagm(facenorm_sq_reciprocals)
    end
end

"""
These functions assemble energy matrices for primal and mixed cases
"""
function assemble_primal_energy_matrix(mesh, k, μ_inv=x -> 1)
    M = Primal.assemble_mass_matrix(mesh, k)
    A = Primal.assemble_stiffness_matrix(mesh, k, μ_inv)
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

# """
# Maps faces to elements globally. Transpose does the opposite.
# Returns a matrix of size: num_elements x num_faces
# """
# function div(mesh)
#     return mesh.cell_faces' 
# end

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
    M_0::SparseMatrixCSC
end

"""
Constructor
"""
function AuxPreconditioner_Darcy(smoother_choice, mesh::Meshing.Mesh, k=0, μ_inv=x -> 1)
    P = AuxPreconditioner(smoother_choice, mesh, k, μ_inv)
    M_0 = Primal.assemble_mass_matrix(mesh, k, μ_inv)       # mass matrix for Darcy system

    return AuxPreconditioner_Darcy(P, M_0)
end

"""
Apply auxiliary space preconditioner P to a vector in a mixed Darcy system
"""
function (\)(D::AuxPreconditioner_Darcy, v::Union{AbstractVector,SparseVector})
    num_cells = size(D.M_0)[1]

    v_prec = zeros(length(v))
    v_prec[1:end-num_cells] = D.P \ v[1:end-num_cells]
    v_prec[end-num_cells+1:end] = D.M_0 \ v[end-num_cells+1:end]

    return v_prec
end

LinearAlgebra.ldiv!(D::Union{AuxPreconditioner,AuxPreconditioner_Darcy}, v::AbstractVector) = v .= D \ v
LinearAlgebra.ldiv!(y, D::Union{AuxPreconditioner,AuxPreconditioner_Darcy}, v::AbstractVector) = y .= D \ v

"""
Applies the preconditioner P to matrix Mat that scales as divdiv, or a mixed Darcy system
"""
# function apply_precond_to_mat(P::Union{AuxPreconditioner,AuxPreconditioner_Darcy}, Mat)
#     mat_size = size(Mat)
#     M_prec = spzeros(mat_size)

#     for i = 1:mat_size[2]
#         M_prec[:,i] = P \ Mat[:,i]
#         if i%floor(Int,mat_size[2]/6)==0
#             print(" ; iter = $i of num_cols = ", mat_size[2])    
#         end
#     end
#     return M_prec
# end
function apply_precond_to_mat(P::Union{AuxPreconditioner,AuxPreconditioner_Darcy}, Mat)
    mat_size = size(Mat)
    num_threads = Threads.nthreads()
    num_cols = mat_size[2]
    
    # Calculate the range of columns each thread will handle
    # LinRange is used to ensure all columns are covered even when num_cols is not divisible by num_threads
    column_ranges = floor.(Int, LinRange(1, num_cols, num_threads + 1))
    
    # Initialize a container for the thread-local matrices
    local_mats = Vector{SparseMatrixCSC{Float64,Int64}}(undef, num_threads)

    Threads.@threads for tid in 1:num_threads
        start_col = column_ranges[tid]
        end_col = column_ranges[tid + 1] - 1
        # Adjust the end_col for the last thread to ensure it includes the last column if num_cols isn't evenly divisible
        if tid == num_threads
            end_col = num_cols
        end
        # Pre-allocate the size for each thread's local matrix
        local_mats[tid] = spzeros(mat_size[1], end_col - start_col + 1)

        for i = start_col:end_col
            # Process each column and place it directly in the correct position within the thread's local matrix
            local_mats[tid][:, i - start_col + 1] = P \ Mat[:,i]
            # if i % floor(Int, num_cols / 6) == 0 || i == end_col # Print progress at specified intervals or at the end of each thread's range
            #     println(" ; iter = $i of num_cols = $num_cols on thread = $tid")
            # end
        end
        println("thread $tid done!")
    end

    # Concatenate all thread-local matrices into a single matrix
    M_prec = hcat(local_mats...)
    return M_prec
end



end

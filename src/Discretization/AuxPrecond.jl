module AuxPrecond

using SparseArrays
using LinearAlgebra

import ..Primal
# import ..Monomials
# import ..Meshing
# import ..NumIntegrate

function assemble_vector_primal_stiffness_matrix(mesh, k, μ_inv)
    A_1 = Primal.assemble_stiffness_matrix(mesh, k, μ_inv)
    zero_mat = zeros(size(A_1, 1), size(A_1, 2))
    return [A_1 zero_mat; zero_mat A_1]
end

function curl(mesh)
    return mesh.face_nodes'
end

function assemble_vector_primal_mass_matrix(mesh, k)
    M_1 = Primal.assemble_mass_matrix(mesh, k)
    zero_mat = zeros(size(M_1, 1), size(M_1, 2))
    return [M_1 zero_mat; zero_mat M_2]
end

function assemble_smoother(mesh, k, μ_inv)
    A = assemble_vector_primal_stiffness_matrix(mesh, k, μ_inv)
    M = assemble_vector_primal_mass_matrix(mesh, k)
    return diag(M + A) # TODO sparse diag matrix
end

end
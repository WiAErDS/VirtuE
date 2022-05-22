module AuxPrecond

using SparseArrays
using LinearAlgebra

import ..Monomials
import ..Meshing
import ..NumIntegrate

import Primal

function assemble_vector_primal_stiffness_matrix(mesh, k, μ_inv)
    I = Int[]
    J = Int[]
    V = Float64[]

    for (cell, nodes) in enumerate(mesh.cell_nodes) # Loops over mesh elements
        Proj, PreProj, G = Primal.element_projection_matrices(mesh, cell, k)

        if μ(mesh.cell_centroids[cell,:]) != 1
            G = Monomials.scaled_element_stiffness_matrix(mesh, cell, k, μ_inv)
        end

        K_el = Primal.element_stiffness_matrix(Proj, PreProj, G)

        append!(I, repeat(nodes, length(nodes)))
        append!(J, repeat(nodes, inner=length(nodes)))
        append!(V, vec(K_el))
    end

    A_1 = sparse(I, J, V)
    A_2 = sparse(I, J, V)
    zero_mat = zeros(size(A_1,1),size(A_1,2))

    return [A_1 zero_mat; zero_mat A_2]
end

end
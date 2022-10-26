module Mixed

using SparseArrays
using LinearAlgebra

import ..Monomials
import ..Meshing
import ..NumIntegrate
import ..Primal # needed only for assemble_divdiv_matrix

#-------------- Mixed VEM k=0 --------------#
function Darcy_setup(mesh, k, source::Function, p_naturalBC::Function, μ_inv::Function=x -> 1)
    A = assemble_lhs(mesh, k, μ_inv)
    b = assemble_rhs(mesh, k, source, p_naturalBC)

    ξ = A \ b

    return A, b, ξ
end

function element_projection_matrices(mesh, cell, k)

    # geometry information
    faces, orient_vals = findnz(mesh.cell_faces[:, cell])

    h = mesh.cell_diams[cell]
    centroid = mesh.cell_centroids[cell, :]
    normals = Meshing.get_face_normals(mesh, faces)
    fc = Meshing.get_face_centers(mesh, faces)

    # Scaled monomials
    monExps = Monomials.mon_exp(k + 1)[2:end, :]
    grads = [Monomials.eval_grad_mon(centroid, centroid, h, α) for α in eachrow(monExps)]
    grads = hcat(grads...)

    D = normals * grads

    C = [Monomials.eval_scaled_mon(c, centroid, h, α) * orient_val for α in eachrow(monExps), (c, orient_val) in zip(eachrow(fc), orient_vals)]

    G = C * D

    PreProj = G \ C # Does quick inv(G)*C
    Proj = D * PreProj

    return Proj, PreProj, G
end

function element_mass_matrix(Proj, PreProj, G)
    # TODO: VEM param before (I-Proj)
    return PreProj' * G * PreProj + (I - Proj)' * (I - Proj)
end

function assemble_mass_matrix(mesh, k, μ_inv=x -> 1)
    @assert(k == 0, "Only implemented k=0")

    I = Int[]
    J = Int[]
    V = Float64[]

    for cell in 1:Meshing.get_num_cells(mesh) # Loops over mesh elements
        faces = mesh.cell_faces[:, cell].nzind

        Proj, PreProj, G = element_projection_matrices(mesh, cell, k)

        if μ_inv(mesh.cell_centroids[cell, :]) != 1
            G = Monomials.scaled_element_stiffness_matrix(mesh, cell, k + 1, μ_inv)
            G = G[2:end, 2:end]
        end

        K_el = element_mass_matrix(Proj, PreProj, G)

        append!(I, repeat(faces, length(faces)))
        append!(J, repeat(faces, inner=length(faces)))
        append!(V, vec(K_el))
    end

    return sparse(I, J, V)
end

function assemble_divdiv_matrix(mesh, k, μ_inv=x -> 1)
    @assert(k == 0, "Only implemented k=0")

    B = mesh.cell_faces'
    M_0 = Primal.assemble_mass_matrix(mesh, k, μ_inv)

    return B' * (M_0 \ B)
end

function assemble_lhs(mesh, k, μ_inv=x -> 1)
    @assert(k == 0, "Only implemented k=0")

    M = assemble_mass_matrix(mesh, k, μ_inv)
    B = mesh.cell_faces'

    zero_mat = zeros(size(B, 1), size(B, 1))

    return [M -B'; B zero_mat]
end

function assemble_rhs(mesh, k, source, p_naturalBC)
    @assert(k == 0, "Only implemented k=0")

    b = zeros(Meshing.get_num_cells(mesh) + Meshing.get_num_faces(mesh))

    bdry_dofs = Meshing.get_bdry_dofs(mesh)[2] # Gets the face dofs
    for face in findnz(bdry_dofs)[1] # Loops over boundary faces
        orient = -mesh.cell_faces[face, :].nzval[1] # Is there a nicer way to remove brackets around a singleton [x]?
        b[face] += orient * p_naturalBC(Meshing.get_face_centers(mesh, face))
    end

    for cell in 1:Meshing.get_num_cells(mesh) # Loops over mesh elements
        b[cell+Meshing.get_num_faces(mesh)] += Meshing.get_poly_area(mesh, cell) * source(Meshing.get_poly_centroid(mesh, cell))
    end

    return b # can't A\b for sparse vectors
end

""" Computes L2 norm of a function
- vem_dofs is a vector of dofs from the computed solution
- sol is the true solution in function form
"""
function norm_L2(mesh, k, vem_dofs, sol::Function)
    @assert(k == 0, "Only implemented k=0")

    sum² = 0
    if length(vem_dofs) == Meshing.get_num_cells(mesh)
        for (cell, nodes) in enumerate(mesh.cell_nodes)
            coords = mesh.node_coords[nodes, :]
            integrand(x) = (sol(x) - vem_dofs[cell])^2

            sum² += NumIntegrate.quad_integral_el(coords, integrand, k + 2)
        end
    elseif length(vem_dofs) == Meshing.get_num_faces(mesh)
        for (cell, nodes) in enumerate(mesh.cell_nodes)
            coords = mesh.node_coords[nodes, :]
            pC = mesh.cell_centroids[cell, :]
            h = mesh.cell_diams[cell]
            faces = mesh.cell_faces[:, cell].nzind

            PreProj = element_projection_matrices(mesh, cell, k)[2] # Preproj: projects the VEM space to a subspace
            coef_list = PreProj * vem_dofs[faces]
            vem_sol(x) = Monomials.eval_grad_polynomial(x, coef_list, pC, h, k + 1)

            integrand(x) = dot(sol(x) - vem_sol(x), sol(x) - vem_sol(x))
            sum² += NumIntegrate.quad_integral_el(coords, integrand, k + 2)
        end
    end
    return sqrt(sum²)
end

end
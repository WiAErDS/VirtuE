module Primal

using SparseArrays
using LinearAlgebra

import ..NumIntegrate
import ..Monomials
import ..Meshing

function Darcy_setup(mesh,source::Function,p_bdry::Function,μ::Function,k::Int)
    A = assemble_stiffness_matrix(mesh, k, μ)
    b = assemble_rhs(mesh,source, k)

    # Find dofs related to essential BCs
    bdry_dofs = Meshing.get_bdry_dofs(mesh)[1]
    E_bdry = create_restriction(bdry_dofs)
    E_0    = create_restriction(.!bdry_dofs)

    ξ_bdry = [p_bdry(x) for x in eachrow(E_bdry * mesh.node_coords)]

    # Restrict system to the actual dofs
    A_0 = E_0 * A * E_0'
    b_0 = E_0 * (b - A * E_bdry' * ξ_bdry)

    # Solve
    ξ_0 = A_0 \ b_0

    # Expand to full vector
    ξ = E_0' * ξ_0 + E_bdry' * ξ_bdry
    return A_0, ξ
end

#-------------- VEM k=1 --------------#
function element_projection_matrices(mesh, cell, k=1)

    @assert(k==1, "Only implemented k=1")
    monExps = Monomials.mon_exp(k)

    # geometry information
    coords = mesh.node_coords[mesh.cell_nodes[cell],:]
    nvx = size(coords, 1) # number of vertices
    d_vec = circshift(coords, 1) - circshift(coords, -1)
    d_perp = 1/2 * ([0 -1; 1 0] * d_vec')

    h = mesh.cell_diams[cell]
    centroid = mesh.cell_centroids[cell, :]

    D = [Monomials.eval_scaled_mon(coord, centroid, h, exps) for coord in eachrow(coords), exps in eachrow(monExps)]

    gradMon = [Monomials.grad_mon(centroid, centroid, h, exps)' for exps in eachrow(monExps[2:end, :])] #create grads
    gradMon = vcat(gradMon...) # reshape to 2D array

    B = [ones(1, nvx) / nvx; gradMon * d_perp]

    G = B*D

    PreProj = G\B # As a rule: inv(G)*B is superslow.
    Proj = D*PreProj

    return Proj, PreProj, G
end

function element_stiffness_matrix(Proj, PreProj, G)
    Gtilde = copy(G)
    Gtilde[1,:] .= 0
    # TODO: VEM param before (I-Proj)
    return PreProj'*Gtilde*PreProj + (I-Proj)'*(I-Proj)
end

function assemble_stiffness_matrix(mesh, k, μ = x->1)

    I = Int[]
    J = Int[]
    V = Float64[]

    for (cell, nodes) in enumerate(mesh.cell_nodes) # Loops over mesh elements
        Proj, PreProj, G = element_projection_matrices(mesh, cell, k)

        if μ(mesh.cell_centroids[cell,:]) != 1
            G = Monomials.scaled_element_stiffness_matrix(mesh, cell, k, μ)
        end

        K_el = element_stiffness_matrix(Proj, PreProj, G)

        append!(I, repeat(nodes, length(nodes)))
        append!(J, repeat(nodes, inner=length(nodes)))
        append!(V, vec(K_el))
    end

    return sparse(I, J, V)
end

function assemble_rhs(mesh, source, k)

    @assert(k==1, "Only implemented k=1: [ADD REASON WHY]")

    I = Int[]
    V = Float64[]

    for (cell, nodes) in enumerate(mesh.cell_nodes)
        PreProj = element_projection_matrices(mesh, cell, k)[2]
        coords = mesh.node_coords[nodes,:]
        pC = mesh.cell_centroids[cell, :]
        h = mesh.cell_diams[cell]
        f_el = zeros(length(nodes))

        for i in 1:length(nodes)
            integrand(x) = source(x) * Monomials.eval_scaled_polynomial(x,PreProj[:,i],pC,h,k)
            f_el[i] = NumIntegrate.quad_integral_el(coords, integrand, 2*k)
        end

        append!(I, nodes)
        append!(V, f_el)

        # Old way accurate for k=0 (Π^∇ = P_0):
        # f_el = source(pC) * area / length(nodes_el)
        # append!(I, nodes_el)
        # append!(V, repeat([f_el], length(nodes_el)))
    end

    J = ones(Int, length(I))

    return sparse(I, J, V)
end

function create_restriction(dofs)
    I = 1:sum(dofs)
    J = findall(vec(dofs))
    V = trues(sum(dofs))

    return sparse(I, J, V, sum(dofs), length(dofs))
end

""" Projects the basis functions of an element to L2 """
function loc_proj_L2(coords) # Obsolete?

    #@assert(k==1, "Only implemented k=1: Π_0 = Π^∇")

    # Π^∇φ_i = 1/2|E|(x-V)*d+1/N^V

    nvx = size(coords, 1) # number of vertices
    d_vec = circshift(coords, 1) - circshift(coords, -1)
    d_perp = (1/2 * [0 -1; 1 0] * d_vec')' # 3x2 adjoint
    @assert(size(d_perp,1)>size(d_perp,2))

    centroid = [sum(coords[:, i])/nvx for i in 1:2]
    area = get_poly_area(coords)

    proj_collection(x,idx) = 1/(2*area)*dot(x-centroid,d_perp[idx,:])+1/nvx
    return proj_collection
end


""" Computes L2 norm of a function
- vem_dofs is a vector of dofs from the computed solution
- sol is the true solution in function form
"""
function norm_L2(mesh, k, vem_dofs, sol::Function)
    @assert(k==1, "Only implemented k=1")

    sum² = 0
    for (cell, nodes) in enumerate(mesh.cell_nodes)
        coords = mesh.node_coords[nodes,:]
        pC = mesh.cell_centroids[cell, :]
        h = mesh.cell_diams[cell]

        PreProj = element_projection_matrices(mesh, cell, k)[2] # Preproj: projects the VEM space to a subspace
        polynomial_list = PreProj*vem_dofs[nodes]

        vem_sol(x) = Monomials.eval_scaled_polynomial(x,polynomial_list,pC,h,k)
        
        integrand(x) = (sol(x)-vem_sol(x))^2

        sum² += NumIntegrate.quad_integral_el(coords, integrand, 3)

        #--- Special node quadrature
        # findrow(row,mat) = findfirst(i->all(j->row[j] == mat[i,j],1:size(mat,2)),1:size(mat,1))
        # coord_2_node(x) = findrow(x,coords)
        # integrand(x) = (sol(x)-vem_dofs[nodes[coord_2_node(x)]])^2
        # sum² += quad_integral_el(coords, integrand, -3) # -3 gives vertex quadrature
    end

    return sqrt(sum²)
end

end
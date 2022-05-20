module VEMPrimalHigh

import ..NumIntegrate
import ..Monomials
import ..DOFPrimal
# Since Monomials is a module, one can call its functions as Monomials.scaled_mod...

#-------------- VEM k>1 --------------#
function element_projection_matrices(mesh, cell, k)
    monExps = Monomials.mon_exp(k)
    nmon = size(monExps, 1) # n_k = (k+1)*(k+2)/2

    # geometry information
    coords = mesh.node_coords[mesh.cell_nodes[cell], :]
    nvx = size(coords, 1) # number of vertices
    edgeEnds = [[circshift(coords, -1)[i, :], circshift(coords, -2)[i, :]] for i in 1:nvx] # list of 2 element arrays of 2 elements
    # d_vec = circshift(coords, 1) - circshift(coords, -1)
    # d_perp = 1/2 * ([0 -1; 1 0] * d_vec')

    h = mesh.cell_diams[cell]
    centroid = mesh.cell_centroids[cell, :]
    area = mesh.cell_areas[cell]

    # D matrix computation <- dofs, geometry, monomial_j(p)
    mon((p, j)) = Monomials.scaled_mon(p, centroid, h, monExps[j, :])
    Dvtx = [DOFPrimal.vertex_DOF(mon ∘ (p -> (p, j)), coords) for j in 1:nmon]
    Dedge = [DOFPrimal.edge_DOF(mon ∘ (p -> (p, j)), k, locEdgePoints) for locEdgePoints in edgeEnds, j in 1:nmon]
    Dmom = [DOFPrimal.moment_DOF(mon ∘ (p -> (p, j)), k, coords, area, centroid, h) for j in 1:nmon] # are they ordered so that this becomes correct??
    Dvtx = hcat(Dvtx...) # nbVertex X nbMonomials
    Dedge = hcat([vcat(Dedge[:, i]...) for i = 1:size(Dedge, 2)]...) # nbQuads X nbMonomials
    Dmom = hcat(Dmom...) # nbMoments X nbMonomials
    D = vcat(Dvtx, Dedge, Dmom)

    # gradMon = [Monomials.grad_mon(coords[1,:], centroid, h, monExps[i, :])' for i in 2:nmon] #create grads
    # gradMon = vcat(gradMon...) # reshape to 2D array

    # B matrix computation:
    ndof = size(D, 1) #Int(nvx*k+(k-1)*k/2)
    ∫Δmv = zeros(nmon, ndof)
    ∫∇m_nv = zeros(nmon, ndof)
    for i = 1:nmon
        α = monExps[i, :] # gets monomial i, eg (1,0)
        Δ_struct = Monomials.Δ_mon_coef(h, α) # gets coef of Δmon, eg α=[1,1] => Δ_struct empty
        if isempty(Δ_struct)
            continue
        end
        Δ_coef = [Δ_struct[i][1] for i = 1:length(Δ_struct)] # Δ_struct = [(coef_x, [β_x1,β_x2]), (coef_y, [β_y1,β_y2])]
        β_idx = [Monomials.idx_mon(Δ_struct[i][2]) for i = 1:length(Δ_struct)] # idx_mon eg (0,0) -> 1, (1,0) -> 2, (0,2) -> 6
        for j = 1:ndof
            for m = 1:length(Δ_struct)
                ∫Δmv[β_idx[m], j] += area * Δ_coef[m] * DOFPrimal.BF_on_DOF(k * nvx + β_idx[m], j) # [think this is wrong somehow...]
            end
            for edge_idx = 1:length(edgeEnds) # entry [i,j] gets incremented by ∇mon(quad)⋅n if j==dof_idx_at(quad)
                locEdgeEnds = edgeEnds[edge_idx]
                a = locEdgeEnds[1]
                b = locEdgeEnds[2]
                normal = [0 -1; 1 0] * (a - b) ./ norm(a - b)

                mapToDOF(m) = DOFPrimal.local_map_quad_to_dof(m, edge_idx, locEdgeEnds, nvx, k, mesh.node_coords)
                ∇m_nv((p, m)) = dot(Monomials.grad_mon(p, centroid, h, α), normal) * DOFPrimal.BF_on_DOF(j, mapToDOF(m))

                ∫∇m_nv[i, j] += NumIntegrate.edge_integral(∇m_nv, k, locEdgeEnds)
            end
        end
    end
    B = -∫Δmv + ∫∇m_nv

    G = B * D

    # PreProj = G\B
    # Proj = D*PreProj

    return D, ∫∇m_nv#Proj, PreProj, G # D,B for debug
end


end

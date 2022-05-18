module DOFPrimal

# using LinearAlgebra # must have the used packages INSIDE module
# using CalculusWithJulia
# using Images
using FastGaussQuadrature # (might need to Pkd.add)
include("Monomials.jl")
include("../Quadrature/NumIntegrate.jl")

#export vertexDOF,edgeDOF,triDOF

""" Evaluates the global vertex DOFs of v
- v is the function v: R^2 -> R
- coords is the nvx*2 coordinate matrix of the points of an element
"""
function vertex_DOF(v,coords)
    return [v([coord[1],coord[2]]) for coord in eachrow(coords)]
end

""" Evaluates v on the interior Gauss-Lobatto quadrature nodes of an edge
- v is the function
- k>1 is the degree of the monomial space, k+1=nbrGaussLobattoNodes
- edgeEnds is 2 element array of end point 2 element arrays of coordinates
"""
function edge_DOF(v,k,edgeEnds)
    a = edgeEnds[1] # first end point [x,y]
    b = edgeEnds[2]
    edgeMap(t) = 1/2*([b-a [0,1]]*[t,0]+(a+b)) # maps a point p=[t,0] on [-1,1] to a global edge defined by a and b

    nodes,_ = gausslobatto(k+1) # uses FastGaussQuadrature, includes edge points
    return [v(edgeMap(t)) for t in nodes[2:end-1]]
end

""" Integrates the moment up to order k-2. Uses NumIntegrate.jl
- v is the function
- k>1 is the degree of the monomial/poly space, poly's of deg <= k are included
- coords are the coordinates of the vertices/nodes
"""
function moment_DOF(v,k,coords,area,p_c,h)
    @assert(k>1, "No moment dofs for case k=1")
    K = k-2
    αList = Monomials.mon_exp(K) # uses Monomials
    n_K = size(αList,1) # (K+1)*(K+2)/2 = dim of polynomial space

    dofv_list = zeros(n_K)
    for i in 1:n_K
        α = αList[i,:]; mon_α(p) = Monomials.eval_scaled_mon(p,p_c,h,α)
        fun(p) = v(p)*mon_α(p)/area

        # want to integrate K exactly -> quad order 2q-1=K
        dofv_list[i] = NumIntegrate.quad_integral_el(coords, fun, 1+Int(floor((1+K)/2)))
    end
    return dofv_list
end

# include("../Meshing/Meshing.jl") # [needed for get_node_from_coord... ugly, how to fix?]
# """ Maps edge quadrature points to their respective dof's, correctly ordered as vertex dofs, edge dofs
# - quad_idx is the idx of the quad s.t. eg quads[idx] = -0.7
# - edge_idx is the idx of the edge in the order they appear inside edgeEnds (in VEMPrimal)
# - edge_ends is the local end points of the edge with idx edge_idx
# - nvx is #vertices
# returns an integer being the dof number of quad_idx of edge_idx
# """
# function local_map_quad_to_dof(quad_idx, edge_idx, edge_ends, nvx, k, node_coords)
#     if quad_idx == 1
#         return get_node_from_coord(edge_ends[1],node_coords)
#     elseif quad_idx == k+1
#         return get_node_from_coord(edge_ends[2],node_coords)
#     else
#         return nvx+(k-1)*(edge_idx-1)+quad_idx
#     end
# end

""" Evaluates the kronecker delta δ_ij
"""
function BF_on_DOF(i,j)
    return Int(i==j)
end

end

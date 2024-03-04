module NumIntegrate

using MiniQhull
using LinearAlgebra
using FastGaussQuadrature # for edge_integral

import ..Meshing # for get_poly_area
import ..Gaussquad_table

function get_poly_area(vertices) # @wikipedia polygon
    v1 = vertices
    v2 = circshift(vertices, -1)
    return 1 / 2 * abs(dot(v1[:, 1], v2[:, 2]) - dot(v2[:, 1], v1[:, 2]))
end

""" Maps p linearly from the reference triangle to the global triangle """
function from_ref_map(coords_tri, p)
    p0 = coords_tri[1, :] # gives P[:,loc2glb[i]] =eg array([0 0.125])
    p1 = coords_tri[2, :]
    p2 = coords_tri[3, :]

    F_T = [p1[1]-p0[1] p2[1]-p0[1]; p1[2]-p0[2] p2[2]-p0[2]]
    return F_T * p + p0
end

""" Integrates f over triangle with vertices glob_coords_tri according
    to Gauss-Legendre rule order n """
function integrate_tri(glob_coords_tri, f::Function, n::Int)
    quads = Gaussquad_table.get_corner_tridata()
    if n != -3
        quads = Gaussquad_table.get_gauss_tridata(n)
    end
    Δ = get_poly_area(glob_coords_tri) # triangle area
    # quad[3] is the weight, quad[1:2] = [x, y]
    result = Δ * sum(quad[3] * f(from_ref_map(glob_coords_tri, quad[1:2]))
                     for quad in eachrow(quads)) # eachrow returns col vecs
    return result
end

""" Integrates f over general polygon """
function quad_integral_el(coords, fun::Function, quad_order::Int)
    ∫fun = 0.0
    if size(coords, 1) == 3 # if the element is just a triangle
        ∫fun = integrate_tri(coords, fun, quad_order)
    else
        triangulation = MiniQhull.delaunay(Array(coords'))' # ' transposes the vector
        for k in axes(triangulation, 1) # add contributions from each triangle
            coords_tri = coords[triangulation[k, :], :]
            ∫fun += integrate_tri(coords_tri, fun, quad_order)
        end
    end
    return ∫fun
end

""" Integrates v on the edge specified by edgeEnds
- v is a function v: coord -> R
- k>=1 is the degree of the monomial space, k+1=nbrGaussLobattoNodes
- edgeEnds is (node x dim) = 2x2 matrix of edge end points
"""
function edge_integral(v, k, edgeEnds)
    a = edgeEnds[1, :]
    b = edgeEnds[2, :]
    edgeMap(t) = 1 / 2 * ([b - a [0, 1]] * [t, 0] + (a + b)) # maps a point p=[t,0] on [-1,1] to a global edge defined by a and b

    quads, weights = gausslobatto(k + 1) # uses FastGaussQuadrature, includes edge points
    return sum([v(edgeMap(quads[m])) * weights[m] for m = 1:k+1])
end

end
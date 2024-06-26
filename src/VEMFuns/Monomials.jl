module Monomials

import ..Meshing
#export ScaledMon,GradMon,MonExp

"""Scaled monomial of degree |α|
- p=[x y] is the point in R^n
- p_c=[x_c y_c] is the centroid of the space
- h is the diameter of the space
- α is the multiindex exponent
"""
function eval_scaled_mon(x, p_c, h, α)
    m = ((x - p_c) / h) .^ α
    return prod(m)
end

function scaled_mon(grid::Meshing.Mesh, cell::Int, α)
    p_c = grid.cell_centroids[cell, :]
    h = grid.cell_diams[cell]
    return x -> eval_scaled_mon(x, p_c, h, α)
end

""" Gradient of scaled monomial of degree |α|
- p=[x y] is the point in R^n
- p_c=[x_c y_c] is the centroid of the space
- h is the diameter of the space
- α is the multiindex exponent OF THE MONOM TO DIFFERENTIATE
"""
function eval_grad_mon(x, p_c, h, α)
    m_1 = zeros(length(x))
    for i in 1:length(x)
        m_1[i] = α[i] / h * eval_scaled_mon(x, p_c, h, reduce_exp(α, i))
    end
    return m_1
end

function reduce_exp(α, i)
    α_copy = copy(α)
    α_copy[i] = max(α[i] - 1, 0) # reduce exponent of deriv by 1
    return α_copy
end

function grad_mon(grid::Meshing.Mesh, cell::Int, α)
    p_c = grid.cell_centroids[cell, :]
    h = grid.cell_diams[cell]
    return x -> eval_grad_mon(x, p_c, h, α)
end

""" Coefficients of Laplace-op on monomial of degree |α|
- h is the diameter of the space
- α is the multiindex exponent OF THE MONOM TO DIFFERENTIATE
returns a list of 2 tuples (coef, [β1,β2]) one for each dimension s.t.
Δmon = ∑ coef mon_[β1,β2] =_(dim=2) coef_x mon_[β_x1,β_x2] + coef_y mon_[β_y1,β_y2]
"""
function Δ_mon_coef(h, α) # note that α[i] or α[i]-1 might be 0
    # create a matrix nbAlpha x dim, of the resulting multiindex exponents of each summand
    # Δ(α,i) = (α[i]-2)<0 ? zeros(length(α)) : [i==j ? (α[j]-2>0)*(α[j]-2) : α[j] for j in 1:length(α)]
    Δ(α, i) = [i == j ? α_j - 2 : α_j for (j, α_j) in enumerate(α)]
    β = [Δ(α, i) for i in eachindex(α)] # [alpha1,alpha2] of summands along dimension i=1,2

    nonzero_coef(α_i) = α_i * (α_i - 1) != 0
    return [(α[i] * (α[i] - 1) / h^2, β[i]) for i in 1:length(α) if nonzero_coef(α[i])]
end

""" mon_exp(deg) generates the (nmon,2=dim) table of exponents of the
 2D monomials of given degree
"""
function mon_exp(deg)
    dimPk = trunc(Int, (deg + 1) * (deg + 2) / 2)
    αList = zeros(dimPk, 2)

    for i = 2:dimPk
        if αList[i-1, 1] == 0
            αList[i, 1] = αList[i-1, 2] + 1
        else
            αList[i, :] = [αList[i-1, 1] - 1 αList[i-1, 2] + 1]
        end
    end
    return αList
end

""" idx_mon(α) takes a multiindex exponent and returns its
integer value idx, e.g. (2,0) -> 4
"""
function idx_mon(α)
    fam = Int(sum(α))
    nb_before = 0
    if fam == 0
        return fam + 1
    else
        nb_before = sum([i - 1 for i = 1:fam+1])
    end
    return floor(Int, nb_before + α[2] + 1)
end

function eval_scaled_polynomial(x, βList, p_c, h, deg)
    αList = mon_exp(deg)
    output = 0.0
    for (α, β) in zip(eachrow(αList), βList)
        output += β * eval_scaled_mon(x, p_c, h, α)
    end
    return output
end

function eval_grad_polynomial(x, coef_list, p_c, h, deg)
    αList = mon_exp(deg)
    αList = αList[2:end, :] # removes the constant monomial
    output = [0.0, 0.0]
    for (α, c) in zip(eachrow(αList), coef_list)
        output += c * eval_grad_mon(x, p_c, h, α)
    end
    return output
end

function scaled_element_stiffness_matrix(mesh, cell, k, μ)
    monExps = Monomials.mon_exp(k)

    h = mesh.cell_diams[cell]
    centroid = mesh.cell_centroids[cell, :]

    gradMon = [eval_grad_mon(centroid, centroid, h, exps)' for exps in eachrow(monExps)] #create grads
    gradMon = vcat(gradMon...) # reshape to 2D array

    return mesh.cell_areas[cell] * gradMon * μ(centroid) * gradMon'
end

end

module ProjectionOps

# using LinearAlgebra # must have the used packages INSIDE module
# using Images

export P0Proj

#= Projection onto constants
- vhDOF is matrix of typeOfDOF*numDOFs, where zeros are
entries which are either =0 or don't exist
- deg is the polynomial degree
    =#
function P0Proj(vhDOF,deg)
    if deg==1
        vhVVDOF = vhDOF[1,:] # vertex-value-dofs
        n_v = length(vhDOFs)
        return 1/n_v*sum(vhVVDOF)
    else # k>=2
        vhMDOF = vhDOF[3,:] # moment-dofs (up to order k-2)
        return vhMDOF[1] # first moment
    end
end

end

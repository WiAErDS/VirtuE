module VirtuE

using SparseArrays
using LinearAlgebra

include("Meshing/__init__.jl")
include("VEMFuns/__init__.jl")
include("Quadrature/__init__.jl")
include("Discretization/__init__.jl")
include("Solvers/__init__.jl")

end
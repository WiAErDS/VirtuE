using LinearAlgebra
using SparseArrays

using Revise
using VirtuE

## -------------- (u,v)+(divu,divv) refinement tests --------------#
k = 0 # Polynomial degree
N = 10 # size of mesh

eps_list = 10.0 .^ (-3:-1:-7)
cond_nbrs = []
cond_nbrs_prec = []

for eps = eps_list
    mesh = Meshing.create_tri_mesh(N)

    levelset(x) = x[2] - (0.5 + eps)
    mesh = Meshing.remesh(mesh, levelset)

    M = AuxPrecond.assemble_mixed_energy_matrix(mesh, k)
    P = AuxPrecond.AuxPreconditioner(mesh)
    M_prec = AuxPrecond.apply_aux_precond_to_mat(P, collect(M))

    eigs = extrema(eigvals(collect(M)))
    append!(cond_nbrs, eigs[2] / eigs[1])
    eigs_prec = extrema(eigvals(collect(M_prec)))
    append!(cond_nbrs_prec, eigs_prec[2] / eigs_prec[1])
end

display(hcat(eps_list, cond_nbrs, cond_nbrs_prec)')

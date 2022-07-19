using LinearAlgebra
using SparseArrays

using Revise
using VirtuE

##-------------- Preconditioned Darcy system --------------#
h_list = 10.0 .^ (-2:-1:-6)
cond_nbrs = [];
cond_nbrs_prec = [];
k = 0 # Polynomial degree

# [Uncomment/comment lines 30-31 / 32-34]
# for h = 2 .^ (-1:-1:-4) # size of mesh

for h = h_list
    levelset(x) = x[2] - (0.5 + h)

    mesh = Meshing.create_tri_mesh(10)
    mesh = Meshing.remesh(mesh, levelset)
    num_faces = Meshing.get_num_faces(mesh)

    M = Mixed.assemble_lhs(mesh, k)
    M[num_faces+1:end, :] *= -1 # symmetrize the system
    P = AuxPrecond.AuxPreconditioner(mesh)
    M_prec = AuxPrecond.apply_Darcy_precond_to_mat(P, M)

    eigs = extrema(abs.(eigvals(collect(M))))
    append!(cond_nbrs, eigs[2] / eigs[1])
    eigs_prec = extrema(abs.(eigvals(collect(M_prec))))
    append!(cond_nbrs_prec, eigs_prec[2] / eigs_prec[1])
end

display(hcat(h_list, cond_nbrs, cond_nbrs_prec)')
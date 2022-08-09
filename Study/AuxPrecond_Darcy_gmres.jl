using LinearAlgebra
using SparseArrays
using IterativeSolvers

using Revise
using VirtuE

##-------------- Preconditioned Darcy system iterative solver check --------------#

# ----- Problem setup 
levelset(x) = x[2] - (0.5 + 1e-6)

μ_inv(x) = [1 0; 0 1]

# source(x) = 0
# p_bdry(x) = 1 - x[1]

source_scalar(x) = 40 * π^2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
p_bdry(x) = 0
u_sol(x) = -[4π * cos(2π * x[1]) * sin(4π * x[2]), 8π * cos(4π * x[2]) * sin(2π * x[1])]
p_sol(x) = 2 * sin(2 * π * x[1]) * sin(4 * π * x[2])

# source_scalar(x) = 0
# p_sol(x) = -sin(x[1])*sinh(x[2]) - (cos(1) - 1)*(cosh(1) - 1)
# u_sol(x) = [cos(x[1])*sinh(x[2]), sin(x[1])*cosh(x[2])]
# p_bdry(x) = p_sol(x)

k = 0 # Polynomial degree

# ----- Output vectors

h_list = []
u_errors = []
p_errors = []
u_errors_prec = []
p_errors_prec = []
cond_nbrs = []
cond_nbrs_prec = []

# ----- Computing loop
for N = 10 * 2 .^ (0:1)
    append!(h_list, 1 / N)

    mesh = Meshing.create_tri_mesh(N)
    mesh = Meshing.remesh(mesh, levelset)
    num_faces = Meshing.get_num_faces(mesh)

    M = Mixed.assemble_lhs(mesh, k, μ_inv)
    b = Mixed.assemble_rhs(mesh, k, source_scalar, p_bdry)
    ξ = @time gmres(M, b)#; verbose=true)

    u = ξ[1:Meshing.get_num_faces(mesh)]
    p = ξ[Meshing.get_num_faces(mesh)+1:end]
    u_error = Mixed.norm_L2(mesh, k, u, u_sol)
    p_error = Mixed.norm_L2(mesh, k, p, p_sol)
    append!(u_errors, u_error)
    append!(p_errors, p_error)

    # M[num_faces+1:end, :] *= -1 # symmetrize the system
    P = AuxPrecond.AuxPreconditioner(mesh)
    PQ_inv = AuxPrecond.get_mat_for_gmres_darcy(P)
    PQ_inv = factorize(PQ_inv)
    # M_prec = AuxPrecond.apply_Darcy_precond_to_mat(P, M)
    # b_prec = AuxPrecond.apply_Darcy_precond_to_mat(P, b)
    ξ_prec = @time gmres(M, b; Pl=PQ_inv)

    u_prec = ξ[1:Meshing.get_num_faces(mesh)]
    p_prec = ξ[Meshing.get_num_faces(mesh)+1:end]
    u_error_prec = Mixed.norm_L2(mesh, k, u, u_sol)
    p_error_prec = Mixed.norm_L2(mesh, k, p, p_sol)
    append!(u_errors_prec, u_error_prec)
    append!(p_errors_prec, p_error_prec)

    # eigs = extrema(abs.(eigvals(collect(M))))
    # append!(cond_nbrs, eigs[2] / eigs[1])
    # eigs_prec = extrema(abs.(eigvals(collect(M_prec))))
    # append!(cond_nbrs_prec, eigs_prec[2] / eigs_prec[1])
end

display(hcat(h_list, cond_nbrs, cond_nbrs_prec)')
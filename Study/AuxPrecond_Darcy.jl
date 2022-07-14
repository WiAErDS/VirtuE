using LinearAlgebra
using SparseArrays

using Revise
using VirtuE

##-------------- Preconditioned Darcy system, refinement test --------------#

μ_inv(x) = [1 0; 0 1]

source_scalar(x) = 40 * π^2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
p_bdry(x) = 0
u_sol(x) = -[4π * cos(2π * x[1]) * sin(4π * x[2]), 8π * cos(4π * x[2]) * sin(2π * x[1])]
p_sol(x) = 2 * sin(2 * π * x[1]) * sin(4 * π * x[2])

# source_scalar(x) = 0
# p_sol(x) = -sin(x[1])*sinh(x[2]) - (cos(1) - 1)*(cosh(1) - 1)
# u_sol(x) = [cos(x[1])*sinh(x[2]), sin(x[1])*cosh(x[2])]
# p_bdry(x) = p_sol(x)

h = [];
u_errors = [];
p_errors = [];
cond_nbrs = [];
cond_nbrs_prec = [];
k = 0 # Polynomial degree
levelset(x) = x[2] - (0.501 + 10.0^(-1))

# [Uncomment/comment lines 30-31 / 32-34]
# for N = 2 .^ (1:4) # size of mesh
#     append!(h, 1 / N)
for N = 2:6
    append!(h, 10.0^(-N))
    levelset(x) = x[2] - (0.5 + 10.0^(-N))


    mesh = Meshing.create_tri_mesh(N)
    mesh = Meshing.remesh(mesh, levelset)
    num_faces = Meshing.get_num_faces(mesh)

    A, b, ξ, M, B = Mixed.Darcy_setup(mesh, k, source_scalar, p_bdry, μ_inv, true)
    A_prec, b_prec = AuxPrecond.apply_Darcy_precond(M, B, b, mesh, k, μ_inv)

    # areas = Meshing.get_cell_areas(mesh)
    # area_ratio = minimum(areas) / maximum(areas)

    # append!(cond_nbrs, cond(Array(A)))
    # append!(cond_nbrs_prec, cond(Array(A_prec)))
    eigs = eigvals(collect(M))
    append!(cond_nbrs, maximum(eigs) / minimum(eigs))
    eigs_prec = eigvals(collect(M_prec))
    append!(cond_nbrs_prec, maximum(eigs_prec) / minimum(eigs_prec))

    # u = ξ[1:num_faces]
    # p = ξ[num_faces+1:end]
    # u_error = Mixed.norm_L2(mesh, k, u, u_sol)
    # p_error = Mixed.norm_L2(mesh, k, p, p_sol)
    # append!(u_errors, u_error)
    # append!(p_errors, p_error)

    ξ_prec = A_prec \ b_prec
    u = ξ_prec[1:num_faces]
    p = ξ_prec[num_faces+1:end]
    u_error = Mixed.norm_L2(mesh, k, u, u_sol)
    p_error = Mixed.norm_L2(mesh, k, p, p_sol)
    append!(u_errors, u_error)
    append!(p_errors, p_error)

    println("Iteration N=", N, " done.")
end
u_convgs = log.(u_errors[2:end] ./ u_errors[1:end-1]) ./ log.(h[2:end] ./ h[1:end-1])
p_convgs = log.(p_errors[2:end] ./ p_errors[1:end-1]) ./ log.(h[2:end] ./ h[1:end-1])

display(u_convgs)
display(p_convgs)

display(hcat([h, cond_nbrs, cond_nbrs_prec]))
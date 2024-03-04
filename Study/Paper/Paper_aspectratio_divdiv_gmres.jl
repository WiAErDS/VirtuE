using LinearAlgebra
using SparseArrays
using IterativeSolvers

using Revise
using VirtuE

## -------------- (u,v)+(divu,divv) refinement tests --------------#
k = 0 # Polynomial degree
N = 5 # size of mesh
# offset = 1/4N
mesh = Meshing.create_rect_mesh(N);

for i = 1:N/2
    r = i/(N+0.5)#+offset*(i==N/2)
    # println(r)
    levelset_0(x) = (x[1]-0.5)^10 + (x[2]-0.5)^10 - r^10
    mesh = Meshing.remesh(mesh, levelset_0);
end
plt = Meshing.draw_mesh(mesh)

eps = 5 * 10.0^(-7)
levelset(x) = x[2] - (0.5 + eps)
mesh = Meshing.remesh(mesh, levelset)
# Meshing.draw_line_on_mesh(plt, mesh, levelset)

# Problem matrices and preconditioning
M = AuxPrecond.assemble_mixed_energy_matrix(mesh, k);
P = AuxPrecond.AuxPreconditioner("energy",mesh);

b = randn(size(M, 1))
restart = size(b, 1)

x_unpr, log_unpr = gmres(M, b, restart=restart, log=true);
x_prec, log_prec = gmres(M, b, Pl=P, restart=restart, log=true);

println("Unpreconditioned: ", log_unpr)
println("Preconditioned: ", log_prec)

# Aspect ratios as indicator for cond nbrs 
areas = mesh.cell_areas
diams = mesh.cell_diams
aspect_ratios = diams .^ 2 ./ areas
println("Maximum aspect ratio after expansion: ", maximum(aspect_ratios))

# Condition numbers
M_prec = AuxPrecond.apply_precond_to_mat(P, collect(M))
eigs = extrema(abs.(eigvals(collect(M))))
println(eigs[2] / eigs[1])
# println(cond(collect(M))) # = above, since symmetric problem
eigs_prec = extrema(abs.(eigvals(collect(M_prec))))
println(eigs_prec[2] / eigs_prec[1])
# println(cond(collect(M_prec))) = above, since symmetric problem
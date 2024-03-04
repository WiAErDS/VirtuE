using LinearAlgebra
using SparseArrays
using IterativeSolvers

using Revise
using VirtuE

## -------------- (u,v)+(divu,divv) refinement tests --------------#
k = 0 # Polynomial degree
for N in [5, 10, 20] # loop over mesh sizes
    # N = 5 # size of mesh
    # offset = 1/4N
    mesh = Meshing.create_rect_mesh(N);

    h = 1/N #maximum(mesh.cell_diams)
    p = 1
    r = 0.75*h # 0.35, 2/3, 0.75
    nodes = mesh.node_coords
    for (x, y) in eachrow(nodes)
        levelset_0(z) = (abs(z[1]-x))^p + (abs(z[2]-y))^p - r^p
        mesh = Meshing.remesh(mesh, levelset_0);
    end
    plt = Meshing.draw_mesh(mesh)

    eps = 5 * 10.0^(-7)
    levelset(x) = x[2] - (0.5 + eps)
    mesh = Meshing.remesh(mesh, levelset)
    # Meshing.draw_line_on_mesh(plt, mesh, levelset)

    # Problem matrices and preconditioning
    M = AuxPrecond.assemble_mixed_energy_matrix(mesh, k);
    P = AuxPrecond.AuxPreconditioner(mesh);

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
end
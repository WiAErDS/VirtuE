using LinearAlgebra
using SparseArrays
using IterativeSolvers

using Revise
using VirtuE

## -------------- (u,v)+(divu,divv) refinement testing grounds --------------#
    k = 0 # Polynomial degree
    N = 10 # size of mesh
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


## -------------- (u,v)+(divu,divv) refinement tests --------------#
meshes = []
for N in [5,10,20]
    mesh = Meshing.create_rect_mesh(N);

    h = 1/N #maximum(mesh.cell_diams)
    p = 1
    r = 0.75*h # 0.35, 2/3, 0.75
    nodes = mesh.node_coords
    for (x, y) in eachrow(nodes)
        levelset_0(z) = (abs(z[1]-x))^p + (abs(z[2]-y))^p - r^p
        mesh = Meshing.remesh(mesh, levelset_0);
    end
    meshes = [meshes..., mesh]
end

k = 0 # Polynomial degree
diam_list = ["Diameters"]
cond_list = ["κ(M)"]
diagcond_list = ["κ(diagM/M)"]
auxcond_list = ["κ(PM)"]
gmres_list = ["GMRES M"]
diaggmres_list = ["GMRES diagM/M"]
auxgmres_list = ["GMRES PM"]
# for N in [5,10,20]
#     mesh = Meshing.create_rect_mesh(N);

#     h = 1/N #maximum(mesh.cell_diams)
#     p = 1
#     r = 0.75*h # 0.35, 2/3, 0.75
#     nodes = mesh.node_coords
#     for (x, y) in eachrow(nodes)
#         levelset_0(z) = (abs(z[1]-x))^p + (abs(z[2]-y))^p - r^p
#         mesh = Meshing.remesh(mesh, levelset_0);
#     end
#    # plt = Meshing.draw_mesh(mesh)

for i = 1:eachindex(meshes)
    mesh = meshes[i]
    # Problem matrices and preconditioning
    M = AuxPrecond.assemble_mixed_energy_matrix(mesh, k);
    P = AuxPrecond.AuxPreconditioner("energy",mesh);
    M_diag = spdiagm(diag(M))

    b = randn(size(M, 1))
    restart = size(b, 1)

    x_unpr, log_unpr = gmres(M, b, restart=restart, log=true);
    x_diag, log_diag = gmres(M_diag\M, M_diag\b, restart=restart, log=true);
    x_prec, log_prec = gmres(M, b, Pl=P, restart=restart, log=true);

    gmres_list = [gmres_list..., log_unpr]
    diaggmres_list = [diaggmres_list..., log_diag]
    auxgmres_list = [auxgmres_list..., log_prec]

    # Aspect ratios as indicator for cond nbrs 
    areas = mesh.cell_areas
    diams = mesh.cell_diams
    aspect_ratios = diams .^ 2 ./ areas
    diam_list = [diam_list..., maximum(aspect_ratios)]

    # Condition numbers
    M_prec = AuxPrecond.apply_precond_to_mat(P, collect(M))
    eigs = extrema(abs.(eigvals(collect(M))))
    cond_list = [cond_list..., eigs[2] / eigs[1]]

    eigs_diag = extrema(abs.(eigvals(collect(M_diag\M))))
    diagcond_list = [diagcond_list..., eigs_diag[2] / eigs_diag[1]]

    eigs_prec = extrema(abs.(eigvals(collect(M_prec))))
    auxcond_list = [auxcond_list..., eigs_prec[2] / eigs_prec[1]]
end
# iter_list = [0, 1, 2, 3, 4, 5, 6]
# eps_list = [5 * 10.0^(-i) for i in 4:9]

table = [diam_list cond_list diagcond_list auxcond_list gmres_list diaggmres_list auxgmres_list]
table[:,1:4]
table[:,5:end]
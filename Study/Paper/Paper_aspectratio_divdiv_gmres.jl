using LinearAlgebra
using SparseArrays
using IterativeSolvers

using Revise
using VirtuE

## -------------- (u,v)+(divu,divv) testing ground --------------#
    k = 0 # Polynomial degree
    N = 10 # size of mesh
    # offset = 1/4N
    mesh = Meshing.create_rect_mesh(N);

    for i = 1:N/2
        r = i/(N+0.5)#+offset*(i==N/2)
        levelset_0(x) = (x[1]-0.5)^10 + (x[2]-0.5)^10 - r^10
        # r = i/(N+5.1)
        # levelset_0(x) = abs(x[1]-0.5) + abs(x[2]-0.5) - r
        mesh = Meshing.remesh(mesh, levelset_0);
    end
    plt = Meshing.draw_mesh(mesh)

    eps = 5 * 10.0^(-9)
    levelset(x) = x[2] - (0.5 + eps)
    mesh = Meshing.remesh(mesh, levelset)
    # # Meshing.draw_line_on_mesh(plt, mesh, levelset)

    # Problem matrices and preconditioning
    M = AuxPrecond.assemble_mixed_energy_matrix(mesh, k);
    P = AuxPrecond.AuxPreconditioner("energy",mesh);
    M_diag = spdiagm(diag(M))

    b = randn(size(M, 1))
    restart = size(b, 1)

    x_unpr, log_unpr = gmres(M, b, restart=restart, log=true);
    x_diag, log_diag = gmres(M_diag\M, M_diag\b, restart=restart, log=true);
    x_prec, log_prec = gmres(M, b, Pl=P, restart=restart, log=true);

    println("Unpreconditioned: ", log_unpr)
    println("Diagonal: ", log_diag)
    println("Preconditioned: ", log_prec)

    # Aspect ratios as indicator for cond nbrs 
    areas = mesh.cell_areas
    diams = mesh.cell_diams
    aspect_ratios = diams .^ 2 ./ areas
    println("Maximum aspect ratio after expansion: ", maximum(aspect_ratios))

    # Condition numbers
    M_prec = AuxPrecond.apply_precond_to_mat(P, collect(M))
    eigs = extrema(abs.(eigvals(collect(M))))
    println(eigs[2] / eigs[1])# println(cond(collect(M))) # = above, since symmetric problem
    eigs_diag = extrema(abs.(eigvals(collect(M_diag\M))))
    println(eigs_diag[2] / eigs_diag[1])
    eigs_prec = extrema(abs.(eigvals(collect(M_prec))))
    println(eigs_prec[2] / eigs_prec[1])# println(cond(collect(M_prec))) = above, since symmetric problem


## -------------- (u,v)+(divu,divv) refinement tests --------------#
    k = 0 # Polynomial degree
    N = 10 # size of mesh
    # offset = 1/4N
    mesh = Meshing.create_rect_mesh(N);

    for i = 1:N/2
        r = i/(N+0.5)#+offset*(i==N/2)
        # println(r)
        levelset_0(x) = (x[1]-0.5)^10 + (x[2]-0.5)^10 - r^10
        mesh = Meshing.remesh(mesh, levelset_0);
    end
    plt = Meshing.draw_mesh(mesh)
    meshes = [mesh]
    for i in 4:9
        eps = 5 * 10.0^(-i)
        levelset(x) = x[2] - (0.5 + eps)
        meshes = [meshes..., Meshing.remesh(mesh, levelset)]
        # Meshing.draw_line_on_mesh(plt, mesh, levelset)
    end

    diam_list = ["Diameters"]
    cond_list = ["κ(M)"]
    diagcond_list = ["κ(diagM/M)"]
    auxcond_list = ["κ(PM)"]
    gmres_list = ["GMRES M"]
    diaggmres_list = ["GMRES diagM/M"]
    auxgmres_list = ["GMRES PM"]
    for i in eachindex(meshes)
        # Problem matrices and preconditioning
        M = AuxPrecond.assemble_mixed_energy_matrix(meshes[i], k);
        M_diag = spdiagm(diag(M))
        P = AuxPrecond.AuxPreconditioner("energy",meshes[i]);
        # P = AuxPrecond.AuxPreconditioner("face",mesh);

        b = randn(size(M, 1))
        restart = size(b, 1)

        x_unpr, log_unpr = gmres(M, b, restart=restart, log=true);
        x_diag, log_diag = gmres(M_diag\M, M_diag\b, restart=restart, log=true);
        x_prec, log_prec = gmres(M, b, Pl=P, restart=restart, log=true);

        gmres_list = [gmres_list..., log_unpr]
        diaggmres_list = [diaggmres_list..., log_diag]
        auxgmres_list = [auxgmres_list..., log_prec]

        # Aspect ratios as indicator for cond nbrs 
        areas = meshes[i].cell_areas
        diams = meshes[i].cell_diams
        aspect_ratios = diams .^ 2 ./ areas
        diam_list = [diam_list..., maximum(aspect_ratios)]

        # Condition numbers
        eigs = extrema(abs.(eigvals(collect(M))))
        cond_list = [cond_list..., eigs[2] / eigs[1]]

        eigs_diag = extrema(abs.(eigvals(collect(M_diag\M))))
        diagcond_list = [diagcond_list..., eigs_diag[2] / eigs_diag[1]]

        M_prec = AuxPrecond.apply_precond_to_mat(P, collect(M))
        eigs_prec = extrema(abs.(eigvals(collect(M_prec))))
        auxcond_list = [auxcond_list..., eigs_prec[2] / eigs_prec[1]]
        # auxcond_list = [auxcond_list..., cond(collect(M_prec))]
        
        M,P,M_diag,M_prec = Nothing,Nothing,Nothing,Nothing
    end
    # iter_list = [0, 1, 2, 3, 4, 5, 6]
    # eps_list = [5 * 10.0^(-i) for i in 4:9]

    table = [diam_list cond_list diagcond_list auxcond_list gmres_list diaggmres_list auxgmres_list]
    display(table[:,1:4])
    display(table[:,5:end])
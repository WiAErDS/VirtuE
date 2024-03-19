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

# RHS data (from Study/Mixed_convg.jl)
source_vector(x) = -[2π * cos(2π * x[1]) * sin(4π * x[2]), 4π * cos(4π * x[2]) * sin(2π * x[1])]

" CHOOSE ADDITIVE OR MULTIPLICATIVE"
# preconditioner_choice = "additive"
preconditioner_choice = "multiplicative"

" CHOOSE PRECONDITIONER "
# smoother_choice = "energy"
smoother_choice = "face"

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

for i = 1:length(meshes)
    mesh = meshes[i]
    # Problem matrices and preconditioning
    M, MassMat = AuxPrecond.assemble_mixed_energy_matrix(mesh, k);
    M_diag = spdiagm(diag(M))
    P = AuxPrecond.AuxPreconditioner(smoother_choice,mesh);
    M_prec = AuxPrecond.apply_precond_to_mat(P, collect(M))
    if preconditioner_choice == "multiplicative"
        P = AuxPrecondMultiplicative.AuxPreconditionerMultiplicative(smoother_choice,mesh);
        M_prec = AuxPrecondMultiplicative.apply_precond_to_mat(P, collect(M))
    end

    # b = randn(size(M, 1))
    b = Mixed.assemble_divdiv_rhs(mesh, k, source_vector, MassMat)
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

using Printf
# Open a file in write mode
open("Study/Paper/Refine_divdiv/Paper_refine_divdiv_"*preconditioner_choice*"_"*smoother_choice*".txt", "w") do file
    # Write the matrix to the file with formatting
    for i in 1:size(table, 1)
        if i == 1 
            write(file, lpad(string(table[i, 1]), 5))
            write(file, lpad(string(table[i, 2]), 17))
            write(file, lpad(string(table[i, 3]), 18))
            write(file, lpad(string(table[i, 4]), 12))
            write(file, lpad(string(table[i, 5]), 34))
            write(file, lpad(string(table[i, 6]), 34))
            write(file, lpad(string(table[i, 7]), 34))
            write(file, "\n") # New line at the end of each row
        else
            for j in 1:size(table, 2)
                # Check if the element is a number
                if isa(table[i, j], Number)
                    # Format numbers in scientific notation
                    formatted_number = @sprintf("%.2E", table[i, j])
                    write(file, lpad(formatted_number, 15))
                else
                    # Just write the string as it is, left-aligned
                    write(file, lpad(string(table[i, j]), 40))
                end
            end
            write(file, "\n") # New line at the end of each row
        end
    end
end
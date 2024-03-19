using LinearAlgebra
using SparseArrays
using IterativeSolvers

using Revise
using VirtuE

## -------------- Darcy precond testing grounds --------------#
    k = 0 # Polynomial degree

    μ_inv(x) = [1 0; 0 1]
    # source_scalar(x) = 40 * π^2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
    # p_bdry(x) = 0
    # u_sol(x) = -[4π * cos(2π * x[1]) * sin(4π * x[2]), 8π * cos(4π * x[2]) * sin(2π * x[1])]
    # p_sol(x) = 2 * sin(2 * π * x[1]) * sin(4 * π * x[2])

    N = 10 # size of mesh

    eps = 5 * 10.0^-4

    mesh = Meshing.create_tri_mesh(N)

    levelset(x) = x[2] - (0.5 + eps)
    mesh = Meshing.remesh(mesh, levelset)

    M = Mixed.assemble_lhs(mesh, k, μ_inv)
    P = AuxPrecond.AuxPreconditioner_Darcy(mesh)

    b = randn(size(M, 1))
    restart = size(b, 1)

    x_unpr, log_unpr = gmres(M, b, restart=restart, log=true)
    x_prec, log_prec = gmres(M, b, Pl=P, restart=restart, log=true)

    println("Unpreconditioned: ", log_unpr)
    println("Preconditioned: ", log_prec)

## -------------- Darcy refinement tests --------------#
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
source_scalar(x) = 40 * π^2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
source_vector(x) = -[2π * cos(2π * x[1]) * sin(4π * x[2]), 4π * cos(4π * x[2]) * sin(2π * x[1])]
p_bdry(x) = 0
u_sol(x) = -[4π * cos(2π * x[1]) * sin(4π * x[2]), 8π * cos(4π * x[2]) * sin(2π * x[1])]
p_sol(x) = sin(2 * π * x[1]) * sin(4 * π * x[2])

" THERE IS ONLY ADDITIVE "
preconditioner_choice = "additive"

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

for i = 1:length(meshes)
    mesh = meshes[i]
    # Problem matrices and preconditioning
    A,M = Mixed.assemble_lhs(mesh, k)
    P_diag = spdiagm(vcat(diag(M), ones(Meshing.get_num_cells(mesh))))
    P = AuxPrecond.AuxPreconditioner_Darcy(smoother_choice,mesh);
    A_prec = AuxPrecond.apply_precond_to_mat(P, collect(A))

    # b = randn(size(M, 1))
    b = Mixed.assemble_rhs(mesh, k, source_scalar, source_vector, p_bdry, M)
    restart = size(b, 1)

    x_unpr, log_unpr = gmres(A, b, restart=restart, log=true);
    x_diag, log_diag = gmres(P_diag\A, P_diag\b, restart=restart, log=true);
    x_prec, log_prec = gmres(A, b, Pl=P, restart=restart, log=true);

    gmres_list = [gmres_list..., log_unpr]
    diaggmres_list = [diaggmres_list..., log_diag]
    auxgmres_list = [auxgmres_list..., log_prec]

    # Aspect ratios as indicator for cond nbrs 
    areas = mesh.cell_areas
    diams = mesh.cell_diams
    aspect_ratios = diams .^ 2 ./ areas
    diam_list = [diam_list..., maximum(aspect_ratios)]

    # Condition numbers
    eigs = extrema(abs.(eigvals(collect(A))))
    cond_list = [cond_list..., eigs[2] / eigs[1]]

    eigs_diag = extrema(abs.(eigvals(collect(P_diag\A))))
    diagcond_list = [diagcond_list..., eigs_diag[2] / eigs_diag[1]]

    eigs_prec = extrema(abs.(eigvals(collect(A_prec))))
    auxcond_list = [auxcond_list..., eigs_prec[2] / eigs_prec[1]]
end
# iter_list = [0, 1, 2, 3, 4, 5, 6]
# eps_list = [5 * 10.0^(-i) for i in 4:9]

table = [diam_list cond_list diagcond_list auxcond_list gmres_list diaggmres_list auxgmres_list]
# table[:,1:4]
# table[:,5:end]

using Printf
# Open a file in write mode
open("Study/Paper/Refine_darcy/Paper_refine_darcy_"*preconditioner_choice*"_"*smoother_choice*".txt", "w") do file
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
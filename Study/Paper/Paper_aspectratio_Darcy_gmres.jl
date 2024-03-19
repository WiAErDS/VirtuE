using LinearAlgebra
using SparseArrays
using IterativeSolvers

using Revise
using VirtuE

## -------------- Darcy testing ground --------------#
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


## -------------- Darcy aspectratio tests --------------#
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
# plt = Meshing.draw_mesh(mesh)
meshes = [mesh]
for i in 4:9
    eps = 5 * 10.0^(-i)
    levelset(x) = x[2] - (0.5 + eps)
    meshes = [meshes..., Meshing.remesh(mesh, levelset)]
    # Meshing.draw_line_on_mesh(plt, mesh, levelset)
end

# RHS data (from Study/Mixed_convg.jl)
source_scalar(x) = 0
source_vector(x) = [cos(x[1])*sinh(x[2]), sin(x[1])*cosh(x[2])]
p_sol(x) = -sin(x[1])*sinh(x[2]) - (cos(1) - 1)*(cosh(1) - 1)
u_sol(x) = 2*[cos(x[1])*sinh(x[2]), sin(x[1])*cosh(x[2])]
p_bdry(x) = p_sol(x)

" CHOOSE PRECONDITIONER "
# smoother_choice = "energy"
smoother_choice = "face"

" THERE IS ONLY ADDITIVE"
preconditioner_choice = "additive"

diam_list = ["Diameters"]
cond_list = ["κ(M)"]
diagcond_list = ["κ(diagM/M)"]
auxcond_list = ["κ(PM)"]
gmres_list = ["GMRES M"]
diaggmres_list = ["GMRES diagM/M"]
auxgmres_list = ["GMRES PM"]
for i in eachindex(meshes)
    # Problem matrices and preconditioning
    A,M = Mixed.assemble_lhs(meshes[i], k)
    P_diag = spdiagm(vcat(diag(M), ones(Meshing.get_num_cells(meshes[i]))))
    P = AuxPrecond.AuxPreconditioner_Darcy(smoother_choice,meshes[i]);
    A_prec = AuxPrecond.apply_precond_to_mat(P, collect(A))
    
    # b = randn(size(M, 1))
    b = Mixed.assemble_rhs(meshes[i], k, source_scalar, source_vector, p_bdry, M)
    restart = size(b, 1)

    x_unpr, log_unpr = gmres(A, b, restart=restart, log=true);
    diagP_prod = prod(vcat(diag(M), ones(Meshing.get_num_cells(meshes[i]))))
    if diagP_prod == 0 
        log_diag = "Diagonal of P_diag contains a zero"
    else
        x_diag, log_diag = gmres(P_diag\A, P_diag\b, restart=restart, log=true);
    end
    x_prec, log_prec = gmres(A, b, Pl=P, restart=restart, log=true);

    gmres_list = [gmres_list..., log_unpr]
    diaggmres_list = [diaggmres_list..., log_diag]
    auxgmres_list = [auxgmres_list..., log_prec]

    # Aspect ratios as indicator for cond nbrs 
    areas = meshes[i].cell_areas
    diams = meshes[i].cell_diams
    aspect_ratios = diams .^ 2 ./ areas
    diam_list = [diam_list..., maximum(aspect_ratios)]

    # Condition numbers
    eigs = extrema(abs.(eigvals(collect(A))))
    cond_list = [cond_list..., eigs[2] / eigs[1]]

    if diagP_prod == 0 
        eigs_diag = [Inf,Inf]
    else
        eigs_diag = extrema(abs.(eigvals(collect(P_diag\A))))
    end
    diagcond_list = [diagcond_list..., eigs_diag[2] / eigs_diag[1]]

    eigs_prec = extrema(abs.(eigvals(collect(A_prec))))
    auxcond_list = [auxcond_list..., eigs_prec[2] / eigs_prec[1]]
    # auxcond_list = [auxcond_list..., cond(collect(M_prec))]
    
    # M,P,M_diag,M_prec = Nothing,Nothing,Nothing,Nothing
end
# iter_list = [0, 1, 2, 3, 4, 5, 6]
# eps_list = [5 * 10.0^(-i) for i in 4:9]

table = [diam_list cond_list diagcond_list auxcond_list gmres_list diaggmres_list auxgmres_list];
# display(table[:,1:4])
# display(table[:,5:end])

using Printf
# Open a file in write mode
open("Study/Paper/Aspectratio_darcy/Paper_aspectratio_darcy_"*preconditioner_choice*"_"*smoother_choice*".txt", "w") do file
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


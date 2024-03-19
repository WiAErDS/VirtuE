using LinearAlgebra
using SparseArrays
using IterativeSolvers

using Revise
using VirtuE

using Printf

## -------------- printing to file function --------------#
function export_result_to_file(mesh_choice, problem_choice, preconditioner_choice, smoother_choice, table)
    # Open a file in write mode
    open("Study/Paper/"*mesh_choice*"_"*problem_choice*"/"*preconditioner_choice*"_"*smoother_choice*".txt", "w") do file
        # Write the matrix to the file with formatting
        for i in 1:size(table, 1)
            if i == 1 
                write(file, rpad(string(table[i, 1]), 20))
                write(file, rpad(string(table[i, 2]), 20))
                write(file, rpad(string(table[i, 3]), 20))
                write(file, rpad(string(table[i, 4]), 20))
                write(file, rpad(string(table[i, 5]), 20))
                write(file, rpad(string(table[i, 6]), 20))
                write(file, rpad(string(table[i, 7]), 20))
                write(file, "\n") # New line at the end of each row
            else
                for j in 1:4
                    # Check if the element is a number
                    if isa(table[i, j], Number)
                        # Format numbers in scientific notation
                        formatted_number = @sprintf("%.2E", table[i, j])
                        write(file, rpad(formatted_number, 20))
                    else
                        # Just write the string as it is, left-aligned
                        write(file, rpad(string(table[i, j]), 20))
                    end
                end
                for j in 5:size(table, 2)
                    formatted_number = @sprintf("%.f", table[i, j])
                    write(file, rpad(formatted_number, 20))
                end
                write(file, "\n") # New line at the end of each row
            end
        end
    end
end

## -------------- test function --------------#
function tests_for_paper(mesh_choice, problem_choice, smoother_choice, preconditioner_choice, do_all_cond=true)
    if problem_choice == "darcy" # only additive fo Darcy
        preconditioner_choice = "additive"
    end

    k = 0 # Polynomial degree

    meshes = []
    if mesh_choice == "aspectratio"
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
    else 
        for N in [4,8,16,32]
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
    end

    # RHS data (from Study/Mixed_convg.jl)
    source_scalar(x) = 0
    source_vector(x) = [cos(x[1])*sinh(x[2]), sin(x[1])*cosh(x[2])]
    p_sol(x) = -sin(x[1])*sinh(x[2]) - (cos(1) - 1)*(cosh(1) - 1)
    u_sol(x) = 2*[cos(x[1])*sinh(x[2]), sin(x[1])*cosh(x[2])]
    p_bdry(x) = p_sol(x)
    if mesh_choice == "refinement"
        source_scalar(x) = 40 * π^2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
        source_vector(x) = -[2π * cos(2π * x[1]) * sin(4π * x[2]), 4π * cos(4π * x[2]) * sin(2π * x[1])]
        p_bdry(x) = 0
        u_sol(x) = -[4π * cos(2π * x[1]) * sin(4π * x[2]), 8π * cos(4π * x[2]) * sin(2π * x[1])]
        p_sol(x) = sin(2 * π * x[1]) * sin(4 * π * x[2])
    end

    diam_list = ["Diameters"]
    cond_list = ["κ(M)"]
    diagcond_list = ["κ(diagM/M)"]
    auxcond_list = ["κ(PM)"]
    gmres_list = ["GMRES M"]
    diaggmres_list = ["GMRES diagM/M"]
    auxgmres_list = ["GMRES PM"]
    for i in eachindex(meshes)
        println("Mesh ", i)
        if problem_choice == "divdiv"
            # Problem matrices and preconditioning
            M, MassMat = AuxPrecond.assemble_mixed_energy_matrix(meshes[i], k);
            M_diag = spdiagm(diag(M))
            
            P = [] 
            M_prec = []
            if preconditioner_choice == "additive"
                P = AuxPrecond.AuxPreconditioner(smoother_choice, meshes[i]);
                M_prec = AuxPrecond.apply_precond_to_mat(P, collect(M))
            else
                P = AuxPrecondMultiplicative.AuxPreconditionerMultiplicative(smoother_choice, meshes[i]);
                M_prec = AuxPrecondMultiplicative.apply_precond_to_mat(P, collect(M))
            end
            b = Mixed.assemble_divdiv_rhs(meshes[i], k, source_vector, MassMat)
            restart = size(b, 1)

            _, log_unpr = gmres(M, b, restart=restart, log=true);
            _, log_diag = gmres(M_diag\M, M_diag\b, restart=restart, log=true);
            _, log_prec = gmres(M, b, Pl=P, restart=restart, log=true);

            gmres_list = [gmres_list..., log_unpr.iters]
            diaggmres_list = [diaggmres_list..., log_diag.iters]
            auxgmres_list = [auxgmres_list..., log_prec.iters]

            # Aspect ratios as indicator for cond nbrs 
            areas = meshes[i].cell_areas
            diams = meshes[i].cell_diams
            if mesh_choice == "aspectratio"
                aspect_ratios = diams .^ 2 ./ areas
                diam_list = [diam_list..., maximum(aspect_ratios)]
            else
                diam_list = [diam_list..., maximum(diams)]
            end

            # Condition numbers
            if do_all_cond
                eigs = extrema(abs.(eigvals(collect(M))))
                cond_list = [cond_list..., eigs[2] / eigs[1]]

                eigs_diag = extrema(abs.(eigvals(collect(M_diag\M))))
                diagcond_list = [diagcond_list..., eigs_diag[2] / eigs_diag[1]]
            else
                cond_list = [cond_list..., "See other"]
                diagcond_list = [diagcond_list..., "See other"]
            end

            eigs_prec = extrema(abs.(eigvals(collect(M_prec))))
            auxcond_list = [auxcond_list..., eigs_prec[2] / eigs_prec[1]]
        else
            # Problem matrices and preconditioning
            A,M = Mixed.assemble_lhs(meshes[i], k)
            P_diag = spdiagm(vcat(diag(M), ones(Meshing.get_num_cells(meshes[i]))))
            P = AuxPrecond.AuxPreconditioner_Darcy(smoother_choice,meshes[i]);
            A_prec = AuxPrecond.apply_precond_to_mat(P, collect(A))
            
            b = Mixed.assemble_rhs(meshes[i], k, source_scalar, source_vector, p_bdry, M)
            restart = size(b, 1)

            _, log_unpr = gmres(A, b, restart=restart, log=true);
            _, log_diag = gmres(P_diag\A, P_diag\b, restart=restart, log=true);
            _, log_prec = gmres(A, b, Pl=P, restart=restart, log=true);

            gmres_list = [gmres_list..., log_unpr.iters]
            diaggmres_list = [diaggmres_list..., log_diag.iters]
            auxgmres_list = [auxgmres_list..., log_prec.iters]

            # Aspect ratios as indicator for cond nbrs 
            areas = meshes[i].cell_areas
            diams = meshes[i].cell_diams
            if mesh_choice == "aspectratio"
                aspect_ratios = diams .^ 2 ./ areas
                diam_list = [diam_list..., maximum(aspect_ratios)]
            else
                diam_list = [diam_list..., maximum(diams)]
            end

            # Condition numbers
            if do_all_cond
                eigs = extrema(abs.(eigvals(collect(A))))
                cond_list = [cond_list..., eigs[2] / eigs[1]]

                eigs_diag = extrema(abs.(eigvals(collect(P_diag\A))))
                diagcond_list = [diagcond_list..., eigs_diag[2] / eigs_diag[1]]
            else
                cond_list = [cond_list..., "See other file"]
                diagcond_list = [diagcond_list..., "See other file"]
            end
            eigs_prec = extrema(abs.(eigvals(collect(A_prec))))
            auxcond_list = [auxcond_list..., eigs_prec[2] / eigs_prec[1]]
            # auxcond_list = [auxcond_list..., cond(collect(M_prec))]
        end
    end

    table = [diam_list cond_list diagcond_list auxcond_list gmres_list diaggmres_list auxgmres_list];
    export_result_to_file(mesh_choice, problem_choice, preconditioner_choice, smoother_choice, table)
end

## -------------- tests --------------#
" CHOOSE MESH / TEST "
# mesh_choice = "aspectratio","refinement"
" CHOOSE PROBLEM "
# problem_choice = "divdiv","darcy"
" CHOOSE SMOOTHER "
# smoother_choice = "energy","face"
" CHOOSE PRECONDITIONER"
# preconditioner_choice = "additive","multiplicative"
" SKIP REDO CONDITION NUMBER "
# do_all_cond = true,false

tests_for_paper("aspectratio", "divdiv", "energy", "additive", true)
tests_for_paper("aspectratio", "divdiv", "face", "additive", false)
tests_for_paper("aspectratio", "divdiv", "energy", "multiplicative", false)
tests_for_paper("aspectratio", "divdiv", "face", "multiplicative", false)

tests_for_paper("refinement", "divdiv", "energy", "additive", true)
tests_for_paper("refinement", "divdiv", "face", "additive", false)
tests_for_paper("refinement", "divdiv", "energy", "multiplicative", false)
tests_for_paper("refinement", "divdiv", "face", "multiplicative", false)

tests_for_paper("aspectratio", "darcy", "energy", "additive", true)
tests_for_paper("aspectratio", "darcy", "face", "additive", false)

tests_for_paper("refinement", "darcy", "energy", "additive", true)
tests_for_paper("refinement", "darcy", "face", "additive", false)
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

## -------------- compute meshes --------------#
function construct_meshes(mesh_choice)
    meshes = []
    if mesh_choice == "aspectratio"
        N = 8 # size of mesh
        # offset = 1/4N
        mesh = Meshing.create_rect_mesh(N);
        for i = 1:N
            r = 1/(2N)+(i-1)*1/N
            levelset_0(x) = abs(x[1]-0.5) + abs(x[2]-0.5) - r
            mesh = Meshing.remesh(mesh, levelset_0);
        end
        # plt = Meshing.draw_mesh(mesh)
        meshes = [mesh]
        for i in [2,4,6,8]
            eps = 10.0^(-i)
            levelset(x) = x[2] - (0.5 + eps)
            meshes = [meshes..., Meshing.remesh(mesh, levelset)]
            # Meshing.draw_line_on_mesh(plt, mesh, levelset)
        end
    else 
        # for N in [4,8,16]
        for N in [4,8,16,32]
        # for N in [4,8,16,32,64]
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
            println("Mesh", N, " done")
        end
    end
    return meshes
end

## -------------- test function --------------#
function tests_for_paper(meshes, mesh_choice, problem_choice, smoother_choice, preconditioner_choice, do_all_cond=true)
    k = 0 # Polynomial degree

    # RHS data (from Study/Mixed_convg.jl)
    source_scalar(x) = 0
    source_vector(x) = [cos(x[1])*sinh(x[2]), sin(x[1])*cosh(x[2])]
    p_sol(x) = -sin(x[1])*sinh(x[2]) - (cos(1) - 1)*(cosh(1) - 1)
    u_sol(x) = 2*[cos(x[1])*sinh(x[2]), sin(x[1])*cosh(x[2])]
    p_bdry(x) = p_sol(x)
    if mesh_choice == "refinement"
        source_scalar(x) = -40 * π^2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
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
    # for i = 1:3
        println("Mesh ", i)
        if problem_choice == "divdiv"
            # Problem matrices and preconditioning
            M, MassMat = AuxPrecond.assemble_mixed_energy_matrix(meshes[i], k);
            M_diag = spdiagm(diag(M))
            
            P = [] 
            M_prec = []
            if preconditioner_choice == "additive"
                P = AuxPrecond.AuxPreconditioner(smoother_choice, meshes[i]);
                M_prec = AuxPrecond.apply_precond_to_mat(P, M)
            else
                P = AuxPrecondMultiplicative.AuxPreconditionerMultiplicative(smoother_choice, meshes[i]);
                M_prec = AuxPrecondMultiplicative.apply_precond_to_mat(P, M)
            end
            b = Mixed.assemble_divdiv_rhs(meshes[i], k, source_vector, MassMat)
            restart = size(b, 1)

            _, log_unpr = gmres(M, b, restart=restart, log=true);
            _, log_diag = gmres(M_diag\M, M_diag\b, restart=restart, log=true);
            _, log_prec = gmres(M, b, Pl=P, restart=restart, log=true);
            println("GMRES aux prec done")
            # _, log_unpr = cg(M, b, log=true);
            # _, log_diag = cg(M_diag\M, M_diag\b, log=true);
            # _, log_prec = cg(M, b, Pl=P, log=true);
            # println("PCG aux prec done")

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
                cond_list = [cond_list..., "-1"]
                diagcond_list = [diagcond_list..., "-1"]
            end

            eigs_prec = extrema(abs.(eigvals(collect(M_prec))))
            auxcond_list = [auxcond_list..., eigs_prec[2] / eigs_prec[1]]
        else
            # Problem matrices and preconditioning
            A,M = Mixed.assemble_lhs(meshes[i], k)
            # P_mass = spdiagm(vcat(diag(M), ones(Meshing.get_num_cells(meshes[i]))))
            P_mass = spdiagm(vcat(diag(M), diag(Primal.assemble_mass_matrix(meshes[i], k))))
            if preconditioner_choice == "additive"
                P = AuxPrecond.AuxPreconditioner_Darcy(smoother_choice,meshes[i]);
            else
                P = AuxPrecondMultiplicative.AuxPreconditionerMult_Darcy(smoother_choice,meshes[i]);
            end
            println("Constructing precond done")
            
            b = Mixed.assemble_rhs(meshes[i], k, source_scalar, source_vector, p_bdry, M)
            println("RHS assembly done")

            restart = size(b, 1)
            # restart = 10
            _, log_unpr = gmres(A, b, restart=restart, log=true);
            _, log_diag = gmres(P_mass\A, P_mass\b, restart=restart, log=true);
            _, log_prec = gmres(A, b, Pl=P, restart=restart, log=true);
            println("GMRES aux prec done")
            # _, log_unpr = cg(A, b, log=true);
            # _, log_diag = cg(P_mass\A, P_mass\b, log=true);
            # _, log_prec = cg(A, b, Pl=P, log=true);
            # println("PCG aux prec done")

            gmres_list = [gmres_list..., log_unpr.iters]
            diaggmres_list = [diaggmres_list..., log_diag.iters]
            auxgmres_list = [auxgmres_list..., log_prec.iters]

            # Aspect ratios as indicator for cond nbrs 
            areas = meshes[i].cell_areas
            diams = meshes[i].cell_diams
            if mesh_choice == "aspectratio"
                diam_list[1] = "Aspect ratios"
                aspect_ratios = diams .^ 2 ./ areas
                diam_list = [diam_list..., maximum(aspect_ratios)]
            else
                diam_list = [diam_list..., maximum(diams)]
            end
            # Condition numbers
            if (do_all_cond && mesh_choice=="aspectratio") || (do_all_cond && mesh_choice=="refinement" && i < 4)
                # cond_list = [cond_list..., cond(collect(A),2)]
                # diagcond_list = [diagcond_list..., cond(collect(P_mass\A),2)]
                eigs = extrema(abs.(eigvals(collect(A))))
                cond_list = [cond_list..., eigs[2] / eigs[1]]
                eigs_diag = extrema(abs.(eigvals(collect(P_mass\A))))
                diagcond_list = [diagcond_list..., eigs_diag[2] / eigs_diag[1]]
                if preconditioner_choice == "additive"
                    A_prec = AuxPrecond.apply_precond_to_mat(P, A)
                else
                    A_prec = AuxPrecondMultiplicative.apply_precond_to_mat(P, A)
                end
                # auxcond_list = [auxcond_list..., cond(collect(A_prec),2)]
                eigs_prec = extrema(abs.(eigvals(collect(A_prec))))
                auxcond_list = [auxcond_list..., eigs_prec[2] / eigs_prec[1]]
            else
                cond_list = [cond_list..., "-1"]
                diagcond_list = [diagcond_list..., "-1"]
                auxcond_list = [auxcond_list..., "-1"]
            end
            # _,s,_ = svd(collect(AuxPrecond.apply_precond_to_mat(P, A)))
            # histogram(s, bins=100, title="Singular values of preconditioned A", label="Mesh $i", xlabel="s", ylabel="Frequency")
        end
    end

    [diam_list cond_list diagcond_list auxcond_list gmres_list diaggmres_list auxgmres_list]
    table = [diam_list cond_list diagcond_list auxcond_list gmres_list diaggmres_list auxgmres_list];
    export_result_to_file(mesh_choice, problem_choice, preconditioner_choice, smoother_choice, table)
end

## -------------- meshes --------------#
aspectratio_meshes = construct_meshes("aspectratio")
refinement_meshes = construct_meshes("refinement")

## -------------- plots and dofs --------------#
# using Plots
# plt1 = Meshing.draw_mesh(aspectratio_meshes[1])
# savefig(plt1, "../../KTH/Projects/vem/local_tex/figure/aspratmesh_0.pdf")
# Meshing.draw_line_on_mesh(plt1, aspectratio_meshes[1], x -> x[2] - (0.5 + 10^(-2)))
# savefig(plt1, "../../KTH/Projects/vem/local_tex/figure/aspratmesh_eps.pdf")

# plt4 = Meshing.draw_mesh(refinement_meshes[1])
# savefig(plt4, "../../KTH/Projects/vem/local_tex/figure/diamondmesh_4.pdf")
# plt8 = Meshing.draw_mesh(refinement_meshes[2])
# savefig(plt8, "../../KTH/Projects/vem/local_tex/figure/diamondmesh_8.pdf")

# for i=eachindex(refinement_meshes)
#     size_tuple = size(refinement_meshes[i].cell_faces)
#     println("DOFS divdiv: ", size_tuple[1])
#     println("DOFS darcy:  ", size_tuple[1]+size_tuple[2])
# end

## -------------- tests --------------#
# " CHOOSE MESH / TEST " # mesh_choice = "aspectratio","refinement"
# " CHOOSE PROBLEM " # problem_choice = "divdiv","darcy"
# " CHOOSE SMOOTHER " # smoother_choice = "energy","face"
# " CHOOSE PRECONDITIONER" # preconditioner_choice = "additive","multiplicative"
# " SKIP REDO CONDITION NUMBER " # do_all_cond = true,false

tests_for_paper(aspectratio_meshes, "aspectratio", "divdiv", "energy", "additive", true) # true
tests_for_paper(aspectratio_meshes, "aspectratio", "divdiv", "face", "additive", false)
tests_for_paper(aspectratio_meshes, "aspectratio", "divdiv", "energy", "multiplicative", true)
tests_for_paper(aspectratio_meshes, "aspectratio", "divdiv", "face", "multiplicative", false)

tests_for_paper(refinement_meshes, "refinement", "divdiv", "energy", "additive", false) # true
tests_for_paper(refinement_meshes, "refinement", "divdiv", "face", "additive", false)
tests_for_paper(refinement_meshes, "refinement", "divdiv", "energy", "multiplicative", true)
tests_for_paper(refinement_meshes, "refinement", "divdiv", "face", "multiplicative", false)

tests_for_paper(aspectratio_meshes, "aspectratio", "darcy", "energy", "additive", true) # true
tests_for_paper(aspectratio_meshes, "aspectratio", "darcy", "face", "additive", false)
tests_for_paper(aspectratio_meshes, "aspectratio", "darcy", "energy", "multiplicative", true)
tests_for_paper(aspectratio_meshes, "aspectratio", "darcy", "face", "multiplicative", false)

tests_for_paper(refinement_meshes, "refinement", "darcy", "energy", "additive", true) # true
tests_for_paper(refinement_meshes, "refinement", "darcy", "face", "additive", false)
tests_for_paper(refinement_meshes, "refinement", "darcy", "energy", "multiplicative", true)
tests_for_paper(refinement_meshes, "refinement", "darcy", "face", "multiplicative", false)
using LinearAlgebra
using SparseArrays

using Revise
using VirtuE # [Note for future: make sure every module has last line = end, otherwise really nasty error which are hard to trace]

##-------------- (u,v)+(divu,divv) debug section --------------#
levelset(x) = x[2] - (0.5 + 1e-7)

k = 0 # Polynomial degree

N = 10 # size of mesh
mesh = Meshing.create_tri_mesh(N)
mesh = Meshing.remesh(mesh, levelset)

num_faces = Meshing.get_num_faces(mesh)

M = AuxPrecond.assemble_mixed_energy_matrix(mesh, k, x -> 1)
M_prec = AuxPrecond.apply_aux_precond(M, mesh, k)

eigs = (eigvals(collect(M)))
maximum(eigs) / minimum(eigs)

eigs_prec = (eigvals(collect(M_prec)))
maximum(eigs_prec) / minimum(eigs_prec)

# ##-------------- (u,v)+(divu,divv) refinement tests --------------#
k = 0 # Polynomial degree
N = 10 # size of mesh

eps = [];
cond_nbrs = [];
cond_nbrs_prec = [];
for j = 3:7
    append!(eps, 10.0^(-j))
    mesh = Meshing.create_tri_mesh(N)

    levelset(x) = x[2] - (0.5 + 10.0^(-j))
    mesh = Meshing.remesh(mesh, levelset)

    M = AuxPrecond.assemble_mixed_energy_matrix(mesh, k, x -> 1)
    M_prec = AuxPrecond.apply_aux_precond(M, mesh, k)

    eigs = eigvals(collect(M))
    append!(cond_nbrs, maximum(eigs) / minimum(eigs))
    eigs_prec = eigvals(collect(M_prec))
    append!(cond_nbrs_prec, maximum(eigs_prec) / minimum(eigs_prec))
    # append!(cond_nbrs, cond(Array(M), 1))
    # append!(cond_nbrs_prec, cond(Array(M_prec), 1))

    # areas = mesh.cell_areas
    # area_ratio = minimum(areas) / maximum(areas)
    # println(area_ratio)

    AuxPrecond.
    Meshing.draw_mesh(mesh)
    println("Proximity length ", j, " done.")
end
# convgs = log.(errors[2:end] ./ errors[1:end-1]) ./ log.(h[2:end] ./ h[1:end-1])#log(evec[i]/evec[i-1])/log(h[i]/h[i-1])
display(hcat([eps, cond_nbrs, cond_nbrs_prec]))

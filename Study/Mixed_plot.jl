using LinearAlgebra
using SparseArrays
using Plots

using Revise
using VirtuE

##-------------- Initialise mesh --------------#
N = 10 # size of mesh
mesh = Meshing.create_tri_mesh(N)
# mesh = create_rect_mesh(N)
# mesh = create_pentagon_mesh()

##-------------- Remesh wrt levelset --------------#
radius = 1 / sqrt(11) # [1/sqrt(11) gets a REAALLY bad cut]
center = [0.5, 0.5]
# radius = 1/sqrt(2)
# center = [0, 0]

# <0:inside, >0:outside
levelset(x) = norm(x - center) - radius
mesh = Meshing.remesh(mesh, levelset)

Meshing.draw_mesh(mesh)

##-------------- Testing ground --------------#

k = 0 # Polynomial degree
μ(x) = [1 0; 0 1]

# source(x) = 0
# p_bdry(x) = 1 - x[1]
# p_sol(x) = p_bdry(x)
# u_sol(x) = [-1, 0]

source(x) = 40 * π^2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
p_bdry(x) = 0
p_sol(x) = 2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
u_sol(x) = -[4π * cos(2π * x[1]) * sin(4π * x[2]), 8π * cos(4π * x[2]) * sin(2π * x[1])]

# # No preconditioner
# A,b,ξ = Mixed.Darcy_setup(mesh,k,source,p_bdry,μ)
# # Preconditioner
# A = Mixed.assemble_lhs(mesh, k, μ)
# b = Mixed.assemble_rhs(mesh, k, source, p_bdry)
# P = AuxPrecond.AuxPreconditioner_Darcy(mesh)
# A = AuxPrecond.apply_precond_to_mat(P, A)
# b = vec(AuxPrecond.apply_precond_to_mat(P, b))
# ξ = A \ b
# Preconditioner with gmres
using IterativeSolvers
A = Mixed.assemble_lhs(mesh, k, μ)
b = Mixed.assemble_rhs(mesh, k, source, p_bdry)
P = AuxPrecond.AuxPreconditioner_Darcy(mesh)
restart = size(b, 1)
ξ, _ = gmres(A, b, Pl=P, restart=restart, log=true)


u = ξ[1:Meshing.get_num_faces(mesh)]
p = ξ[Meshing.get_num_faces(mesh)+1:end]

# print(norm(p - p_true))
println(Mixed.norm_L2(mesh, k, p, p_sol))
println(Mixed.norm_L2(mesh, k, u, u_sol))


##-------------- Interpolate using Dierckx.jl, then plot --------------#
using Dierckx

p_true = [p_sol(x) for x in eachrow(mesh.cell_centroids)]
# u_true = vcat([u_sol(x) for x in eachrow(mesh.cell_faces)]...)

# Pressure
xC_list = unique(sort(mesh.cell_centroids[:,1]))
yC_list = unique(sort(mesh.cell_centroids[:,2]))

ph_spl = Dierckx.Spline2D(mesh.cell_centroids[:,1],mesh.cell_centroids[:,2],p,s=1) # s is "smoothing factor", needs >0.0 default to work
Zph = evalgrid(ph_spl,xC_list,yC_list)

heatmap(xC_list, yC_list, Zph',
    #    c=cgrad([:blue,:white,:red]),
        xlabel="x", ylabel="y", xlims = (0,1), ylims=(0,1),
        title="VEM p", aspect_ratio = :equal)

p_spl = Dierckx.Spline2D(mesh.cell_centroids[:,1],mesh.cell_centroids[:,2],p_true,s=1)
Zp = evalgrid(p_spl,xC_list,yC_list)
hms = [heatmap(xC_list, yC_list, Zph',
#        c=cgrad([:blue,:white,:red]),
        xlabel="x", ylabel="y", xlims = (0,1), ylims=(0,1),
        title="Prec VEM solution", aspect_ratio = :equal),
       heatmap(xC_list, yC_list, Zp',
#        c=cgrad([:blue,:white,:red]),
        xlabel="x", ylabel="y", xlims = (0,1), ylims=(0,1),
        title="True solution", aspect_ratio = :equal)]
plt = plot(hms..., layout = (1,2), colorbar = true)
# savefig(plt,"sols.png")

# Velocity
# Plots.scatter(get_face_centers(mesh)[:,1],get_face_centers(mesh)[:,2])
# xF_list = unique(sort(get_face_centers(mesh)[:,1]))
# yF_list = unique(sort(get_face_centers(mesh)[:,2]))

# uh_spl = Dierckx.Spline2D(get_face_centers(mesh)[:,1],get_face_centers(mesh)[:,2],u,s=2) # s is "smoothing factor", needs >0.0 default to work
# Zuh = evalgrid(ph_spl,xF_list,yF_list)

# u_plt = heatmap(xF_list, yF_list, Zuh',
#     #    c=cgrad([:blue,:white,:red]),
#         xlabel="x", ylabel="y", xlims = (0,1), ylims=(0,1),
#         title="VEM u", aspect_ratio = :equal)


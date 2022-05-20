using LinearAlgebra
using Plots

using Revise
using VirtuE

##-------------- Initialise mesh --------------#
N = 30 # size of mesh
mesh = Meshing.create_tri_mesh(N)
# mesh = Meshing.create_rect_mesh(N)
# mesh = Meshing.create_pentagon_mesh()

radius = 1 / sqrt(11) # [1/sqrt(11) gets a REAALLY bad cut]
center = [0.5, 0.5]
levelset(x) = LinearAlgebra.norm(x - center) - radius

##-------------- Problem setup --------------#

k = 1 # Polynomial degree
μ(x) = (levelset(x) < 0) * 1 + (levelset(x) >= 0) * 1
source(x) = 40 * π^2 * sin(2 * π * x[1]) * sin(4 * π * x[2])
p_bdry(x) = 0

u = Primal.Darcy_setup(mesh, source, p_bdry, μ, k)[2]

## ------------ plotting ------------

Plots.scatter(mesh.node_coords[:, 1], mesh.node_coords[:, 2], vec(u), aspect_ratio=:equal)
Plots.scatter(mesh.node_coords[:, 1], mesh.node_coords[:, 2], zcolor=vec(u), aspect_ratio=:equal)

##-------------- Interpolate using Dierckx.jl, then plot --------------#
# Pkg.add("Dierckx")
using Dierckx

xlist = unique(sort(mesh.node_coords[:, 1]))
ylist = unique(sort(mesh.node_coords[:, 2]))

# s is "smoothing factor", needs >0.0 default to work
vem_spl = Dierckx.Spline2D(mesh.node_coords[:, 1], mesh.node_coords[:, 2], u[:, 1], s=1)
sol_spl = Dierckx.Spline2D(mesh.node_coords[:, 1], mesh.node_coords[:, 2], u_true, s=1)
ZZ = evalgrid(vem_spl, xlist, ylist)
ZZtrue = evalgrid(sol_spl, xlist, ylist)

heatmap(xlist, ylist, ZZ',
    #        c=cgrad([:blue,:white,:red]),
    xlabel="x", ylabel="y", xlims=(0, 1), ylims=(0, 1),
    title="VEM solution", aspect_ratio=:equal)

hms = [heatmap(xlist, ylist, ZZ',
        #        c=cgrad([:blue,:white,:red]),
        xlabel="x", ylabel="y", xlims=(0, 1), ylims=(0, 1),
        title="VEM solution", aspect_ratio=:equal),
    heatmap(xlist, ylist, ZZtrue',
        #        c=cgrad([:blue,:white,:red]),
        xlabel="x", ylabel="y", xlims=(0, 1), ylims=(0, 1),
        title="True solution", aspect_ratio=:equal)]

plt = plot(hms..., layout=(1, 2), colorbar=true)
savefig(plt, "sols.png")
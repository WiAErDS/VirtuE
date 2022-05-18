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

# <0:inside, >0:outside
levelset(x) = LinearAlgebra.norm(x - center) - radius

mesh, itf_faces = Meshing.remesh(mesh, levelset, get_itf_faces = true)

println("Area ratio before: ", minimum(mesh.cell_areas) / maximum(mesh.cell_areas))

macro_cells = MacroCell.init_macro(mesh, 0.1, itf_faces)
mesh = MacroCell.remesh(mesh, macro_cells)

println("Area ratio after: ", minimum(mesh.cell_areas) / maximum(mesh.cell_areas))

# mesh_plt = Meshing.draw_mesh(mesh)
# savefig(mesh_plt, "mesh.png")
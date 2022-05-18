
############################## TODO #########################
## LSM:
# (1.5 Write more stable time FD, optionally needed)
# 2. Fix Reinit
# 3. Get method for calculating curvature flow
# 4. Implement ways of getting curves.. need sampled data

## VEM:
# Find general way to compute diameter from coordinates ✓
# Make sparse instead of boolean, findall ✓
# Implement rudimentary plotting ✓
# Implement quadrature integration (using MiniQhull) ✓
# Permeability constant depend on each element? Implement ✓
# Setup cutfem with the same geometry and problem, compare condition numbers ✓
# Implement proper L2 norm using L2 projection of VEM sol ✓
# Implement natural macro elements
# Rename mesh.nodes to something like mesh.coords
# Remake itf_faces to directly be itf_faces = findnz(itf_faces)[1]?
# Find mu field radial example (or Hansbo example)
# Script that exports Julia->Paraview
# Higher order k=2 VEM
# Mixed VEM
# Make code files into Modules for readability (where is that function from?)
# Erik: Delete Delaunay library
# Erik: Look more into Wietse changes, direct mesh & poly mesh
# Erik: Julia Modules??

############################## NOTES #########################
#-------------- On determinant --------------#
# det function in LinearAlgebra is UNSTABLE and can return faulty values
# still true?

#-------------- C++ code for rotating calipers for poly diam --------------#
#=
function RotatingCalipers_PolyDiam(vertices)
    vector<Point> h = convexHull(vertices);
    size_t m = h.size();
    if (1 == m) return 0;
    if (2 == m) return dist(h[0], h[1]);

    size_t k = 1;
    while (area(h[m - 1], h[0], h[(k + 1) % m]) > area(h[m - 1], h[0], h[k])) k++;

    double res = 0;
    for (size_t i = 0, j = k; i <= k && j < m; i++)
    {
        res = max(res, dist(h[i], h[j]));
        while (j < m && area(h[i], h[(i + 1) % m], h[(j + 1) % m]) > area(h[i], h[(i + 1) % m], h[j]))
        {
            res = max(res, dist(h[i], h[(j + 1) % m]));
            j++;
        }
    }
    return res;
end
=#

#-------------- Draw abstract plotting geometry --------------#
#=
function draw_mesh_tmp(coords,connect)
    plt = Plots.scatter(coords[:,1], coords[:,2], legend=false, aspect_ratio=:equal)

    for k in 1:size(connect,1)
        vertices = connect[k,:]
        append!(vertices,vertices[1])
        Plots.plot!(plt, coords[vertices,1], coords[vertices,2], linecolor="black")
    end
    display(plt)
end
=#

#-------------- Plot mesh using Makie --------------#
#=
function draw_mesh_with_Makie(mesh)
    Makie.scatter(mesh.node_coords)

    for k = 1:size(mesh.cell_nodes,1)
        idxs = mesh.cell_nodes[k]
        append!(idxs,idxs[1])
        drawNodes = mesh.node_coords[idxs,:] #[nodes[a,:]';nodes[b,:]';nodes[c,:]';nodes[a,:]']
        Makie.lines!(drawNodes)
    end
    current_figure()
end
=#
#-------------- Check correctness of B and D --------------#
#=
coords = CreatePentagon()

k=1
nvx = size(coords, 1) # number of vertices
mon_exps = Monomials.MonExp(k)
nmon = size(mon_exps, 1) # (k+1)*(k+2)/2

D = zeros(nvx,nmon)
B = zeros(nmon,nvx)

# geometry information
# area_k = PolyArea(coords) # Not necessary?
pC = PolyCentroid(coords)
h = PolyDiam(coords)

coords=vcat(coords,[0.0 0.0])
connects = [[coords[i,:], coords[i+1,:]] for i=1:(length(coords[:,1])-1)]
edges = [connects[i][2]-connects[i][1] for i=1:length(connects)]
tangents = [edges[i]/norm(edges[i]) for i=1:length(edges)]
trTangents = [convert(Array{Float32},tangents[i]') for i=1:length(tangents)]
normals = [nullspace(trTangents[i]) for i=1:length(trTangents)]
# hypothesis: determinant of [tan;nor] is + if -orientation, and - if +orientation (ie nor pointing outside)
normals = [(-sign(det([trTangents[i];normals[i]'])))*normals[i] for i=1:length(normals)]

for j=1:nvx
    B[1,j] = 1/nvx # k=1
    # println(coords[(j-1)*(j!=1)+end*(j==1)])
    for i=1:nmon
        D[j,i] = Monomials.ScaledMon(coords[j,:],pC,h,mon_exps[i,:])
        # coord j = vertex j, bFun j has nonzero edges j & j-1
        if i>1
            grad_mon_i = Monomials.GradMon([1 1],pC,h,mon_exps[i,:])
            B[i,j] = 1/2*(  norm(edges[(j-1)*(j!=1)+end*(j==1)])
                            *dot(grad_mon_i,normals[(j-1)*(j!=1)+end*(j==1)])
                            + norm(edges[j])*dot(grad_mon_i,normals[j]) ) # k=1
                            # TODO: needs a critical look...
        end
    end
end


faces_el = cell_faces.rowval[cell_faces.colptr[1]:cell_faces.colptr[1+1]-1] # Find faces adjacent to cell
face_nodes_el = face_nodes[:, faces_el] * spdiagm(0 => cell_faces[faces_el ,1]) # Find corresponding nodes
nodes_el = create_loop(face_nodes_el)

edges = abs.(transpose(face_nodes) * nodes)
edges_el = edges[faces_el, :]
normals = ([0 1; -1 0] * edges_el')'
=#

#-------------- Check delaunay triangulation --------------#
#=

using MiniQhull
using Plots

coords = CreatePentagon()
coords = coords'
tri = MiniQhull.delaunay(coords)'
coords = coords'

draw_mesh_tmp(coords,tri)

=#

#-------------- "Abstract art" plots, to include at end of computation --------------#

#=
xlist = unique(sort(mesh.nodes[:,1]))
ylist = unique(sort(mesh.nodes[:,2]))
Z = sparse(zeros(length(xlist),length(ylist)))
Ztrue = sparse(zeros(length(xlist),length(ylist)))
i=1;j=1; idx=0
for x in xlist
    for y in ylist
        idx = findall(Bool[ [x, y] == mesh.nodes[i,:] for i=1:size(mesh.nodes,1) ])
        if length(idx)>0
            Z[i,j] = u[idx[1]]
            Ztrue[i,j] = u_true[idx[1]]
        end
        j = j+1
    end
    j = 1
    i = i+1
end
# x,y,matrix
heatmap(xlist, ylist, Z,
    c=cgrad([:blue,:white,:red]),
    xlabel="x", ylabel="y",
    title="VEM solution")
heatmap(xlist, ylist, Ztrue,
    c=cgrad([:blue,:white,:red]),
    xlabel="x", ylabel="y",
    title="True solution")

=#

#-------------- Tried to interpolate using Interpolations pkg, failed --------------#

# Pkg.add("Interpolations")
# using Interpolations

# nodes = ([x for x in xlist[findnz(Z)[1]]], [y for y in ylist[findnz(Z)[2]]])
# ZZ = interpolate(nodes, findnz(Z)[3], Gridded(Linear()))

# ZZ = interpolate((xlist,ylist),Z,Gridded(Linear()))
# ZZ = interpolate(Z, BSpline(Cubic(Line(OnGrid()))))
# ZZ = interpolate(Z, BSpline(Linear()))
# heatmap(xlist, ylist, ZZ,
#     c=cgrad([:blue,:orange,:red]),
#     xlabel="x", ylabel="y",
#     title="VEM solution")

# num_x = length(xlist); num_y = length(ylist)
# for i = 1:num_x
#     nzs = findnz(Z[i,:])
#     nz_idx = (nzs[1],); nz_vals = nzs[2]; num_nz = length(nz_vals)
#     if num_nz > 1
#         # itp = interpolate(nz_vals,BSpline(Cubic()))
#         itp = interpolate(nz_idx,nz_vals,Gridded(Linear()))
#         #println(length(itp(1:(num_nz-0.999999)/num_x:length(itp))),num_nz)
#         Z[i,:] = itp(1:(num_nz-0.999999)/num_x:num_nz) # -> 48 element interpolant
#     end
#     for j = 1:num_y
#         nz_vals = findnz(Z[:,j])[2]; num_nz = length(nz_vals)
#         if num_nz > 1
#             itp = interpolate(nz_vals,BSpline(Linear()))
#             #println(length(itp(1:(num_nz-0.999999)/num_x:length(itp))),num_nz)
#             Z[:,j] = itp(1:(num_nz-0.999999)/num_x:num_nz) # -> 48 element interpolant
#         end
#     end
# end

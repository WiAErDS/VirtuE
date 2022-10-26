module Meshing

using SparseArrays
using LinearAlgebra
using Plots

struct Mesh
    node_coords::Array{Float64,2} # mesh.node_coords[mesh.cell_nodes[1],:] # dim = nodes x 2
    cell_faces::SparseMatrixCSC
    face_nodes::SparseMatrixCSC

    cell_nodes::Array{Array{Int64,1},1}
    cell_areas::Array{Float64,1}
    cell_centroids::Array{Float64,2}
    cell_diams::Array{Float64,1}
end

function Mesh(node_coords::Array{Float64,2}, cell_faces::SparseMatrixCSC, face_nodes::SparseMatrixCSC)

    @assert(face_nodes * cell_faces == spzeros(face_nodes.m, cell_faces.n)) # div curl = 0

    cell_nodes = compute_cell_nodes(cell_faces, face_nodes)

    cell_areas = zeros(size(cell_faces, 2))
    cell_centroids = zeros(size(cell_faces, 2), 2)
    cell_diams = zeros(size(cell_faces, 2))

    for (cell, nodes) in enumerate(cell_nodes)
        cell_areas[cell] = get_poly_area(node_coords[nodes, :])
        cell_centroids[cell, :] = get_poly_centroid(node_coords[nodes, :])
        cell_diams[cell] = get_poly_diam(node_coords[nodes, :])
    end

    return Mesh(node_coords, cell_faces, face_nodes, cell_nodes, cell_areas, cell_centroids, cell_diams)
end

# """ Convenience function to get the node idx from a coord
# - coord array, eg [0.34, 0.2] (NOTE must be in this form of 2-element array)
# returns the index node s.t. node_coords[idx] = [0.34, 0.2]
# """
# function get_node_from_coord(coord::Array{Float64,1}, node_coords)
#     qualified = [norm(coord-row) < 1e-3 for row in eachrow(node_coords)]
#     return findall(qualified)
# end

function get_rowvals(A::SparseMatrixCSC, ind::Int)
    #Because I got tired of doing this over and over again....
    return A.rowval[nzrange(A, ind)]
end

function create_node_loop(face_nodes)
    # Create a positively oriented loop over all nodes of an element
    i_node, i_face, i_value = findnz(face_nodes)

    nodes = zeros(Int, size(face_nodes, 2))
    nodes[1] = i_node[1]

    for j in eachindex(nodes)[2:end]
        next_face = i_face[findfirst((i_node .== nodes[j-1]) .& (i_value .< 0))]
        nodes[j] = i_node[findfirst((i_face .== next_face) .& (i_value .> 0))]
    end

    return nodes
end

function compute_cell_nodes(cell_faces, face_nodes)

    cell_nodes = fill(Int[], size(cell_faces, 2))

    for el in axes(cell_faces, 2)
        faces_el = get_rowvals(cell_faces, el) # Find faces adjacent to cell
        face_nodes_el = face_nodes[:, faces_el] * spdiagm(0 => cell_faces[faces_el, el]) # Extract positively oriented face_node connectivity
        cell_nodes[el] = create_node_loop(face_nodes_el)
    end

    return cell_nodes
end

# ------------ mesh data ------------

function get_num_cells(grid)
    return size(grid.cell_faces, 2)
end

function get_num_faces(grid)
    return size(grid.face_nodes, 2)
end

function get_num_nodes(grid)
    return size(grid.node_coords, 1)
end

function get_bdry_dofs(grid)
    # Returns a Boolean array of the boundary nodes and faces

    bdryfaces = sum(abs.(grid.cell_faces), dims=2) .== 1
    bdrynodes = abs.(grid.face_nodes) * bdryfaces .> 0

    return bdrynodes, bdryfaces
end

function get_tangents(grid, face_id=Colon())
    return grid.face_nodes[:, face_id]' * grid.node_coords
end

function get_face_normals(grid, face_id=Colon())
    return get_tangents(grid, face_id) * [0 -1; 1 0]
end

function get_face_centers(grid, face_id=Colon())
    return abs.(grid.face_nodes[:, face_id]') * grid.node_coords / 2
end


function get_poly_area(vertices) # @wikipedia polygon
    v1 = vertices
    v2 = circshift(vertices, -1)

    return 1 / 2 * abs(dot(v1[:, 1], v2[:, 2]) - dot(v2[:, 1], v1[:, 2]))
end

function get_poly_centroid(vertices) # @wikipedia polygon
    A = get_poly_area(vertices)
    v1 = vertices
    v2 = circshift(vertices, -1)
    c = 1 / (6A) * (v1[:, 1] .* v2[:, 2] - v2[:, 1] .* v1[:, 2])' * (v1 + v2)

    return vec(c)
end

function get_poly_diam(vertices) # O(n^2) algo for any simple polygon
    distances = [norm(p - q) for p in eachrow(vertices), q in eachrow(vertices)]
    return maximum(distances)
end

#-------------- Alternative call using mesh and cell nr --------------#
function get_poly_area(mesh, cell::Int)
    return mesh.cell_areas[cell]
end

function get_poly_centroid(mesh, cell::Int)
    return mesh.cell_centroids[cell, :]
end

function get_poly_diam(mesh, cell::Int)
    return mesh.cell_diams[cell]
end

#-------------- Regular triangle mesh --------------#
function create_tri_mesh(n)
    # n is #BOXES per row/col
    # indices start at 0 => index #i = point #(i+1)
    m = n + 1 # nbr of nodes in a row/col
    h = 1 / n

    nn = (n + 1)^2 # one extra node per row, n_n = (n+1)^2
    nt = (n^2) * 2 # n^2 boxes, for every box 2 triangles
    ne = (nt + 1) + nn - 2 # nbr edges, from Euler's formula (outside of graph is counted as face)

    x_list = 0:h:1
    nodes = hcat(repeat(x_list, m), repeat(x_list, inner=m))

    I_h = [[k, k + 1] for j in 1:m:m*m for k in j:j+n-1] # horizontal ⇒
    I_v = [[k, k + m] for j in 1:m:m*n for k in j:j+n] # vertical ⇑
    I_d = [[k + 1 + m, k] for j in 1:m:m*n for k in j:j+n-1] # diagonal ⇙
    I = vcat(I_h..., I_v..., I_d...)

    J = [[e, e] for e in 1:ne] # boxes x nodes = nbr faces
    J = vcat(J...)

    V = ones(Int, length(I))
    V[1:2:end] *= -1 # assign orientation

    face_nodes = sparse(I, J, V)

    # verticals need to make one jump per row
    I_r = [[k, k + 1 + n * m + (j - 1) / n, k + 2n * m] for j in 1:n:n^2 for k in j:j+n-1] # faces
    I_l = [[k + n, k + n * m + (j - 1) / n, k + 2n * m] for j in 1:n:n^2 for k in j:j+n-1]
    J_r = [[e, e, e] for e in 1:2:nt]
    J_l = [[e, e, e] for e in 2:2:nt]
    I = vcat(I_r..., I_l...)
    J = vcat(J_r..., J_l...)
    V = ones(Int, length(I))
    V[length(I)÷2+1:end] *= -1 # Upper-left triangles
    cell_faces = sparse(I, J, V)

    return Mesh(nodes, cell_faces, face_nodes)
end

#-------------- Regular rectangular mesh --------------#
function create_rect_mesh(n)
    # n is #BOXES per row/col
    # indices start at 0 => index #i = point #(i+1)
    m = n + 1 # nbr of nodes in a row/col
    h = 1 / n

    nn = (n + 1)^2 # one extra node per row, n_n = (n+1)^2
    nb = (n^2) # n^2 boxes
    ne = (nb + 1) + nn - 2 # nbr edges, from Euler's formula (outside of graph is counted as face)

    x_list = 0:h:1
    nodes = hcat(repeat(x_list, m), repeat(x_list, inner=m))

    I_h = [[k, k + 1] for j in 1:m:m*m for k in j:j+n-1] # horizontal ⇒
    I_v = [[k, k + m] for j in 1:m:m*n for k in j:j+n] # vertical ⇑
    I = vcat(I_h..., I_v...)

    J = [[e, e] for e in 1:ne] # boxes x nodes = nbr faces
    J = vcat(J...)

    V = ones(Int, length(I))
    V[1:2:end] *= -1 # assign orientation

    face_nodes = sparse(I, J, V)

    # verticals need to make one jump per row
    I = [[k, k + 1 + n * m + (j - 1) / n, k + n, k + n * m + (j - 1) / n] for j in 1:n:n^2 for k in j:j+n-1] # faces
    J = [[e, e, e, e] for e in 1:nb]
    V = [[1, 1, -1, -1] for e in 1:nb]
    I = vcat(I...)
    J = vcat(J...)
    V = vcat(V...)

    cell_faces = sparse(I, J, V)

    return Mesh(nodes, cell_faces, face_nodes)
end


#-------------- Remeshing --------------#

## Handy mappings
function create_new_entity_map(cut_entities, offset=0)
    # Mapping of n_new x n_old in which (i_new, i_old) = 1 if i_new is a new entity placed on i_old
    n = sum(cut_entities)
    return sparse((1:n) .+ offset, cut_entities.nzind, trues(n), n + offset, length(cut_entities))
end

function create_splitting_map(indices, offset=0)
    # Mapping of n_new x n_old in which (i_new, i_old) = 1 if i_new is a split of i_old
    I = 1:2*sum(indices)
    J = repeat(indices.nzind, inner=2)
    V = trues(2 * sum(indices))

    return sparse(I .+ offset, J, V, 2 * sum(indices) + offset, length(indices))
end

## introducing new nodes and marking
function mark_intersections(mesh, levelset)
    new_node_coords = Array{Float64,1}[]
    new_node_on_faces = spzeros(Int, get_num_faces(mesh))

    for face = 1:get_num_faces(mesh) # loop over faces
        face_vertices = get_rowvals(mesh.face_nodes, face) # size 2 vector
        levelset_vals = [levelset(mesh.node_coords[vertex, :]) for vertex in face_vertices]

        if prod(levelset_vals) < 0

            v_coords = mesh.node_coords[face_vertices, :]

            t0 = levelset_vals[1] / (levelset_vals[1] - levelset_vals[2])
            cut_point = v_coords[1, :] + t0 * (v_coords[2, :] - v_coords[1, :])
            push!(new_node_coords, cut_point)
            new_node_on_faces[face] = length(new_node_coords)

        elseif prod(levelset_vals) == 0
            error("Level set passes exactly through a node")
        end
    end

    cut_faces = new_node_on_faces .!= 0
    cut_cells = abs.(mesh.cell_faces') * cut_faces .> 0
    new_node_coords = hcat(new_node_coords...)'

    @assert(maximum(abs.(mesh.cell_faces)' * cut_faces) <= 2, "A cell has more than two cut faces")

    return cut_cells, cut_faces, new_node_coords, new_node_on_faces
end

## Create new faces

function create_new_face_nodes(mesh, cut_cells, cut_faces, entity_maps)
    I_fn = Int[]
    J_fn = Int[]
    V_fn = Int[]

    for cell in cut_cells.nzind
        faces_el = get_rowvals(mesh.cell_faces, cell)
        cut_faces_el = faces_el[cut_faces[faces_el]]
        new_nodes_el = [get_rowvals(entity_maps[0, 1], face)[1] for face in cut_faces_el]

        append!(I_fn, sort(new_nodes_el))
        append!(J_fn, repeat(get_rowvals(entity_maps[1, 2], cell), 2))
        append!(V_fn, [-1 1])
    end

    for face in cut_faces.nzind
        nodes = get_rowvals(mesh.face_nodes, face)
        orientation = mesh.face_nodes[nodes, face]

        new_faces = get_rowvals(entity_maps[1, 1], face)
        new_node = get_rowvals(entity_maps[0, 1], face)[1]

        append!(I_fn, [nodes[1], new_node, new_node, nodes[2]])
        append!(J_fn, repeat(new_faces, inner=2))
        append!(V_fn, repeat(orientation, 2))
    end

    return sparse(I_fn, J_fn, V_fn)
end

## Add new cells
function create_new_cell_faces(mesh, cut_cells, cut_faces, entity_maps, face_nodes)
    I_cf = Int[]
    J_cf = Int[]
    V_cf = Int[]

    for el in cut_cells.nzind
        new_cells = get_rowvals(entity_maps[2, 2], el)
        faces_el = get_rowvals(mesh.cell_faces, el)

        face_nodes_el = mesh.face_nodes[:, faces_el] * spdiagm(0 => mesh.cell_faces[faces_el, el]) # Extract positively oriented face_node connectivity

        (I, J, V) = findnz(face_nodes_el)
        loop_starts = I[(V.==1).&cut_faces[faces_el[J]]]
        loop_ends = reverse(I[(V.==-1).&cut_faces[faces_el[J]]])

        nodes_el = mesh.cell_nodes[el]

        for i in 1:2
            # Faces that are uncut
            nodes_el = circshift(nodes_el, 1 - findfirst(nodes_el .== loop_starts[i]))
            sub_nodes = nodes_el[1:findfirst(nodes_el .== loop_ends[i])]
            sub_faces = [faces_el[J[(V.==-1).&(I.==sn)]][1] for sn in sub_nodes[1:end-1]]
            append!(I_cf, sub_faces)
            append!(V_cf, mesh.cell_faces[sub_faces, el])

            # Faces that are cut
            start_face = faces_el[J[(I.==loop_starts[i]).&(V.==1)]][1]
            splits_at_start = get_rowvals(entity_maps[1, 1], start_face)
            face_at_start = splits_at_start[face_nodes[loop_starts[i], splits_at_start].nzind[1]]

            end_face = faces_el[J[(I.==loop_ends[i]).&(V.==-1)]][1]
            splits_at_end = get_rowvals(entity_maps[1, 1], end_face)
            face_at_end = splits_at_end[face_nodes[loop_ends[i], splits_at_end].nzind[1]]

            push!(I_cf, face_at_start, face_at_end)
            append!(V_cf, mesh.cell_faces[[start_face end_face], el])

            # The new face cutting through the element
            cutting_face = get_rowvals(entity_maps[1, 2], el)[1]
            push!(I_cf, cutting_face)
            push!(V_cf, face_nodes[get_rowvals(entity_maps[0, 1], start_face)[1], cutting_face])

            append!(J_cf, repeat([new_cells[i]], 3 + length(sub_faces)))
        end
    end

    return sparse(I_cf, J_cf, V_cf)
end

function remesh(mesh::Mesh, levelset::Function; get_itf_faces=false, keep_old_mesh=false)
    cut_cells, cut_faces, new_node_coords, _ = mark_intersections(mesh, levelset)
    node_coords = vcat(mesh.node_coords, new_node_coords)

    # (0,1) => node_on_face, (1,2) => face_on_cell, (2,2) => cell_on_cell, (1,1) => face_on_face
    entity_maps = Dict{Tuple{Int64,Int64},SparseMatrixCSC{Bool,Int64}}()
    entity_maps[0, 1] = create_new_entity_map(cut_faces, get_num_nodes(mesh))
    entity_maps[1, 2] = create_new_entity_map(cut_cells, get_num_faces(mesh))
    entity_maps[2, 2] = create_splitting_map(cut_cells, get_num_cells(mesh))
    entity_maps[1, 1] = create_splitting_map(cut_faces, size(entity_maps[1, 2], 1))

    new_face_nodes = create_new_face_nodes(mesh, cut_cells, cut_faces, entity_maps)
    face_nodes = new_face_nodes + sparse(findnz(mesh.face_nodes)..., size(new_face_nodes)...)

    new_cell_faces = create_new_cell_faces(mesh, cut_cells, cut_faces, entity_maps, face_nodes)
    cell_faces = new_cell_faces + sparse(findnz(mesh.cell_faces)..., size(new_cell_faces)...)

    itf_faces = sparsevec(rowvals(entity_maps[1, 2]), trues(nnz(entity_maps[1, 2])), face_nodes.n)

    new_cell_nodes = compute_cell_nodes(new_cell_faces[:, get_num_cells(mesh)+1:end], face_nodes)
    cell_nodes = vcat(mesh.cell_nodes, new_cell_nodes)

    new_areas = [get_poly_area(node_coords[nodes, :]) for nodes in new_cell_nodes]
    new_centroids = [get_poly_centroid(node_coords[nodes, :]) for nodes in new_cell_nodes]
    new_diams = [get_poly_diam(node_coords[nodes, :]) for nodes in new_cell_nodes]

    cell_areas = vcat(mesh.cell_areas, new_areas)
    cell_centroids = vcat(mesh.cell_centroids, new_centroids'...)
    cell_diams = vcat(mesh.cell_diams, new_diams)

    if !keep_old_mesh
        keep_cells = vcat(.!cut_cells, trues(2 * sum(cut_cells)))
        keep_faces = vcat(.!cut_faces, trues(2 * sum(cut_faces) + sum(cut_cells)))

        # TODO: do these operations using restrictions
        cell_faces = cell_faces[keep_faces, keep_cells]
        face_nodes = face_nodes[:, keep_faces]
        cell_nodes = cell_nodes[keep_cells]
        itf_faces = itf_faces[keep_faces]

        cell_areas = cell_areas[keep_cells]
        cell_centroids = cell_centroids[keep_cells, :]
        cell_diams = cell_diams[keep_cells]
    end

    new_mesh = Mesh(node_coords, cell_faces, face_nodes, cell_nodes, cell_areas, cell_centroids, cell_diams)

    if get_itf_faces
        return new_mesh, itf_faces
    else
        return new_mesh
    end
end

""" Throws away part of the mesh consisting of cells all of whose nodes have levelset(node)>0 
Also gives the coordinates of the mesh-interface cuts

"""
function remesh_unfitted(mesh::Mesh, levelset::Function)
    # nodes_to_delete = []
    # faces_to_delete = []
    # cells_to_delete = []
    # for (cell, nodes) in enumerate(mesh.cell_nodes)
    #     faces = mesh.cell_faces[:,cell]
    #     if sum([levelset(mesh.node_coords[node,:]) for node in nodes].<0)==3
    #         append!(cells_to_delete,cell)
    #         append!(nodes_to_delete,nodes)
    #         append!(faces_to_delete,faces)
    #     end
    # end
    # node_coords = node_coords[ 1:end .∉ [nodes_to_delete], 1:end .∉ [[:]] ]
    # face_nodes = face_nodes[ 1:end .∉ [nodes_to_delete], 1:end .∉ [faces_to_delete] ]
    # cell_faces = cell_faces[ 1:end .∉ [faces_to_delete], 1:end .∉ [cells_to_delete] ]
    # cell_nodes = cell_nodes[ 1:end .∉ [cells_to_delete] ]
    # cell_areas = cell_areas[ 1:end .∉ [cells_to_delete] ]
    # cell_centroids = cell_centroids[ 1:end .∉ [cells_to_delete], 1:end .∉ [[:]] ]
    # cell_diams = cell_diams[ 1:end .∉ [cells_to_delete] ]

    nodes_to_keep = falses(get_num_nodes(mesh))
    faces_to_keep = falses(get_num_faces(mesh))
    cells_to_keep = falses(get_num_cells(mesh))
    for (cell, nodes) in enumerate(mesh.cell_nodes)
        faces = get_rowvals(mesh.cell_faces, cell)
        if any([levelset(mesh.node_coords[node, :]) for node in nodes] .< 0) # at least 1 is inside
            cells_to_keep[cell] = true
            nodes_to_keep[nodes] .= true
            faces_to_keep[faces] .= true
        end
    end

    cell_faces = mesh.cell_faces[faces_to_keep, cells_to_keep] # faces x cells sparse matrix
    face_nodes = mesh.face_nodes[nodes_to_keep, faces_to_keep] # nodes x faces sparse matrix
    node_coords = mesh.node_coords[nodes_to_keep, :] # nodes x dim matrix

    new_mesh = Mesh(node_coords, cell_faces, face_nodes)

    return new_mesh
end

#-------------- New functions from 19/09/22 -----------#
function get_rows_from_cols(A::SparseMatrixCSC, cols::SparseVector; ignored_row_entries=1)
    rows = (abs.(A) * cols .!= 0)
    if ignored_row_entries != 1
        @assert(typeof(ignored_row_entries) == SparseVector{Bool,Int64})
        rows[ignored_row_entries] *= 0
        dropzeros!(rows)
    end
    return rows
end

function get_ngbr_nodes(grid, nodes; in_interior=false, ignored_faces=1)
    exterior_nodes = sparsevec(get_bdry_dofs(grid)[1])
    if in_interior
        nodes[exterior_nodes] *= 0
        dropzeros!(nodes)
    end

    if ignored_faces != 1
        @assert(typeof(ignored_faces) == SparseVector{Bool,Int64})

        conn_faces = get_rows_from_cols(sparse(grid.face_nodes'), nodes; ignored_row_entries=ignored_faces)
        if in_interior
            ignored_nodes = (exterior_nodes + nodes .!= 0)
            return get_rows_from_cols(grid.face_nodes, conn_faces; ignored_row_entries=ignored_nodes)
        else
            return get_rows_from_cols(grid.face_nodes, conn_faces; ignored_row_entries=nodes)
        end
    else
        conn_faces = get_rows_from_cols(sparse(grid.face_nodes'), nodes)
        if in_interior
            ignored_nodes = (exterior_nodes + nodes .!= 0)
            return get_rows_from_cols(grid.face_nodes, conn_faces; ignored_row_entries=ignored_nodes)
        else
            return get_rows_from_cols(grid.face_nodes, conn_faces; ignored_row_entries=nodes)
        end
    end
end

function expand_cells(grid, conn_nodes)
    conn_nodes = findnz(conn_nodes)[1]
    new_avgd_coords = grid.node_coords[conn_nodes, :]
    j = 1
    for node in conn_nodes
        faces_to_node = grid.face_nodes[node, :]
        ngbr_nodes = (abs.(grid.face_nodes) * faces_to_node .!= 0)

        ngbr_coords = grid.node_coords[ngbr_nodes, :]
        new_coord = [sum(ngbr_coords[:, 1]), sum(ngbr_coords[:, 2])] / length(ngbr_coords[:, 1])
        new_avgd_coords[j, :] = new_coord
        j = j + 1
    end
    grid.node_coords[conn_nodes, :] = new_avgd_coords
    return Mesh(grid.node_coords, grid.cell_faces, grid.face_nodes)
end

#-------------- Plot mesh using Plots --------------#

function draw_mesh(mesh)
    plt = Plots.scatter(mesh.node_coords[:, 1], mesh.node_coords[:, 2], legend=false, aspect_ratio=:equal, markersize=3)

    for k in 1:get_num_faces(mesh)
        vertices = get_rowvals(mesh.face_nodes, k)
        Plots.plot!(plt, mesh.node_coords[vertices, 1], mesh.node_coords[vertices, 2], linecolor="black")
    end
    display(plt)
    return plt
end

function draw_cells_on_mesh(plt, mesh, cells=1:get_num_cells(mesh), color::String="red")
    faces = findnz(mesh.cell_faces[:, cells])[1]
    for face in faces
        vertices = get_rowvals(mesh.face_nodes, face)
        Plots.plot!(plt, mesh.node_coords[vertices, 1], mesh.node_coords[vertices, 2], linecolor=color)
    end
    display(plt)
    return plt
end

function draw_faces_on_mesh(plt, mesh, faces, color::String="red")
    for face in faces
        vertices = get_rowvals(mesh.face_nodes, face)
        Plots.plot!(plt, mesh.node_coords[vertices, 1], mesh.node_coords[vertices, 2], linecolor=color)
    end
    display(plt)
    return plt
end

""" Warning: hard coded for the circle levelset
"""
function draw_curve_on_mesh(plt, mesh, levelset, color::String="blue")
    _, _, cut_coords, _ = mark_intersections(mesh, levelset)
    cut_coords = sortslices(cut_coords, dims=1, by=x -> atan(x[2] - 0.5, x[1] - 0.5)) # sort rows by angle

    for i = 1:size(cut_coords, 1)-1
        # vertices = get_rowvals(mesh.face_nodes, k)
        # Plots.plot!(plt, mesh.node_coords[vertices,1], mesh.node_coords[vertices,2], linecolor=color)
        Plots.plot!(plt, [cut_coords[i, 1], cut_coords[i+1, 1]], [cut_coords[i, 2], cut_coords[i+1, 2]], linecolor=color)
    end
    Plots.plot!(plt, [cut_coords[end, 1], cut_coords[1, 1]], [cut_coords[end, 2], cut_coords[1, 2]], linecolor=color)
    display(plt)
    return plt
end

#-------------- Create mesh of the hitchhiker's pentagon --------------#
function create_pentagon_mesh(nodes=[0.0 0.0; 3.0 0.0; 3.0 2.0; 3.0/2 4.0; 0.0 4.0])
    n = size(nodes, 1)
    cell_faces = sparse(1:n, ones(n), ones(n))
    face_nodes = sparse([1:n; 1:n], [1:n; circshift(1:n, 1)], [-ones(n); ones(n)])

    return Mesh(nodes, cell_faces, face_nodes)
end


end

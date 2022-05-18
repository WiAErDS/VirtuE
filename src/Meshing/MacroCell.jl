module MacroCell

using SparseArrays
import ..Meshing

struct MacroCells
    ref_size::Float64

    chain_pos::SparseVector{Int, Int} # put in cell, get bk pos
    faces_2_rm::SparseVector{Bool, Int} # which faces need to be removed?
end

#-------------- Macro element code --------------#
function init_macro(mesh::Meshing.Mesh,C::Float64,itf_faces)
    ref_size = C*maximum(mesh.cell_areas)

    cut_cells = (itf_faces' * mesh.cell_faces)'

    small_cells = find_small_cells(mesh,cut_cells,ref_size)

    if !any(small_cells)
        chain_pos = spzeros(Meshing.get_num_cells(mesh))
        faces_2_rm = spzeros(Bool, Meshing.get_num_faces(mesh))
    else
        chain_pos, faces_2_rm = build_chain(mesh,itf_faces,small_cells)
    end

    return MacroCells(ref_size,chain_pos, faces_2_rm)
end

function find_small_cells(mesh,cut_cells,ref_size)
    small_cells = spzeros(Bool, Meshing.get_num_cells(mesh))

    for cell in cut_cells.nzind
        small_cells[cell] = mesh.cell_areas[cell] < ref_size
    end

    return small_cells
end

function build_chain(mesh::Meshing.Mesh,itf_faces,small_cells)
    # while loop, each run increases chain_pos
    # for each cell
    tmp_small_cells = copy(small_cells)
    faces_2_rm = spzeros(Bool, Meshing.get_num_faces(mesh))

    chain_pos = spzeros(Int, Meshing.get_num_cells(mesh))
    pos = 1
    while any(tmp_small_cells)
        for cell in tmp_small_cells.nzind

            faces = Meshing.get_rowvals(mesh.cell_faces,cell)
            face_inds, ngbr_cells = findnz(mesh.cell_faces[faces, :])[1:2]
            ngbr_faces = faces[face_inds]

            disqualified = (ngbr_cells .== cell) .| (itf_faces[ngbr_faces]) .| (tmp_small_cells[ngbr_cells])

            if any(.!disqualified)
                fat_cells = ngbr_cells[.!disqualified]
                fat_faces = ngbr_faces[.!disqualified]

                best_ind = argmax(mesh.cell_areas[fat_cells])

                chain_pos[cell] = pos
                faces_2_rm[fat_faces[best_ind]] = true
            end
        end
        cells_2_rm = abs.(mesh.cell_faces)' * faces_2_rm
        tmp_small_cells[cells_2_rm.nzind] .= false
        dropzeros!(tmp_small_cells) # [is this necessary?]

        pos += 1
        @assert(pos < 100, "Failed to find chains")
    end
    return chain_pos, faces_2_rm
end


## Remeshing

function remesh(mesh::Meshing.Mesh, macro_cells::MacroCells)

    cell_faces, keep_cells = create_new_cell_faces(mesh, macro_cells)
    new_cell_faces = cell_faces[:, Meshing.get_num_cells(mesh) + 1 : end]

    new_cell_nodes = Meshing.compute_cell_nodes(new_cell_faces, mesh.face_nodes)

    cell_nodes = vcat(mesh.cell_nodes, new_cell_nodes)

    new_areas = [Meshing.get_poly_area(mesh.node_coords[nodes, :]) for nodes in new_cell_nodes]
    new_centroids = [Meshing.get_poly_centroid(mesh.node_coords[nodes, :]) for nodes in new_cell_nodes]
    new_diams = [Meshing.get_poly_diam(mesh.node_coords[nodes, :]) for nodes in new_cell_nodes]

    cell_areas = vcat(mesh.cell_areas, new_areas)
    cell_centroids = vcat(mesh.cell_centroids, new_centroids'...)
    cell_diams = vcat(mesh.cell_diams, new_diams)

    keep_faces = vcat(.!macro_cells.faces_2_rm)

    cell_faces = cell_faces[keep_faces, keep_cells]
    face_nodes = mesh.face_nodes[:, keep_faces]
    cell_nodes = cell_nodes[keep_cells]

    cell_areas = cell_areas[keep_cells]
    cell_centroids = cell_centroids[keep_cells, :]
    cell_diams = cell_diams[keep_cells]

    new_mesh = Meshing.Mesh(mesh.node_coords, cell_faces, face_nodes, cell_nodes, cell_areas, cell_centroids, cell_diams)

    return new_mesh
end

function create_new_cell_faces(mesh::Meshing.Mesh, macro_cells::MacroCells)
    cell_faces = copy(mesh.cell_faces)
    keep_cells = trues(Meshing.get_num_cells(mesh))

    for face in macro_cells.faces_2_rm.nzind
        old_cells = cell_faces[face, :] .!= 0

        keep_cells[old_cells] .= false
        push!(keep_cells, true)

        new_cell_faces = sign.(cell_faces * old_cells)

        cell_faces = [cell_faces new_cell_faces]
    end

    dropzeros!(cell_faces)

    return cell_faces, keep_cells
end

end
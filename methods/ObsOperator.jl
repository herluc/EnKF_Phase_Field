module ObsOperator

using ProgressMeter
using LinearSolve
using Ferrite, FerriteMeshParser
using FerriteGmsh, Gmsh
using LinearAlgebra
using SparseArrays
using JLD2
using ForwardDiff
using TimerOutputs
using IterativeSolvers
using Statistics
using Optim, NLSolversBase, LineSearches
using Random, Distributions
using KernelFunctions





struct MyPointIterator{PH<:PointEvalHandler, V <: Vec}
    ph::PH
    coords::Vector{V}
end
function MyPointIterator(ph::PointEvalHandler{G}) where {D,C,T,G<:Grid{D,C,T}}
    n = Ferrite.nnodes_per_cell(ph.grid)
    coords = zeros(Vec{D,T}, n) # resize!d later if needed
    return MyPointIterator(ph, coords)
end

function Base.iterate(p::MyPointIterator, state = 1)
    if state > length(p.ph.cells)
        return nothing
    elseif p.ph.cells[state] === nothing
        return (nothing, state + 1)
    else
        cid = (p.ph.cells[state])::Int
        local_coord = (p.ph.local_coords[state])::Vec
        n = Ferrite.nnodes_per_cell(p.ph.grid, cid)
        getcoordinates!(resize!(p.coords, n), p.ph.grid, cid)
        point = PointLocation(cid, local_coord, p.coords)
        return (point, state + 1)
    end
end



function assemble_element_obs!(Ke::Matrix, fe::Vector, dh, indicator::Int, refcoord::Vec{2, Float64}, ip, field)
    #2D
    fill!(Ke, 0)
    fill!(fe, 0)
    dofrange_u = dof_range(dh, :u)
    dofrange_d = dof_range(dh, :d)

    
    
    # Add contribution to fe
    if (field == "u1")
        for (i,I) in pairs(dofrange_u[1:2:last(dofrange_u)])
            pt_in_ref_coord = refcoord
            u_point = Ferrite.value(ip, i, pt_in_ref_coord)
            fe[I] += indicator * u_point
        end
    end

    if (field == "u2")
        for (i,I) in pairs(dofrange_u[2:2:last(dofrange_u)])
            pt_in_ref_coord = refcoord
            u_point = Ferrite.value(ip, i, pt_in_ref_coord)
            fe[I] += indicator * u_point
        end
    end


    for (i, I) in pairs(dofrange_d)
        pt_in_ref_coord = refcoord
        u_point = Ferrite.value(ip, i, pt_in_ref_coord)
        fe[I] += 0#indicator * u_point
    end

    return Ke, fe
end



function assemble_element_obs!(Ke::Matrix, fe::Vector, dh, indicator::Int, refcoord::Vec{1, Float64}, ip, field)
    ### 1D


    fill!(Ke, 0)
    fill!(fe, 0)
    dofrange_u = dof_range(dh, :u)

    if (field == "u1")
        for (i,I) in pairs(dofrange_u)
            pt_in_ref_coord = refcoord
            u_point = Ferrite.value(ip, i, pt_in_ref_coord)
            fe[I] += indicator * u_point
        end
    end
    
    return Ke, fe
end




function assemble_global_obs(cellvalues, K::SparseMatrixCSC, dh::DofHandler, cid::Int, refcoord::Vec{2, Float64}, ip, field)
    # 2D
    # Allocate the element stiffness matrix and element force vector
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    fe = zeros(n)
    # Allocate global force vector f
    f = zeros(ndofs(dh))
    # Create an assembler
    assembler = start_assemble(K, f)
    # Loop over all cells
    indicator = 0
    for cell in CellIterator(dh)
        if (cell.cellid.x == cid)
            indicator = 1
        else
            indicator = 0
        end
        cvu, cvd = cellvalues
        Ferrite.reinit!(cvu,cell)
        Ferrite.reinit!(cvd,cell)
        # Compute element contribution
        assemble_element_obs!(Ke, fe, dh, indicator, refcoord, ip, field)
        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end


function assemble_global_obs(cellvalues, K::SparseMatrixCSC, dh::DofHandler, cid::Int, refcoord::Vec{1, Float64}, ip, field)
    # 1D

    # Allocate the element stiffness matrix and element force vector
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    fe = zeros(n)
    # Allocate global force vector f
    f = zeros(ndofs(dh))
    # Create an assembler
    assembler = start_assemble(K, f)
    # Loop over all cells
    indicator = 0
    for cell in CellIterator(dh)
        if (cell.cellid.x == cid)
            indicator = 1
        else
            indicator = 0
        end

        cvu, cvd = cellvalues
        Ferrite.reinit!(cvu,cell)
        Ferrite.reinit!(cvd,cell)
        # Compute element contribution
        assemble_element_obs!(Ke, fe, dh, indicator, refcoord, ip, field)
        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end



function generateP(points,dh,grid,cellvalues,ip;dim2D=false)
    global K = create_sparsity_pattern(dh)
    ph = PointEvalHandler(grid,points)
    cid_list = Int64[]
    local_coord_list = Vector{Float64}[]
    P = Vector{Float64}[]
    
    for (i,point) in enumerate(MyPointIterator(ph))
        
        point === nothing && (print("point is not inside the domain!"); throw(error())) # Skip any points that weren't found
        #print(point.cid)
        push!(cid_list,point.cid)
        push!(local_coord_list,point.local_coord)
        global K, f = assemble_global_obs(cellvalues, K, dh, point.cid, point.local_coord, ip, "u1");
        push!(P,f)
    end
    if dim2D==true
        for (i,point) in enumerate(MyPointIterator(ph))
            point === nothing && (print("point is not inside the domain!"); throw(error())) # Skip any points that weren't found
            push!(cid_list,point.cid)
            push!(local_coord_list,point.local_coord)
            global K, f = assemble_global_obs(cellvalues, K, dh, point.cid, point.local_coord, ip, "u2");
            push!(P,f)
        end
    end
    P_m = stack(P)

    P_sparse = sparse(P_m)
    return P_sparse
end # generateP end


end #module end

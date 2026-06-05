module GlobalDofRange

using Ferrite
using LinearAlgebra
using SparseArrays
using Random, Distributions
using KernelFunctions

function global_dof_range(dh::DofHandler, field_name::Symbol)
    dofs = Set{Int}()
        if field_name ∈ Ferrite.getfieldnames(dh)
            _global_dof_range!(dofs, dh, field_name)
        end
    #end
    return sort!(collect(Int, dofs))
end
function _global_dof_range!(dofs, dh::DofHandler, field_name)
    cellsets = getcellsets(dh.grid)
    cellset = get(cellsets, "all", "error")
    println(cellset)
    println(length(cellset))
    eldofs = celldofs(dh, first(cellset))
    field_range = dof_range(dh, field_name)
    for i in cellset
        celldofs!(eldofs, dh, i)
        for j in field_range
            @inbounds d = eldofs[j]
            d in dofs || push!(dofs, d)
        end
    end
end

end # end module
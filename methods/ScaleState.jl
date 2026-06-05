module ScaleState

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
include("./GlobalDofRange.jl")
using .GlobalDofRange

function scale_it(cellvalues, dh, a, factor)
    cvu, cvd = cellvalues

    n = Ferrite.getnbasefunctions(cvu)*2
    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cvu)
    dofrange_u = dof_range(dh, :u)
    dofrange_d = dof_range(dh, :d)

    # Allocate storage for the scaled vector to store
    a_scaled = deepcopy(a)
    q = [Vec{2,Float64}[] for _ in 1:getncells(dh.grid)]

    for (cell_num, cell) in enumerate(CellIterator(dh))
        q_cell = q[cell_num]
        celldofs!(cell_dofs, dh, cell_num)
        aᵉ = a[cell_dofs]
        aᵉ_scaled = a_scaled[cell_dofs]

        for (i,I) in pairs(dofrange_u)

            aᵉ_scaled[I] = aᵉ[I]*factor
        end

        a_scaled[cell_dofs] = aᵉ_scaled

        Ferrite.reinit!(cvu,cell)
        Ferrite.reinit!(cvd,cell)

    end
    return a_scaled
end

function replace_it(cellvalues, dh, dofrange, a, new_vec)
    cvu, cvd = cellvalues
    n = Ferrite.getnbasefunctions(cvu)*2

    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cvu)
    dofrange_u = dof_range(dh, :u)
    dofrange_d = dof_range(dh, :d)

    # Allocate storage for the scaled vector to store
    a_scaled = deepcopy(a)
    q = [Vec{2,Float64}[] for _ in 1:getncells(dh.grid)]

    for (cell_num, cell) in enumerate(CellIterator(dh))
        q_cell = q[cell_num]
        cell_dofs = celldofs(dh,cell_num) #2D
     #   celldofs!(cell_dofs, dh, cell_num) #1D
        aᵉ = a[cell_dofs]
        aᵉ_scaled = a_scaled[cell_dofs]
        new_vecᵉ = new_vec[cell_dofs]
        for (i,I) in pairs(dofrange)
            aᵉ_scaled[I] = new_vecᵉ[I]
        end
        a_scaled[cell_dofs] = aᵉ_scaled

        Ferrite.reinit!(cvu,cell)
        Ferrite.reinit!(cvd,cell)

    end
    return a_scaled
end


function replace_cov(cov,dh,coords)
    cov_mat = deepcopy(cov)
    index_u = GlobalDofRange.global_dof_range(dh,:u)
    print(index_u)
    c_u = 0.00023^2*with_lengthscale(Matern32Kernel(), 0.05) 
    C_u = kernelmatrix(c_u, ColVecs(stack(coords)))
    for (i,I) in pairs(index_u)
        for (j,J) in pairs(index_u)
            cov_mat[I,J] = C_u[i,j]
        end
    end
    #quit()
    return cov_mat
end


function replace_corr(corr,dh,coords)
    corr_mat = deepcopy(corr)
    index_u_full = GlobalDofRange.global_dof_range(dh,:u)
    index_u = index_u_full[1:end]
    print(index_u)
    c_u = 0.00023^2*with_lengthscale(SqExponentialKernel(), 0.2) 
    C_u = kernelmatrix(c_u, ColVecs(stack(coords[1:end])))
    Dinv = diagm(1 ./ sqrt.(diag(C_u)))
    corr_u = Dinv * C_u * Dinv
    for (i,I) in pairs(index_u)
        for (j,J) in pairs(index_u)
            corr_mat[I,J] = corr_u[i,j]
        end
    end
    return corr_mat
end





function isolate_displacements(cellvalues, dh, a)
    cvu, cvd = cellvalues

    n = Ferrite.getnbasefunctions(cvu)*2
    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cvu)
    dofrange_u = dof_range(dh, :u)
    dofrange_d = dof_range(dh, :d)

    isolated_field = reshape_to_nodes(dh, a, :u)

    return isolated_field
end


function isolate_phasefield(cellvalues, dh, a)
    cvu, cvd = cellvalues

    n = Ferrite.getnbasefunctions(cvd)*2
    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cvu)
    dofrange_u = dof_range(dh, :u)
    dofrange_d = dof_range(dh, :d)

    isolated_field = reshape_to_nodes(dh, a, :d)

    return isolated_field
end


end # end module

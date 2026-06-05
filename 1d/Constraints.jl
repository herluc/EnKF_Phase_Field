module Constraints

using Ferrite, SparseArrays
using LinearAlgebra
using Statistics
using Random, Distributions
using KernelFunctions


struct NoSplit end
struct SpectralSplit end # Not implemented
struct VolumetricSplit end

# Convexifications
struct NoConvexification end
struct RateConvexification end

split_energy(m, ϵ) = split_energy(m.energy_split, m, ϵ)
function split_energy(::Any, m, ϵ)
    λ = m.K - 2 * m.G / 3
    Ψ⁺ = 0.5 * λ * tr(ϵ)^2 + m.G * (ϵ ⊡ ϵ)
    Ψ⁻ = zero(Ψ⁺)
    return Ψ⁺, Ψ⁻
end

function doassemble!(K, r, material, cv, dh, a, a_old, Δt, states, states_old)
    assembler = start_assemble(K, r)
    n = ndofs_per_cell(dh)
    Ke = zeros(n,n)
    re = zeros(n)
    ae_old = zeros(n)
    ae = zeros(n)
    for (i, cell) in enumerate(CellIterator(dh))
        # copy values from a to ae
        map!(i->a[i], ae, celldofs(cell))
        map!(i->a_old[i], ae_old, celldofs(cell))
        fill!(Ke, 0)
        fill!(re, 0)
        state = @view states[:, i]
        state_old = @view states_old[:, i]
        #comp_jacobian!(Ke, re, material, cv, cell, ae, ae_old, Δt, dh, state, state_old)
        element_routine!(Ke, re, material, cv, cell, ae, ae_old, Δt, dh, state, state_old)
        assemble!(assembler, celldofs(cell), Ke, re)
    end
    return K
end
    

function element_routine!(Ke, re, m, cvs, cell, ae::AbstractVector, ae_old, Δt, dh, state, state_old)
    # Setup cellvalues and give easier names
    #reinit!.(cvs, (cell,))

    material = m
    fill!(re, 0)
    fill!(Ke, 0)
    cvu, cvd = cvs
    Ferrite.reinit!(cvu,cell)
    Ferrite.reinit!(cvd,cell)
    dofrange_u = dof_range(dh, :u)
    dofrange_d = dof_range(dh, :d)

    Gcl_cw2 = 2*material.Gc*material.l/material.cw

    for q_point in 1:getnquadpoints(cvu)
        dΩ = getdetJdV(cvu, q_point)
        # For each integration point, compute stress and material stiffness
        ϵ = function_symmetric_gradient(cvu, q_point, ae, dofrange_u) # Total strain
        d = function_value(cvd, q_point, ae, dofrange_d)
        ∇d = function_gradient(cvd, q_point, ae, dofrange_d)
        old_qp_state = state_old[q_point]

        ⁿϕ = old_qp_state.ϕ
        Ψ⁺, Ψ⁻ = split_energy(material, ϵ)
        ϕ = calculate_phasefield(material, Ψ⁺, Ψ⁻, d)
        
        dold = function_value(cvd, q_point, ae_old, dofrange_d)
        dhat = min(max(dold, dold + Δt*old_qp_state.d_rate), zero(dold))
        ϕhat = calculate_phasefield(material, Ψ⁺, Ψ⁻, dhat)
        gϕ = degradation_function(material, ϕhat)
        # Update state variables
        #d_rate = (ForwardDiff.value(d)-dold)/Δt
 
       # Make sure gϕ!=0 to avoid singular stiffness
        
       gϕ_fix = (1-material.g_resid)*gϕ + material.g_resid
       σ = calculate_degraded_stress(material, ϵ, gϕ_fix)

        for (i, I) in pairs(dofrange_u)
            δϵ = shape_symmetric_gradient(cvu, q_point, i)
            δd = shape_value(cvd, q_point, i)
            gϕ = degradation_function(material, δd)
            gϕ_fix = (1-material.g_resid)*gϕ + material.g_resid
          #  σ = calculate_degraded_stress(material,δϵ,gϕ_fix)
            #σ = calculate_degraded_stress(material,δϵ,1.0)

            for (j, J) in pairs(dofrange_u)
                ∇δu = shape_gradient(cvu, q_point, j)
                Ke[I,J] +=  -(σ⊡∇δu) * dΩ
            end

        end



        # for (i, I) in pairs(dofrange_u)
        #     ∇δNu = shape_symmetric_gradient(cvu, q_point, i)
        #     re[I] += (∇δNu ⊡ σ) * dΩ 
        # end
        # for (i, I) in pairs(dofrange_d)
        #     ∇δNd = shape_gradient(cvd, q_point, i)
        #     δNd = shape_value(cvd, q_point, i)
        #     re[I] += (Gcl_cw2* (∇δNd ⋅ ∇d) - material.α*(ϕ-d)*δNd)*dΩ
        # end
       # state[q_point] = MicroMorphicElasticPhaseFieldState(ForwardDiff.value(ϕ), d_rate)
    end
end


function comp_jacobian!(Ke, re, material, cv, cell, ae, ae_old, Δt, dh, state, state_old)
    rf!(re_, ae_) = element_routine!(re_, material, cv, cell, ae_, ae_old, Δt, dh, state, state_old)
    cfg = ForwardDiff.JacobianConfig(rf!, re, ae, ForwardDiff.Chunk{length(ae)}())
    ForwardDiff.jacobian!(Ke, rf!, re, ae, cfg)
end

calculate_phasefield(m, args...) = calculate_phasefield(m.fracture, m, args...)
function calculate_phasefield(::Any, m, Ψ⁺, Ψ⁻, d::T) where {T} # after Kalman shift, w/o ⁿϕ
    #return min((2 * Ψ⁺ + m.α * d) / (2 * Ψ⁺ + m.α + m.Gc / m.l), one(T))
    return (2 * Ψ⁺) / (2 * Ψ⁺ + m.Gc / m.l)
end

function calculate_degraded_stress(m, ϵ, gϕ)
    return (gϕ * 2 * m.G) * dev(ϵ) + (3 * gϕ * m.K) * vol(ϵ)
end


degradation_function(m, ϕ) = degradation_function(m.fracture, m, ϕ)
degradation_function(::Any, ::Any, ϕ) = (1 - ϕ)^2

end # end module

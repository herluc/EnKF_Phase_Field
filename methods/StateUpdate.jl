module StateUpdate

using LinearSolve
using Ferrite
using LinearAlgebra
using SparseArrays
using ForwardDiff
using IterativeSolvers

function doassemble_states!(material, cv, dh, a_updated, a_old, Δt, states, states_old)
    a = a_updated
    n = ndofs_per_cell(dh)
    ae_old = zeros(n)
    ae = zeros(n)
    for (i, cell) in enumerate(CellIterator(dh))
        map!(i->a[i], ae, celldofs(cell))
        map!(i->a_old[i], ae_old, celldofs(cell))
        state = @view states[:, i]
        state_old = @view states_old[:, i]

        element_routine_states!(re, material, cv, cell, ae, ae_old, Δt, dh, state, state_old)

    end
end

function element_routine_states!(material::MicroMorphicElasticPhaseField, cvs, cell, ae::AbstractVector, ae_old, Δt, dh, state, state_old)
    # Setup cellvalues and give easier names

    cvu, cvd = cvs
    Ferrite.reinit!(cvu,cell)
    Ferrite.reinit!(cvd,cell)
    dofrange_u = dof_range(dh, :u)
    dofrange_d = dof_range(dh, :d)

    Gcl_cw2 = 2*material.Gc*material.l/material.cw
    for q_point in 1:getnquadpoints(cvu)
        # For each integration point, compute stress and material stiffness
        ϵ = function_symmetric_gradient(cvu, q_point, ae, dofrange_u) # Total strain
        d = function_value(cvd, q_point, ae, dofrange_d)
        old_qp_state = state_old[q_point]

        ⁿϕ = old_qp_state.ϕ
        Ψ⁺, Ψ⁻ = split_energy(material, ϵ)
        ϕ = calculate_phasefield(material, Ψ⁺, Ψ⁻, d, ⁿϕ)

        # Branching easier than dispatch due to many different arguments
        if isa(material.convexification, NoConvexification)
            gϕ = degradation_function(material, ϕ)
            d_rate = 0.0
        elseif isa(material.convexification, RateConvexification)
            dold = function_value(cvd, q_point, ae_old, dofrange_d)
            dhat = min(max(dold, dold + Δt*old_qp_state.d_rate), zero(dold))
            ϕhat = calculate_phasefield(material, Ψ⁺, Ψ⁻, dhat, ⁿϕ)
            gϕ = degradation_function(material, ϕhat)
            # Update state variables
            d_rate = (ForwardDiff.value(d)-dold)/Δt
        else
            error("Not supported")
        end
        # Make sure gϕ!=0 to avoid singular stiffness
        gϕ_fix = (1-material.g_resid)*gϕ + material.g_resid

        σ = calculate_degraded_stress(material, ϵ, gϕ_fix)

        state[q_point] = MicroMorphicElasticPhaseFieldState(ForwardDiff.value(ϕ), d_rate)

    end
 

end

end # end module
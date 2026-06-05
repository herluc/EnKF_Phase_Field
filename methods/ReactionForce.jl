# Calculation of the reaction force at the displaced boundary
module ReactionForce

using Ferrite

struct ReactionForcePars{}
    rf_dofs::Vector{Int}
    rf_dofs_2::Vector{Int}
    rf_vals::Matrix{Float64}
end

function ReactionForcePars(time_vector, n_samps, dh::Ferrite.AbstractDofHandler; faceset="right", dim=1)
    ch = ConstraintHandler(dh)
    ch2 = ConstraintHandler(dh)
    if dim == 1
        add!(ch, Ferrite.Dirichlet(:u, getfaceset(dh.grid , faceset), Returns(0.0), [1]))
        close!(ch)
        return ReactionForcePars(Ferrite.prescribed_dofs(ch), Vector{Int}(), zeros(length(time_vector),n_samps))
    elseif dim == 2
        add!(ch, Ferrite.Dirichlet(:u, getfaceset(dh.grid , faceset), Returns(0.0), [1]))
        add!(ch2, Ferrite.Dirichlet(:u, getfaceset(dh.grid , faceset), Returns(0.0), [2]))
       # add!(ch, Ferrite.Dirichlet(:u, getfaceset(dh.grid , faceset), Returns(zero(Vec{2})), [1,2]))
        close!(ch)
        close!(ch2)
        return ReactionForcePars(Ferrite.prescribed_dofs(ch), Ferrite.prescribed_dofs(ch2), zeros(length(time_vector),n_samps))
    else
        println("only dimensions 1 and 2 are implemented")
    end
    
    
    
end

function calculate_reaction_force(rf_dofs::Vector{Int}, assembly_routine, K, r, material, cv, dh, a, a_old, Δt, states, states_old)
    r_re = copy(r)
    K_re = copy(K)
    assembly_routine(K_re, r_re, material, cv, dh, a, a_old, Δt, states, states_old)
    return sum(r_re[i] for i in rf_dofs)
end

function postprocess(sample_idx, timestep_idx, assembly_routine, pp::ReactionForcePars, K, r, material, cv, dh, a, a_old, Δt, states, states_old; dim=1)
    
    if dim == 1
        rf = calculate_reaction_force(pp.rf_dofs, assembly_routine, K, r, material, cv, dh, a, a_old, Δt, states, states_old)
        rf_total = sqrt(rf^2)
    elseif dim == 2
        rf = calculate_reaction_force(pp.rf_dofs, assembly_routine, K, r, material, cv, dh, a, a_old, Δt, states, states_old)
        rf2 = calculate_reaction_force(pp.rf_dofs_2, assembly_routine, K, r, material, cv, dh, a, a_old, Δt, states, states_old)
        rf_total = sqrt(rf^2 + rf2^2)
    end
    pp.rf_vals[timestep_idx,sample_idx] = rf_total
  #  push!(pp.rf_vals, rf)
end



end # module

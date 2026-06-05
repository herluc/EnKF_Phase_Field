# compute phase field and displacements separately
# used for regularization scheme

module SeparateComputation

using Ferrite, SparseArrays
using LinearAlgebra
using ForwardDiff

using ..TypesPhasefield
using ..FuncsPhasefield
using ..ScaleState


function solve_phasefield!(material, cv, dh, ch, a, aA, a_old, Δt, states, states_old,l_factor,max_val_pp)
    n_a = length(a)
    σ_e = 1.0


    Kd = create_sparsity_pattern(dh);
    fd = zeros(ndofs(dh))
    assembler = start_assemble(Kd, fd)
    cvu, cvd = cv
    n = ndofs_per_cell(dh)
    Ke = zeros(n,n)
    fe = zeros(n)

    # Perform Newton iterations
    newton_itr = -1
    NEWTON_TOL = 1e-6
    NEWTON_MAXITER = 20
    if l_factor > 1.0
        NEWTON_MAXITER = 1
    else
        NEWTON_MAXITER = 1
    end
        
    Δa = zeros(ndofs(dh))
    ni = 0
    normr_0 = 0
    a_old = deepcopy(a)
    a_A = deepcopy(aA)
    isolated_u = ScaleState.isolate_displacements(cv,dh,a)
    while true; newton_itr += 1          
        Δa .= ScaleState.replace_it(cv,dh,dof_range(dh, :u),Δa,repeat(isolated_u.*0.0,3))

        a .= a_old .+ Δa
        
        assembler = start_assemble(Kd, fd)
        cvu, cvd = cv
        n = ndofs_per_cell(dh)
        Ke = zeros(n,n)
        fe = zeros(n)
        ae_old = zeros(n)
        ae = zeros(n)
        for (i, cell) in enumerate(CellIterator(dh))
            # copy values from a to ae
            map!(i->a[i], ae, celldofs(cell))
            map!(i->a_old[i], ae_old, celldofs(cell))
            fill!(Ke, 0.0)
            fill!(fe, 0.0)
            state = @view states[:, i]
            state_old = @view states_old[:, i]
            comp_jacobian_phasefield!(Ke, fe, material, cv, cell, ae, ae_old, Δt, dh, state, state_old, l_factor, max_val_pp)
            element_routine_phasefield!(Ke,fe, material, cv, cell, ae, ae_old, Δt, dh, state, state_old, l_factor, max_val_pp)
            assemble!(assembler, celldofs(cell), Ke, fe)
        end
        
        normr = norm(fd[Ferrite.free_dofs(ch)])
        if newton_itr == 0
            normr_0 = normr
        end
        norm_rel = normr/normr_0
        
        if 1.0==1.0#l_factor == 1.0
            r = a .- a_A  #proximal residual
            r_d = zeros(ndofs(dh))
            r .= ScaleState.replace_it(cv,dh,dof_range(dh, :u),r,repeat(r.*0.0,3))
            λ = 3e-7#1e-9#1e-5
            grad_J_prox = λ * (I\r)
            hess_J_prox = λ * (I\I)
            lhs = Kd'*Kd + hess_J_prox 
            rhs = Kd'*fd + grad_J_prox
            rhs_fe = Kd'*fd
            println("itr.",newton_itr, ", norm rel/abs phase: ",norm_rel,";",normr," Kr=",norm(rhs_fe[Ferrite.free_dofs(ch)])," grad Penalty=",norm(grad_J_prox[Ferrite.free_dofs(ch)]))
        else
            lhs = Kd'*Kd
            rhs = Kd'*fd
            println("itr.",newton_itr, ", norm rel/abs phase: ",norm_rel,";",normr," Kr=",norm(rhs[Ferrite.free_dofs(ch)]))
        end

        
        if norm_rel < 1e-3
            break
        elseif newton_itr > NEWTON_MAXITER
            break
            ni += 1
            Δa .*=  0.9993
            if ni > 3
                error("Reached maximum Newton iterations, aborting")
            end
        end


        Δa .-= lhs \ rhs

    end

    ad = copy(a)
    return ad
end

function element_routine_phasefield!(Ke, fe::AbstractVector, m::MicroMorphicElasticPhaseField, cvs, cell, ae::AbstractVector, ae_old, Δt, dh, state, state_old, l_factor, max_val_pp)
    # Setup cellvalues and give easier names
    material = m
    fill!(fe, 0.0)
    cvu, cvd = cvs
    n_basefuncs_d = getnbasefunctions(cvd)
    Ferrite.reinit!(cvu,cell)
    Ferrite.reinit!(cvd,cell)
    dofrange_u = dof_range(dh, :u)
    dofrange_d = dof_range(dh, :d)

    Gcl_cw2 = 2*material.Gc*material.l*l_factor/material.cw
    for q_point in 1:getnquadpoints(cvu)
        dΩ = getdetJdV(cvu, q_point)
        # For each integration point, compute stress and material stiffness
        ϵ = function_symmetric_gradient(cvu, q_point, ae, dofrange_u) # Total strain
        d = function_value(cvd, q_point, ae, dofrange_d)
        #  u = function_value(cvd, q_point, ae, dofrange_u)
        ∇d = function_gradient(cvd, q_point, ae, dofrange_d)
        old_qp_state = state_old[q_point]

        ⁿϕ = old_qp_state.ϕ
        Ψ⁺, Ψ⁻ = split_energy(NoSplit(),material, ϵ)

        ϕ = calculate_phasefield_l_fac(material, Ψ⁺, Ψ⁻, d, l_factor, max_val_pp)
        # Branching easier than dispatch due to many different arguments
        if isa(material.convexification, NoConvexification)
            gϕ = degradation_function(material, ϕ)
            d_rate = 0.0
        elseif isa(material.convexification, RateConvexification)
            d_rate = 0.0#(ForwardDiff.value(d)-dold)/Δt
        else
            error("Not supported")
        end

        for (i, I) in pairs(dofrange_u)
            fe[I] = 0.0#u  #if residual based solve, choose 0.0. otherwise u
            Ke[I,I] += 1.0
        end



        ## residual
        Gc_cwl = material.G/(material.l*material.cw)
      #  Ψ⁺, Ψ⁻ = split_energy(material, ϵ)
        for (i, I) in pairs(dofrange_u)
            ∇δNu = shape_symmetric_gradient(cvu, q_point, i)
            fe[I] = 0.0#(∇δNu ⊡ σ) * dΩ 
        end
        for (i, I) in pairs(dofrange_d)
            ∇δNd = shape_gradient(cvd, q_point, i)
            δNd = shape_value(cvd, q_point, i)
            fe[I] += (Gcl_cw2* (∇δNd ⋅ ∇d) - material.α*(ϕ-d)*δNd)*dΩ
        end
        ###########



        state[q_point] = MicroMorphicElasticPhaseFieldState(ForwardDiff.value(ϕ), d_rate)
    end
end


function comp_jacobian_phasefield!(Ke, re, material, cv, cell, ae, ae_old, Δt, dh, state, state_old, l_factor, max_val_pp)
    rf!(re_, ae_) = element_routine_phasefield!(Ke, re_, material, cv, cell, ae_, ae_old, Δt, dh, state, state_old, l_factor, max_val_pp)
    cfg = ForwardDiff.JacobianConfig(rf!, re, ae, ForwardDiff.Chunk{length(ae)}())
    ForwardDiff.jacobian!(Ke, rf!, re, ae, cfg)
end



calculate_phasefield_l_fac(m::MicroMorphicElasticPhaseField, args...) = calculate_phasefield_l_fac(m.fracture, m, args...)
function calculate_phasefield_l_fac(::AT1_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T, l_factor=1, max_val_pp=1) where {T} # after Kalman shift, w/o ⁿϕ
    return min(max((2 * Ψ⁺ + m.α * d - 3 * m.Gc / (8 * m.l)) / (2 * Ψ⁺ + m.α),zero(T)),one(T).*max_val_pp)
end
function calculate_phasefield_l_fac(::AT2_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T, l_factor=1, max_val_pp=1) where {T} # after Kalman shift, w/o ⁿϕ
    return min((2 * Ψ⁺ + m.α * d) / (2 * Ψ⁺ + m.α + m.Gc / (m.l*l_factor)),one(T).*max_val_pp)
end
function calculate_phasefieldl_fac(::AT2_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T, ⁿϕ, l_factor=1, max_val_pp=1) where {T}
    return min(max((2 * Ψ⁺ + m.α * d) / (2 * Ψ⁺ + m.α + m.Gc /  (m.l*l_factor)), convert(T, ⁿϕ)), one(T).*max_val_pp)
end



function solve_displacement!(material, cv, dh, ch, a, aA, a_old, Δt, states, states_old)
    n_a = length(a)
    σ_e = 1#e-3


    Kd = create_sparsity_pattern(dh);
    fd = zeros(ndofs(dh))
    assembler = start_assemble(Kd, fd)
    cvu, cvd = cv
    n = ndofs_per_cell(dh)
    Ke = zeros(n,n)
    fe = zeros(n)

    # Perform Newton iterations
    newton_itr = -1
    NEWTON_TOL = 1e-6
    NEWTON_MAXITER = 20
    Δa = zeros(ndofs(dh))
    ni = 0
    normr_0 = 0
    a_old = deepcopy(a)
    a_A = deepcopy(aA)
   # apply!(a, ch)
   # apply!(a_old, ch)
    isolated_u = ScaleState.isolate_displacements(cv,dh,a)
    while true; newton_itr += 1          
        Δa .= ScaleState.replace_it(cv,dh,dof_range(dh, :d),Δa,a.*0.0)

        a .= a_old .+ Δa
        
        assembler = start_assemble(Kd, fd)
        cvu, cvd = cv
        n = ndofs_per_cell(dh)
        Ke = zeros(n,n)
        fe = zeros(n)
        ae_old = zeros(n)
        ae = zeros(n)
        for (i, cell) in enumerate(CellIterator(dh))
            # copy values from a to ae
            map!(i->a[i], ae, celldofs(cell))
            map!(i->a_old[i], ae_old, celldofs(cell))
            fill!(Ke, 0.0)
            fill!(fe, 0.0)
            state = @view states[:, i]
            state_old = @view states_old[:, i]
            comp_jacobian_displacement!(Ke, fe, material, cv, cell, ae, ae_old, Δt, dh, state, state_old)
            element_routine_displacement!(Ke,fe, material, cv, cell, ae, ae_old, Δt, dh, state, state_old)
            assemble!(assembler, celldofs(cell), Ke, fe)
        end

        apply_zero!(Kd, fd, ch)
        normr = norm(fd[Ferrite.free_dofs(ch)])
        if newton_itr == 0
            normr_0 = normr
        end
        norm_rel = normr/normr_0
        println("itr.",newton_itr, ", norm rel disp: ",norm_rel)
        if norm_rel < 1e-3
            break
        elseif newton_itr > NEWTON_MAXITER
            break
            ni += 1
            Δa .*=  0.9993
            if ni > 3
                error("Reached maximum Newton iterations, aborting")
            end
        end

        r = a - a_A  #proximal residual
        λ = 3e-7#1e-9#1e-5
        grad_J_prox = λ * (I\r)
        hess_J_prox = λ * (I\I)
        lhs = Kd'*Kd + hess_J_prox# +1e-8*I
        rhs = Kd'*fd + grad_J_prox



        Δa .-= lhs \ rhs

    end

    ad = copy(a)
    return ad
end

function element_routine_displacement!(Ke, fe::AbstractVector, m::MicroMorphicElasticPhaseField, cvs, cell, ae::AbstractVector, ae_old, Δt, dh, state, state_old)
    # Setup cellvalues and give easier names
    material = m
   # fill!(Ke, 0.0)
    fill!(fe, 0.0)
    cvu, cvd = cvs
    n_basefuncs_d = getnbasefunctions(cvd)
    Ferrite.reinit!(cvu,cell)
    Ferrite.reinit!(cvd,cell)
    dofrange_u = dof_range(dh, :u)
    dofrange_d = dof_range(dh, :d)

    Gcl_cw2 = 2*material.Gc*material.l*1/material.cw
    for (i_q, q_point) in enumerate(1:getnquadpoints(cvu))
        dΩ = getdetJdV(cvu, q_point)
        # For each integration point, compute stress and material stiffness
        ϵ = function_symmetric_gradient(cvu, q_point, ae, dofrange_u) # Total strain
        d = function_value(cvd, q_point, ae, dofrange_d)
        old_qp_state = state_old[q_point]
        ⁿϕ = old_qp_state.ϕ
        Ψ⁺, Ψ⁻ = split_energy(material, ϵ)

        current_state = state[q_point]
        ϕ = current_state.ϕ

        gϕ = degradation_function(material, ϕ) # test with phi which should be the way to go

        gϕ_fix = (1-material.g_resid)*gϕ + material.g_resid

        σ = calculate_degraded_stress(material, ϵ, gϕ_fix)

        for (i, I) in pairs(dofrange_d)
            fe[I] = 0.0#u  #if residual based solve, choose 0.0. otherwise u
            Ke[I,I] += 1.0
        end

        ## residual
        Gc_cwl = material.G/(material.l*material.cw)
        for (i, I) in pairs(dofrange_u)
            ∇δNu = shape_symmetric_gradient(cvu, q_point, i)
            fe[I] += (∇δNu ⊡ σ) * dΩ 

        end
        for (i, I) in pairs(dofrange_d)
            ∇δNd = shape_gradient(cvd, q_point, i)
            δNd = shape_value(cvd, q_point, i)
            fe[I] = 0.0
        end
        ###########

    end
end


function comp_jacobian_displacement!(Ke, re, material, cv, cell, ae, ae_old, Δt, dh, state, state_old)
    rf!(re_, ae_) = element_routine_displacement!(Ke, re_, material, cv, cell, ae_, ae_old, Δt, dh, state, state_old)
    cfg = ForwardDiff.JacobianConfig(rf!, re, ae, ForwardDiff.Chunk{length(ae)}())
    ForwardDiff.jacobian!(Ke, rf!, re, ae, cfg)
end

calculate_stress(m::MicroMorphicElasticPhaseField, args...) = calculate_stress(m.energy_split, m, args...)
function calculate_stress(::NoSplit, m::MicroMorphicElasticPhaseField, ϵ, gϕ)
    E=210e3
    ν=0.3
    G = E / (2 * (1 + ν))
    K = E / (3 * (1 - 2ν))
    return (2 * G) * dev(ϵ) + (3 * K) * vol(ϵ)
end




end # module

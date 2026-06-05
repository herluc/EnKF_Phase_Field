using Revise
using ProgressMeter
using LinearSolve
using Ferrite, FerriteMeshParser
using FerriteGmsh, Gmsh
using LinearAlgebra
using SparseArrays
using JLD2
using DelimitedFiles
using DataFrames
using CSV, Tables
using ForwardDiff
using TimerOutputs
using IterativeSolvers
using Statistics
using Optim, NLSolversBase, LineSearches
using Random, Distributions
using KernelFunctions
using CovarianceEstimation
includet("../methods/TypesPhasefield.jl")
using .TypesPhasefield
includet("../methods/FuncsPhasefield.jl")
using .FuncsPhasefield
includet("../methods/ScaleState.jl")
using .ScaleState
includet("../methods/ReactionForce.jl")
using .ReactionForce
includet("../methods/SeparateComputation.jl")
using .SeparateComputation
includet("../methods/SeparateComputationData.jl")
using .SeparateComputationData
includet("../methods/ObsOperator.jl")
using .ObsOperator
includet("../methods/GenerateData.jl")
using .GenerateData
includet("./Constraints.jl")
using .Constraints
includet("../methods/KalmanFilter.jl")
using .KalmanFilter
using Plots



calculate_phasefield(m::MicroMorphicElasticPhaseField, args...) = calculate_phasefield(m.fracture, m, args...)
function calculate_phasefield(::AT1_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T, ⁿϕ) where {T}
    return min(max((2 * Ψ⁺ + m.α * d - 3 * m.Gc / (8 * m.l)) / (2 * Ψ⁺ + m.α), convert(T, ⁿϕ)), one(T))
end
function calculate_phasefield(::AT2_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T, ⁿϕ) where {T}
    return min(max((2 * Ψ⁺ + m.α * d) / (2 * Ψ⁺ + m.α + m.Gc / m.l), convert(T, ⁿϕ)), one(T))
end
function calculate_phasefield(::AT1_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T) where {T} # after Kalman shift, w/o ⁿϕ
    return min(max((2 * Ψ⁺ + m.α * d - 3 * m.Gc / (8 * m.l)) / (2 * Ψ⁺ + m.α),zero(T)),one(T))
end
function calculate_phasefield(::AT2_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T) where {T} # after Kalman shift, w/o ⁿϕ
    return (2 * Ψ⁺ + m.α * d) / (2 * Ψ⁺ + m.α + m.Gc / (m.l))
end
calculate_phasefield_0(m::MicroMorphicElasticPhaseField, args...) = calculate_phasefield_0(m.fracture, m, args...)
function calculate_phasefield_0(::AT2_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻) # after Kalman shift, w/o ⁿϕ
    return (2 * Ψ⁺) / (2 * Ψ⁺ + m.α + m.Gc / (m.l*1))
end
function calculate_phasefield_0(::AT1_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻) # after Kalman shift, w/o ⁿϕ
    return (2 * Ψ⁺ - 3 * m.Gc / (8 * m.l)) / (2 * Ψ⁺ + m.α)
end
calculate_phasefield_d(m::MicroMorphicElasticPhaseField, args...) = calculate_phasefield_d(m.fracture, m, args...)
function calculate_phasefield_d(::AT2_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T) where {T} # after Kalman shift, w/o ⁿϕ
    return (m.α * d) / (2 * Ψ⁺ + m.α + m.Gc / (m.l*1))
end
function calculate_phasefield_d(::AT1_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T) where {T} # after Kalman shift, w/o ⁿϕ
    return (m.α * d) / (2 * Ψ⁺ + m.α)
end


grid = generate_grid(Ferrite.Line, (200,));
coords = compute_vertex_values(grid, x -> x)
addcellset!(grid, "all", x -> x==x,all=false)


grid_data = generate_grid(Ferrite.Line, (220,));
coords_data = compute_vertex_values(grid_data, x -> x)
addcellset!(grid_data, "all", x -> x==x,all=false)



ip_tet = Lagrange{1,RefCube,1}()
ip_quad = Lagrange{1,RefCube,1}()

# Interpolations
ipu = Lagrange{1,RefCube,1}() # linear
ipd = Lagrange{1,RefCube,1}() # linear
ip_geo = Lagrange{1, RefCube, 1}() # interpolation for the geometry

dh = DofHandler(grid)
dh_data = DofHandler(grid_data)
add!(dh, :u, 1, ipu)
add!(dh, :d, 1, ipd)
add!(dh_data, :u, 1, ipu)
add!(dh_data, :d, 1, ipd)
close!(dh)
close!(dh_data)
renumber!(dh, DofOrder.FieldWise())
renumber!(dh_data, DofOrder.FieldWise())



dof_coords = zeros(ndofs(dh),1)

for cc in CellIterator(dh)
    cell_dofs = cc.dofs
    cell_coords = cc.coords
    for (i,dof_idx) in enumerate(cell_dofs)
        n_dof = size(cell_coords)[1]
        mapping = [1, 1, 2, 2, 3, 3, 1, 2, 3]
        coord_idx = mapping[i]#(i-1)%n_dof+1 
        coord = cell_coords[coord_idx]
        dof_coords[dof_idx,1] = coord[1]

    end
end

dof_coords = Vector{eltype(dof_coords)}[eachrow(dof_coords)...]




struct FEDomain{M,CV}
    material::M
    cellvalues::CV
end;


material = MicroMorphicElasticPhaseField(;
    l=2.5e-2,
    convexification=RateConvexification(),
    fracture=AT2_FM(),
    energy_split=NoSplit()
)


qr = Ferrite.QuadratureRule{1,RefCube}(1)
cvu = CellVectorValues(qr, ipu, ip_geo)
cvd = CellScalarValues(qr, ipd, ip_geo)
cvs = (cvu,cvd)


domains = [FEDomain(material, cvs)
            ]

ch = ConstraintHandler(dh)
load_function(x, t) = Vec((1e-4 * t))

add!(ch, Ferrite.Dirichlet(:u, getfaceset(grid, "left"), x -> [0.0], [1]))
add!(ch, Ferrite.Dirichlet(:u, getfaceset(grid, "right"), load_function))
close!(ch)
t = 0.0
Ferrite.update!(ch, t)


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
        comp_jacobian!(Ke, re, material, cv, cell, ae, ae_old, Δt, dh, state, state_old)
        element_routine!(re, material, cv, cell, ae, ae_old, Δt, dh, state, state_old)
        assemble!(assembler, celldofs(cell), Ke, re)
    end
end
    

function element_routine!(re::AbstractVector, m::MicroMorphicElasticPhaseField, cvs, cell, ae::AbstractVector, ae_old, Δt, dh, state, state_old)
    # Setup cellvalues and give easier names

    material = m
    fill!(re, 0)
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

        for (i, I) in pairs(dofrange_u)
            ∇δNu = shape_symmetric_gradient(cvu, q_point, i)
            re[I] += (∇δNu ⊡ σ) * dΩ 
        end
        for (i, I) in pairs(dofrange_d)
            ∇δNd = shape_gradient(cvd, q_point, i)
            δNd = shape_value(cvd, q_point, i)
            re[I] += (Gcl_cw2* (∇δNd ⋅ ∇d) - material.α*(ϕ-d)*δNd)*dΩ
        end
        state[q_point] = MicroMorphicElasticPhaseFieldState(ForwardDiff.value(ϕ), d_rate)
    end
end

function comp_jacobian!(Ke, re, material, cv, cell, ae, ae_old, Δt, dh, state, state_old)
    rf!(re_, ae_) = element_routine!(re_, material, cv, cell, ae_, ae_old, Δt, dh, state, state_old)
    cfg = ForwardDiff.JacobianConfig(rf!, re, ae, ForwardDiff.Chunk{length(ae)}())
    ForwardDiff.jacobian!(Ke, rf!, re, ae, cfg)
end


function doassemble_states!(material, cv, dh, a_updated, a_old, Δt, states, states_old, first)
    a = a_updated
    a_new = a_updated
    n = ndofs_per_cell(dh)
    ae_old = zeros(n)
    ae = zeros(n)
    for (i, cell) in enumerate(CellIterator(dh))
        # copy values from a to ae
        map!(i->a[i], ae, celldofs(cell))
        map!(i->a_old[i], ae_old, celldofs(cell))
        state = @view states[:, i]
        state_old = @view states_old[:, i]
        ae_new = element_routine_states!(material, cv, cell, ae, ae_old, Δt, dh, state, state_old, first)
        a_new[celldofs(cell)] = ae_new
    end
    return a_new
end

function element_routine_states!(m::MicroMorphicElasticPhaseField, cvs, cell, ae::AbstractVector, ae_old, Δt, dh, state, state_old,first)
    material = m
    cvu, cvd = cvs
    Ferrite.reinit!(cvu,cell)
    Ferrite.reinit!(cvd,cell)
    dofrange_u = dof_range(dh, :u)
    dofrange_d = dof_range(dh, :d)
    ae_new = deepcopy(ae)

    Gcl_cw2 = 2*material.Gc*material.l/material.cw
    for q_point in 1:getnquadpoints(cvu)
        ϵ = function_symmetric_gradient(cvu, q_point, ae, dofrange_u) # Total strain
        d = function_value(cvd, q_point, ae, dofrange_d)
        old_qp_state = state_old[q_point]

        ⁿϕ = old_qp_state.ϕ
        Ψ⁺, Ψ⁻ = split_energy(material, ϵ)
        if first == 1
            ϕ = d
        else
            ϕ = calculate_phasefield(material, Ψ⁺, Ψ⁻, d)
        end
        
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
        for (i,I) in pairs(dofrange_d)
            ae_new[I] = ϕ
        end
    end
    return ae_new
 
end


pp_mag = 0.7
d_peak = pp_mag    
r_cut = 0.5
d_cut = d_peak/100 # d when r=r_cut
x0 = Vec((0.57))

f(x::Vec) = f(norm(x-x0))
k = (1/d_cut-1/d_peak)/r_cut
f(r::Number) = r<r_cut ? (pp_mag*exp(-r*r/(2*0.05^2))) : 0.0

function solve(dh, ch, material, cv)
    # Pre-allocate solution vectors, etc.
    pvd = paraview_collection("1d/results/phase_field_results.pvd");
    pvd_mean = paraview_collection("1d/results/phase_field_kalman_mean.pvd");
    pvd_cov = paraview_collection("1d/results/phase_field_kalman_cov.pvd");
    pvd_list = []
    for (i,samp) in enumerate(y_samps)
        name = "1d/results/phasefield_kalman_sample"*string(i)*".pvd"
        push!(pvd_list,paraview_collection(name))
    end
    K = create_sparsity_pattern(dh); # tangent stiffness matrix
    r = zeros(ndofs(dh))
    rhs = zeros(ndofs(dh))
    a = zeros(ndofs(dh),n_samps)
    b = zeros(ndofs(dh))
    a_old = deepcopy(a)

    Δa = zeros(ndofs(dh))
    ΔΔa = zeros(ndofs(dh))

    a_u = zeros(round(Int64,ndofs(dh)/2),n_samps)
    


    nqp = 0
    for d in domains
        cvu,cvd = d.cellvalues
        nqp += getnquadpoints(cvd)
    end
    states_old_samps = []
    states_samps = []
    StateType   = typeof(MicroMorphicElasticPhaseFieldState(0.0,0.0))
    states      = Matrix{StateType}(undef, nqp, getncells(grid)) 
    states_old  = Matrix{StateType}(undef, nqp, getncells(grid)) 

    for (i,samp) in enumerate(y_samps)
        global x0 = Vec((samp));
        global pp_mag = pp_mag_samps[i];

        b = deepcopy(a[:,i])
        apply_analytical!(b, dh, :d, f)
        apply!(b, ch)
        a[:,i] = deepcopy(b)

        c = deepcopy(a_old[:,i])
        apply!(c, ch)
        a_old[:,i] = deepcopy(c)

    for (j, cell) in enumerate(CellIterator(dh))
        n = ndofs_per_cell(dh)
        ae_old = zeros(n)
        ae = zeros(n)  
        # copy values from a to ae
        map!(j->a[j,i], ae, celldofs(cell))
        map!(j->a_old[j,i], ae_old, celldofs(cell))
        for q_point in 1:nqp
            states_old[q_point,j] = MicroMorphicElasticPhaseFieldState(function_value(cvd, q_point, ae, dof_range(dh, :d)), zero(function_value(cvd, q_point, ae, dof_range(dh, :d))))
            states[q_point,j] = MicroMorphicElasticPhaseFieldState(function_value(cvd, q_point, ae_old, dof_range(dh, :d)), zero(function_value(cvd, q_point, ae_old, dof_range(dh, :d))))
        end
    end
        
        
        push!(states_old_samps,deepcopy(states_old))
        push!(states_samps,deepcopy(states))
    end

    refine_at = 107.0
    time_vector = collect(1:1:130.0)

    data_times = collect(82.0:10.0:196.0) #82.0:...
    save_times = [70.0,79.0,80.0,90.0,99.0,100.0,110.0]

    reaction_forces = ReactionForce.ReactionForcePars(time_vector,n_samps,dh)

    for (step,timestep) in enumerate(time_vector)
        print("Timestep number ")
        print(timestep)
        print("\n")
        t = timestep

        if timestep ∉ time_vector
            continue
        end

        if timestep <= refine_at
            Δt = 1
        else
            Δt = 1.0
        end

        ### if timestep == saved state timestep, load given state here
        # if timestep in save_times
        #     a = load("1d/a_fine_$timestep.jld2", "a")
        # #    a=a'
        #     a_old = load("1d/a_old_fine_$timestep.jld2", "a_old")
        # #    a_old=a_old'
        #     states_samps = load("1d/states_samps_fine_$timestep.jld2", "states_samps")
        #     states_old_samps = load("1d/states_old_samps_fine_$timestep.jld2", "states_old_samps")
        # end
        ###

        Ferrite.update!(ch, t)
        for i in range(start=1,stop=length(y_samps))
            
            b = deepcopy(a[:,i])
            apply!(b, ch)
            a[:,i] .= deepcopy(b)

            c = deepcopy(a_old[:,i])
            apply!(c, ch)
            a_old[:,i] .= deepcopy(c)


            # Perform Newton iterations
            newton_itr = -1
            NEWTON_TOL = 1e-6
            NEWTON_MAXITER = 200
            prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")
            Δa = zeros(ndofs(dh))
            ni = 0
            normr_0 = 0
            print("sample ")
            print(i)
            print("\n")
            while true; newton_itr += 1          
                # Construct the current guess
                apply_zero!(Δa, ch)
                a[:,i] .= a_old[:,i] .+ Δa
                # Compute residual and tangent for current guess
                doassemble!(K, r, material, cv, dh, a[:,i], a_old[:,i], Δt, states_samps[i], states_old_samps[i])
                # Apply boundary conditions
                apply_zero!(K, r, ch)
                apply_zero!(K, rhs, ch)
                # Compute the residual norm and compare with tolerance
                normr = norm(r[Ferrite.free_dofs(ch)])
                if newton_itr == 0
                    normr_0 = normr
                end
                norm_rel = normr/normr_0
                ProgressMeter.update!(prog, norm_rel; showvalues = [(:iter, newton_itr)])
                
               # if normr < NEWTON_TOL
                if norm_rel < 1e-4
                    break
                elseif newton_itr > NEWTON_MAXITER
                    ni += 1
                    newton_itr = -1
                    Δa .*=  0.9993
                    if ni > 3
                        error("Reached maximum Newton iterations, aborting")
                    end
                end
                Δa .-= K \ r

            end
            ReactionForce.postprocess(i, step, doassemble!, reaction_forces, K, r, material, cv, dh, a[:,i], a_old[:,i], Δt, states_samps[i], states_old_samps[i])

        end # for samples
        print("all samples done")

        if timestep in data_times
            a_scaled = deepcopy(a)
            for (i,samp) in enumerate(y_samps)
                sample = deepcopy(a[:,i])
                sample_scaled = ScaleState.scale_it(cv,dh,sample,1/(1.0))
                a_scaled[:,i] .= deepcopy(sample_scaled)
            end
            a = deepcopy(a_scaled)
            pred_mean_scaled, pred_cov_scaled = vec(mean(a_scaled,dims=2)), cov(a_scaled,dims=2)

            a_red = P'*a  #project ensemble members to observation space
            pred_mean, pred_cov = vec(mean(a,dims=2)), cov(a,dims=2)
            S_uncorrected  = cov(SimpleCovariance(), a')
            pred_mean = vec(mean(a,dims=2))
            pred_cov_red = cov(a_red,dims=2)
            pred_var = vec(var(a,dims=2))

            LSE = LinearShrinkage

            method = LSE(ConstantCorrelation())
            S_ledoitwolf = cov(method, a')

            n_obs = 40
            scaled_data = ScaleState.scale_it(cv,dh,time_samples[step],1/(1.0))
            
            Dinv = diagm(1 ./ sqrt.(diag(pred_cov)))
            D = diagm(1 .* sqrt.(diag(pred_cov)))
            pred_corr = cor(a,dims=2)

            pred_corr_replaced = ScaleState.replace_corr(pred_corr,dh,coords)
            replace!(pred_corr, NaN=>0.0)
            replace!(pred_corr_replaced, NaN=>0.0)
            pred_cov_replaced = D * pred_corr_replaced * D


            y = GenerateData.computeData(time_samples[step],sensor_locs,P_data,n_obs)
            y = GenerateData.computeData(scaled_data,sensor_locs,P_data,n_obs)

           
            b=DataFrame(y, :auto)
            df = DataFrame(a=sensor_locs,b=b)
            b[!, "locs"] = sensor_locs
            CSV.write("1d/results/data_$timestep.csv",  b)
            y_slice = y[1]
            jldsave("1d/results/data_$timestep.jld2";y_slice)
            sum_y = sum(y)
            print("Data computed")

            lower = [log(0.98),log(0.0001),log(0.001)]
            upper = [log(1.01),log(1.5),log(2.0)]
            #res = optimize(b -> KalmanFilter.logLikelihood(sensor_locs,y,pred_mean,pred_cov_replaced,n_sens,P,b),lower, upper,[log(1.0),log(0.002),log(0.08)], NelderMead(),
            res = optimize(b -> KalmanFilter.logLikelihood(sensor_locs,y,pred_mean,pred_cov_red,n_sens,P,b),lower, upper,[log(1.0),log(0.002),log(0.08)], NelderMead(),
            Optim.Options(g_tol = 1e-9,
                            iterations = 100,
                            outer_iterations=10,
                            show_trace = false))

            print("Optimizer Done")

            pars = exp.(Optim.minimizer(res))

            posterior_mean, posterior_cov = KalmanFilter.statFEMupdate(pred_mean,pred_cov_replaced,sum_y,n_obs,P,pars,sensor_locs)
            
            
            a_mix = deepcopy(a)
            a_smooth = deepcopy(a)
            coords_train = coords[1:2:end]
            coords_train = [x[1] for x in coords_train]
            P_smoothing = ObsOperator.generateP(coords[1:2:end],dh,grid,cv,ip_geo)

            divSigBefore = deepcopy(a)
            divSigAfter = deepcopy(a)
            for (i,samp) in enumerate(y_samps)
                sample = deepcopy(a[:,i])
                sample_old = deepcopy(a_old[:,i])

                isolated_u = ScaleState.isolate_displacements(cv,dh,sample)
                isolated_d = ScaleState.isolate_phasefield(cv,dh,sample)
                isolated_u_mean = ScaleState.isolate_displacements(cv,dh,pred_mean)
                u_smooth = KalmanFilter.gaussianSmoothing(isolated_u_mean,isolated_u,0.25,coords)
                println(size(u_smooth))
                println(size(isolated_u[1,:]))
                println(size(isolated_d[1,:]))

                u_smooth = (isolated_d[1,:] .* u_smooth) .+ (((isolated_d[1,:] .*(-1.0)) .+ 1.0) .* isolated_u[1,:])

                sample_repl = ScaleState.replace_it(cv,dh,dof_range(dh, :u),sample,repeat(u_smooth,2))         
                a_smooth[:,i] .= deepcopy(sample_repl)
                apply_zero!(a_smooth[:,i], ch)

            end
            pred_mean_smooth, pred_cov_smooth = vec(mean(a_smooth,dims=2)), cov(a_smooth,dims=2)
            heatmap(pred_cov_smooth, yflip=true)
            savefig("1d/results/cov_smoth_$timestep.pdf")
            Dinv = diagm(1 ./ sqrt.(diag(pred_cov_smooth)))
            D = diagm(1 .* sqrt.(diag(pred_cov_smooth)))
            pred_corr_smooth = cor(a_smooth,dims=2)
            heatmap(pred_corr_smooth, yflip=true)
            savefig("1d/results/corr_smooth_$timestep.pdf")

            a_inflated = deepcopy(a_smooth)
            for i in range(start=1,stop=length(y_samps))
                sample = deepcopy(a_smooth[:,i])
                sample_inflated = pred_mean .+ 1.05 .* (sample .- pred_mean)
                a_inflated[:,i] .= deepcopy(sample_inflated)
            end
            pred_cov_inflated = cov(a_inflated,dims=2)

            a_red = P'*a_inflated
            pred_mean = vec(mean(a_inflated,dims=2))
            pred_cov_red = cov(a_red,dims=2)
            pred_var = vec(var(a_inflated,dims=2))
            N = size(a_inflated, 2)
            mean_a = mean(a_inflated, dims=2)
            A′ = (a_inflated .- mean_a) / sqrt(N - 1)
            Y = P' * A′    # size m × N
            cross_cov = A′ * Y'


            for (i,samp) in enumerate(y_samps)
                sample = deepcopy(a_smooth[:,i])
                sample_old = deepcopy(a_old[:,i])
                C = Constraints.doassemble!(K, r, material, cv, dh, a[:,i], a_old[:,i], Δt, states_samps[i], states_old_samps[i])
                divSigBefore[:,i] = C*deepcopy(ScaleState.scale_it(cv,dh,sample,(1e-4 * timestep)))-rhs
                #shifted_sample = KalmanFilter.kalmanShift(sample,pred_mean_smooth,pred_cov_inflated,sum_y,n_obs,P,pars,sensor_locs,dof_coords)
                shifted_sample = KalmanFilter.kalmanShift(sample,pred_mean_smooth,pred_cov_red,cross_cov,sum_y,n_obs,P,pars,sensor_locs,dof_coords)

                isolated_u = ScaleState.isolate_displacements(cv,dh,shifted_sample)
                isolated_d = ScaleState.isolate_phasefield(cv,dh,shifted_sample)

                u_smooth = KalmanFilter.gaussianSmoothing(isolated_u,isolated_u,0.25,coords)
                
                sample_repl = ScaleState.replace_it(cv,dh,dof_range(dh, :u),shifted_sample,repeat(u_smooth,2))

                shifted_sample_analysis = shifted_sample
                shifted_sample = sample_repl


                shifted_sample_rescaled = ScaleState.scale_it(cv,dh,shifted_sample,(1.0))
                divSigAfter[:,i] = C*deepcopy(shifted_sample_rescaled)
                max_val_pp = maximum(isolated_d)           

                println("new cycle start")
                sample_pp_recalc_1 = SeparateComputation.solve_phasefield!(material, cv, dh, ch, shifted_sample_rescaled, shifted_sample_analysis, a_old[:,i], Δt, states_samps[i], states_old_samps[i],4,1)
                sample_disp_repl = SeparateComputation.solve_displacement!(material, cv, dh, ch, sample_pp_recalc_1, shifted_sample_analysis, a_old[:,i], Δt, states_samps[i], states_old_samps[i])
                sample_pp_recalc_2 = deepcopy(sample_disp_repl)
                for it in 1:1#4
                    fac_pp = 1#max_val_pp
                    if it==2
                        fac_pp = 1#max_val_pp
                    end
                    sample_pp_recalc_2 = SeparateComputation.solve_phasefield!(material, cv, dh, ch, sample_disp_repl, shifted_sample_analysis, a_old[:,i], Δt, states_samps[i], states_old_samps[i],1,fac_pp)
                    sample_disp_repl = SeparateComputation.solve_displacement!(material, cv, dh, ch, sample_pp_recalc_2, shifted_sample_analysis, a_old[:,i], Δt, states_samps[i], states_old_samps[i])
                end
    
                print(shifted_sample == a[:,i])

                a[:,i] .= deepcopy(sample_disp_repl)

                print(shifted_sample == a[:,i])
                

            end
            posterior_cov = cov(a.-posterior_mean,dims=2)
            posterior_cov = 0.5*(posterior_cov.+posterior_cov')

            print("update done")

        end
        

        ## save current state here
        if timestep in save_times
            jldsave("1d/results/a_fine_$timestep.jld2";a)
            jldsave("1d/results/a_old_fine_$timestep.jld2";a_old)
            jldsave("1d/results/states_samps_fine_$timestep.jld2";states_samps)
            jldsave("1d/results/states_old_samps_fine_$timestep.jld2";states_old_samps)
            for (i,samp) in enumerate(y_samps)
                sample = deepcopy(a[:,i])
                isolated_u = ScaleState.isolate_displacements(cv,dh,sample)
    
                a_u[:,i] .= deepcopy(vec(isolated_u))
            end

        end  
        ##

        copyto!(a_old, a)
        if timestep in data_times
            for (i,samp) in enumerate(y_samps)
                states_old_samps[i] .= deepcopy(states_samps[i])
                if i < 5
                    vtk_grid("1d/results/phasefield_kalman_divSigBefore-$i-$step", dh) do vtk
                        vtk_point_data(vtk, dh, divSigBefore[:,i])
                        vtk_save(vtk)
                        pvd_list[i][step] = vtk
                    end
                    vtk_grid("1d/results/phasefield_kalman_divSigAfter-$i-$step", dh) do vtk
                        vtk_point_data(vtk, dh, divSigAfter[:,i])
                        vtk_save(vtk)
                        pvd_list[i][step] = vtk
                    end
                end  
            end
        end

        for (i,samp) in enumerate(y_samps)
            states_old_samps[i] .= deepcopy(states_samps[i])
            if i < 30
                vtk_grid("1d/results/phasefield_kalman_bifurc_fine-$i-$step", dh) do vtk
                    vtk_point_data(vtk, dh, a[:,i])
                    vtk_save(vtk)
                    pvd_list[i][step] = vtk
                end
            end  
        end
        post_mean, post_cov = vec(mean(a,dims=2)), cov(a,dims=2)
        
        vtk_grid("1d/results/phasefield_kalman_mean_bifurc_fine-$step", dh) do vtk
            vtk_point_data(vtk, dh, post_mean)
            vtk_save(vtk)
            pvd_mean[step] = vtk
        end  
        vtk_grid("1d/results/phasefield_kalman_cov_bifurc_fine-$step", dh) do vtk
            vtk_point_data(vtk, dh, sqrt.(diag(post_cov)))
            vtk_save(vtk)
            pvd_cov[step] = vtk
        end  
        writedlm( "1d/results/reaction_forces_samples.csv",  reaction_forces.rf_vals, ',')


    end # for timesteps
    for pvd in pvd_list
        vtk_save(pvd);
    end
    vtk_save(pvd_mean)
    vtk_save(pvd_cov)

    
end

n_sens = 25
sensor_locs = Tensors.Vec{1,Float64}[]
locs = range(start=-1.0, stop=1.0, length=n_sens)
for i in range(start=1, step=1, stop=n_sens)
    rand_x = rand(-1:0.001:1)
    rand_x = locs[i]
    push!(sensor_locs,Vec((rand_x)))
end


P = ObsOperator.generateP(sensor_locs,dh,grid,cvs,ip_geo)
P_data = ObsOperator.generateP(sensor_locs,dh_data,grid_data,cvs,ip_geo)

print("P done")
jldsave("1d/sensor_locs.jld2";sensor_locs)
jldsave("1d/P.jld2";P)


struct Run
    a_samp::Vector{Float64}
end

struct AllRuns
    runs::Vector{Run}
end
all_a = AllRuns(Vector{Run}())
a_samps = Vector{Float64}[]
Random.seed!(26)
y_dist_1 = Normal(-0.25,0.12)


n_samps = 120
y_samps_1 = rand(y_dist_1, n_samps)
y_samps = y_samps_1

pp_mag_dist = Uniform(0.72,0.79)
pp_mag_samps = rand(pp_mag_dist, n_samps)


time_samples = load("1d/results/data_time_series_1d.jld2", "a_data")



solve(dh, ch, material, cvs);

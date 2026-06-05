using Revise
using ProgressMeter
using LinearSolve
using Ferrite, FerriteMeshParser
using FerriteGmsh, Gmsh
using LinearAlgebra
using SparseArrays
using JLD2
using HDF5
using DelimitedFiles
using ForwardDiff
using ADTypes: AutoForwardDiff
using TimerOutputs
using IterativeSolvers
using Statistics
using Optim, NLSolversBase, LineSearches
using Random, Distributions
using KernelFunctions
using OhMyThreads: tmapreduce, @tasks, @localize, @allow_boxed_captures, @local
using BenchmarkTools: @btim
using Profile
Profile.init(n=10^7, delay=0.01)
using ProfileView
using Base.Threads: nthreads
using OhMyThreads: tmap!
includet("../methods/TypesPhasefield.jl")
using .TypesPhasefield
includet("../methods/FuncsPhasefield.jl")
using .FuncsPhasefield
includet("../methods/ScaleState.jl")
using .ScaleState
includet("../methods/ReactionForce.jl")
using .ReactionForce
includet("../methods/ErrorNorm.jl")
using .ErrorNorm
includet("../methods/SeparateComputation.jl")
using .SeparateComputation
includet("../methods/SeparateComputationData.jl")
using .SeparateComputationData
includet("../methods/ObsOperator.jl")
using .ObsOperator
includet("../methods/GenerateData.jl")
using .GenerateData
includet("../methods/KalmanFilter.jl")
using .KalmanFilter



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



grid = mktempdir() do dir
    path = joinpath(@__DIR__, "../meshing/SENS.msh")
    togrid(path)
end
coords = compute_vertex_values(grid, x -> x)
grid_data = mktempdir() do dir
    path = joinpath(@__DIR__, "../meshing/SENS_data.msh")
    togrid(path)
end
coords_data = compute_vertex_values(grid_data, x -> x)

grid_coarse = mktempdir() do dir
    path = joinpath(@__DIR__, "../meshing/COARSETEST.msh")
    togrid(path)
end


ip_tet = Lagrange{2,RefTetrahedron,1}()
ip_quad = Lagrange{2,RefCube,1}()

# Interpolations
ipu = Lagrange{2,RefTetrahedron,1}() # quadratic
ipd = Lagrange{2,RefTetrahedron,1}() # linear
ip_geo = Lagrange{2, RefTetrahedron, 1}() # interpolation for the geometry
projector_u = L2Projector(ipu, grid);
projector_d = L2Projector(ipd, grid);
projector_data_u = L2Projector(ipu, grid_data);
ph = PointEvalHandler(grid, coords_data);
ph_data = PointEvalHandler(grid_data, coords_data);
ph_coarse = PointEvalHandler(grid_coarse, coords_data);

dh = DofHandler(grid)
dh_data = DofHandler(grid_data)
dh_coarse = DofHandler(grid_coarse)
add!(dh, :u, 2, ipu)
add!(dh, :d, 1, ipd)
add!(dh_data, :u, 2, ipu)
add!(dh_data, :d, 1, ipd)
add!(dh_coarse, :u, 2, ipu)
add!(dh_coarse, :d, 1, ipd)
close!(dh)
close!(dh_data)
close!(dh_coarse)
renumber!(dh, DofOrder.FieldWise())
renumber!(dh_data, DofOrder.FieldWise())
renumber!(dh_coarse, DofOrder.FieldWise())


dof_coords = zeros(ndofs(dh),2)

for cc in CellIterator(dh)
    cell_dofs = cc.dofs
    cell_coords = cc.coords
    for (i,dof_idx) in enumerate(cell_dofs)
        n_dof = size(cell_coords)[1]
        mapping = [1, 1, 2, 2, 3, 3, 1, 2, 3]
        coord_idx = mapping[i]#(i-1)%n_dof+1 
        coord = cell_coords[coord_idx]
        dof_coords[dof_idx,1] = coord[1]
        dof_coords[dof_idx,2] = coord[2]
    end
end

dof_coords = Vector{eltype(dof_coords)}[eachrow(dof_coords)...]

material = MicroMorphicElasticPhaseField(;
    l=1.5e-2,
    convexification=RateConvexification(),
    fracture=AT2_FM(),
    energy_split=VolumetricSplit()
)


qr = QuadratureRule{2,RefTetrahedron}(2)
cvu = CellVectorValues(qr, ipu, ip_geo)
cvd = CellScalarValues(qr, ipd, ip_geo)
cvs = (cvu,cvd)

n_threads = Threads.nthreads()
cvs_per_thread = [(CellVectorValues(qr, ipu, ip_geo),CellScalarValues(qr, ipd, ip_geo)) 
                         for _ in 1:n_threads]
K_per_thread = [create_sparsity_pattern(dh) 
                        for _ in 1:n_threads]
r_per_thread = [zeros(ndofs(dh)) 
                        for _ in 1:n_threads]
                        

ch = ConstraintHandler(dh)
load_function(x, t) = Vec((1e-4 * t, 0.0))
add!(ch, Ferrite.Dirichlet(:u, getfaceset(grid, "bottom"), Returns(zero(Vec{2}))))

add!(ch, Ferrite.Dirichlet(:u, getfaceset(grid, "left"), x -> [0.0], [2]))
add!(ch, Ferrite.Dirichlet(:u, getfaceset(grid, "right"), x -> [0.0], [2]))

add!(ch, Ferrite.Dirichlet(:u, getfaceset(grid, "top"), load_function))
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
    


function element_routine!(re::AbstractVector, material::MicroMorphicElasticPhaseField, cvs, cell, ae::AbstractVector, ae_old, Δt, dh, state, state_old)
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
          #  ϕhat = calculate_phasefield(material, Ψ⁰, Ψ⁻, dhat, ⁿϕ)
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
    a = deepcopy(a_updated)
    a_new = deepcopy(a_updated)
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



function element_routine_states!(material::MicroMorphicElasticPhaseField, cvs, cell, ae::AbstractVector, ae_old, Δt, dh, state, state_old, first)
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





d_peak = 0.9
r_cut = 0.025#0.013
d_cut = d_peak/100 # d when r=r_cut
x0 = Vec((0.57, -0.23))
f(x::Vec) = f(norm(x-x0))
k = (1/d_cut-1/d_peak)/r_cut
f(r::Number) = r<r_cut ? (0.95*exp(-r*r/(2*0.011^2))) : 0.0

function solve(dh, ch, material, cv)
    # Pre-allocate solution vectors, etc.
    pvd = paraview_collection("./results/phase_field_results_bifurc_fine.pvd");
    pvd_mean = paraview_collection("./results/phase_field_kalman_mean_bifurc_fine.pvd");
    pvd_cov = paraview_collection("./results/phase_field_kalman_cov_bifurc_fine.pvd");
    pvd_list = []
    for (i,samp) in enumerate(y_samps)
        name = "phasefield_kalman_sample_bifurc_fine"*string(i)*".pvd"
        push!(pvd_list,paraview_collection(name))
    end

    a = zeros(ndofs(dh),n_samps)
    b = zeros(ndofs(dh))
    a_old = deepcopy(a)
    ΔΔa = zeros(ndofs(dh))

    cvu, cvd = cv # Previously, the global cvd was used. 
    nqp = getnquadpoints(cvd)
    states_old_samps = []
    states_samps = []

    skip = zeros(n_samps)

    for (i,samp) in enumerate(y_samps)
        global x0 = Vec((x_samps[i], samp));
        b = deepcopy(a[:,i])
        apply_analytical!(b, dh, :d, f)
        a[:,i] = deepcopy(b)
        apply!(view(a_old,:,i), ch)
        apply!(view(a,:,i), ch)
        StateType   = typeof(MicroMorphicElasticPhaseFieldState(0.0,0.0))
        states      = Matrix{StateType}(undef, nqp, getncells(grid)) 
        states_old  = Matrix{StateType}(undef, nqp, getncells(grid)) 

        n = ndofs_per_cell(dh)
        ae_old = zeros(n)
        ae = zeros(n)   
        for (j, cell) in enumerate(CellIterator(dh))
                # copy values from a to ae
                map!(j->a[j,i], ae, celldofs(cell))
                map!(j->a_old[j,i], ae_old, celldofs(cell))
                for q_point in 1:nqp
                    d = function_value(cvd, q_point, ae, dof_range(dh, :d))
                    d_old = function_value(cvd, q_point, ae_old, dof_range(dh, :d))
                    states_old[q_point,j] = MicroMorphicElasticPhaseFieldState(d, zero(d))
                    states[q_point,j] = MicroMorphicElasticPhaseFieldState(d_old, zero(d_old))
                end
            end
        
        
        push!(states_old_samps,deepcopy(states_old))
        push!(states_samps,deepcopy(states))
    end

    refine_at = 70.0

    time_vector_full = append!(collect(1.0:1:refine_at), collect(refine_at+0.1:0.1:120.0))
    time_vector = collect(105.0:0.1:120.0)
    time_vector = time_vector_full

    data_times = [105.1,115.1]
    save_times = [70.0,90.0,105.0,115.0]
    load_times = []

    reaction_forces = ReactionForce.ReactionForcePars(time_vector_full,n_samps,dh,faceset="top",dim=2)
    error_norms = ErrorNorm.ErrorNormPars(time_vector_full,n_samps,dh,dh_data,ph,ph_data)
    cracktips_mat = zeros(length(time_vector_full), n_samps, 2)

    for (step,timestep) in enumerate(time_vector_full)
        print("Timestep number ")
        print(timestep)
        print("\n")
        t = timestep

        if timestep ∉ time_vector
            continue
        end

        if timestep <= refine_at
            Δt = 1 * 1e-4
        else
            Δt = 0.1 * 1e-4
        end

        ### if timestep == saved state timestep, load given state here
        if timestep in load_times
            a_load = load("./results/a_fine_Vol_review_$timestep.jld2", "a")
            a = a_load[:, 1:n_samps]
            a_old_load = load("./results/a_old_fine_Vol_review_$timestep.jld2", "a_old")
            a_old = a_old_load[:, 1:n_samps]
            states_samps = load("./results/states_samps_fine_Vol_review_$timestep.jld2", "states_samps")
            states_old_samps = load("./results/states_old_samps_fine_Vol_review_$timestep.jld2", "states_old_samps")
        end
        ###
        Ferrite.update!(ch, t)

        elapsed_time = @elapsed begin
        let states_samps = states_samps, states_old_samps = states_old_samps, Δt=Δt, a_old = a_old, a=a

        @tasks for i in range(start=1,stop=length(y_samps))
            thread_id = Threads.threadid()
            cv_local = cvs_per_thread[thread_id]

            K_local = create_sparsity_pattern(dh); # tangent stiffness matrix
            r_local = zeros(ndofs(dh))
            apply!(view(a_old,:,i), ch)
            apply!(view(a,:,i), ch)

            # Perform Newton iterations
            newton_itr = -1
            NEWTON_MAXITER = 80

            local Δa = zeros(ndofs(dh)) 
            normr_0 = 0
            println("sample "*string(i)*" START")
            if skip[i] == 0
                while true; newton_itr += 1          
                    # Construct the current guess
                    apply_zero!(Δa, ch)
                    @views a[:,i] .= a_old[:,i] .+ Δa
                    # Compute residual and tangent for current guess
                    @views doassemble!(K_local, r_local, material, cv_local, dh, a[:,i], a_old[:,i], Δt, states_samps[i], states_old_samps[i])
                    # Apply boundary conditions
                    apply_zero!(K_local, r_local, ch)
                    # Compute the residual norm and compare with tolerance
                    normr = norm(r_local[Ferrite.free_dofs(ch)])
                    if newton_itr == 0
                        normr_0 = normr
                    end
                    norm_rel = normr/normr_0

                    if norm_rel < 1e-3
                        break
                    elseif newton_itr > NEWTON_MAXITER
                            println("Reached maximum Newton iterations")
                            println(step)
                            println(i)
                            println(skip[i])
                            @views skip[i] = 1 # sample will be skipped in the next time steps
                            break
                       # end
                    end
                    Δa .-= K_local \ r_local

                end
            elseif skip[i] == 1
               @views local a[:,i] .= a_old[:,i]
            end

            if skip[i] == 0
                # compute reaction forces
                @views ReactionForce.postprocess(i, step, doassemble!, reaction_forces, K_local, r_local, material, cv_local, dh, a[:,i], a_old[:,i], Δt, states_samps[i], states_old_samps[i], dim=2)
            elseif skip[i] == 1
                println("skipped sample, previous Newton iteration didn't converge. Reaction force remains zero.")
            end
            try
                # compute error norms for all fields
                local i_data = searchsortedfirst(time_vector, timestep)
                @views ErrorNorm.postprocess(i, step, i_data, error_norms, time_samples, a[:,i],coords)
            catch
                println("data object ran out of timesteps")
            end
            println("sample "*string(i)*" END")



        end # for samples
        end #let
        print("all samples done")
        end # elapsed time end
        println("Time taken: ", elapsed_time, " seconds")
        

        ####
        ### if at a time step data is available, begin the EnKF analysis loop 
        ####
        if timestep in data_times
            elapsed_time = @elapsed begin
            
            a_red = P'*a  #project ensemble members to observation space

            # compute statistcal moments:
            pred_mean = vec(mean(a,dims=2))
            pred_cov_red = cov(a_red,dims=2)
            pred_var = vec(var(a,dims=2))

            # compute cross covariance:
            N = size(a, 2)
            mean_a = mean(a, dims=2)
            A′ = (a .- mean_a) / sqrt(N - 1)
            Y = P' * A′    # size m × N
            cross_cov = A′ * Y'

            n_obs = 20 # number of repeated (independent) observations per sensor
            i_data = searchsortedfirst(time_vector, timestep)
            y = GenerateData.computeData(time_samples[step],repeat(sensor_locs,2),P_data,n_obs) # generate artificial mesaurement data
            sum_y = sum(y)
            print("Data computed")
            
            # perform hyperparameter optimization:
            lower = [log(0.95),log(0.00002),log(0.001)] # bounds for the parameters
            upper = [log(1.05),log(0.0003),log(2.0)]
             res = optimize(b -> KalmanFilter.logLikelihood(repeat(sensor_locs,2),y,pred_mean,pred_cov_red,n_sens,P,b),lower, upper,[log(0.999),log(0.000025),log(0.08)], NelderMead(),
             Optim.Options(g_tol = 1e-9,
                             iterations = 20,
                             outer_iterations=10,
                             show_trace = false))

            print("Optimizer Done")
            pars = exp.(Optim.minimizer(res))
            println("EnKF hyperparameters:")
            println(pars)

            a_smooth = deepcopy(a)
            for (i,samp) in enumerate(y_samps)
                sample = deepcopy(a[:,i])
                isolated_u = ScaleState.isolate_displacements(cv,dh,sample)
                isolated_u_mean = ScaleState.isolate_displacements(cv,dh,pred_mean)
                println(size(sample))
                println(isolated_u[:,1234])
                u_smooth = copy(isolated_u)
                isolated_u[1,:] = sample[1:2:Int(size(sample)[1]*(2/3))-1]#isolated_u[3,:]
                isolated_u[2,:] = sample[2:2:Int(size(sample)[1]*(2/3))]
                isolated_u[3,:] = sample[Int(size(sample)[1]*(2/3))+1:1:end] # phase field
                isolated_u_mean[1,:] = pred_mean[1:2:Int(size(pred_mean)[1]*(2/3))-1]#isolated_u[3,:]
                isolated_u_mean[2,:] = pred_mean[2:2:Int(size(pred_mean)[1]*(2/3))]
                isolated_u_mean[3,:] = isolated_u[3,:]
                println(size(stack(isolated_u)))
                println(size(reduce(vcat,isolated_u)))
                println(reduce(vcat,u_smooth)[1:4])
                println(size(sample))
                
                println(u_smooth[1,1:9])
                println(collect(Iterators.flatten(zip(u_smooth[1,:],u_smooth[2,:])))[1:9])

                println(dof_coords[1:9])
                println(sample[1:9])

                ### smoothing the prior with GP regression
                coords_high_pp = findall(>(0.18), isolated_u[3,:])
                coords_low_pp = findall(<(0.1), isolated_u[3,:])
                phasefield_high = isolated_u[3,:].-0.1
                coords_neg = findall(<(0.), phasefield_high)
                phasefield_high[coords_neg] .= 0.0
                phasefield_high = phasefield_high./0.9

                # u_smooth[1,:] = KalmanFilter.smooth2D(isolated_u_mean[1,:],isolated_u[1,:],0.25,dof_coords[1:2:Int(size(sample)[1]*(2/3))-1],coords_high_pp)
                # u_smooth[1,:] = (phasefield_high .* u_smooth[1,:]) .+ (((phasefield_high .*(-1.0)) .+ 1.0) .* isolated_u[1,:])

                # u_smooth[2,:] = KalmanFilter.smooth2D(isolated_u_mean[2,:],isolated_u[2,:],0.25,dof_coords[2:2:Int(size(sample)[1]*(2/3))],coords_high_pp)
                # u_smooth[2,:] = (phasefield_high .* u_smooth[2,:]) .+ (((phasefield_high .*(-1.0)) .+ 1.0) .* isolated_u[2,:])

                # u_smooth[1,:] = KalmanFilter.smooth2D(isolated_u_mean[1,:],isolated_u[1,:],0.25,dof_coords[1:2:Int(size(sample)[1]*(2/3))-1],coords_high_pp)
                # u_smooth[1,:] = (isolated_u[3,:] .* u_smooth[1,:]) .+ (((isolated_u[3,:] .*(-1.0)) .+ 1.0) .* isolated_u[1,:])

                # u_smooth[2,:] = KalmanFilter.smooth2D(isolated_u_mean[2,:],isolated_u[2,:],0.25,dof_coords[2:2:Int(size(sample)[1]*(2/3))],coords_high_pp)
                # u_smooth[2,:] = (isolated_u[3,:] .* u_smooth[2,:]) .+ (((isolated_u[3,:] .*(-1.0)) .+ 1.0) .* isolated_u[2,:])
                    
                #u_smooth[2,:] = KalmanFilter.smooth2D(isolated_u_mean[2,:],isolated_u[2,:],0.25,stack(coords, dims=2))
                u_smooth[1,:] = isolated_u[1,:]
                
                u_smooth[2,:] = isolated_u[2,:]
                print("smoothed sample nr. ")
                println(i)
                sample_new = deepcopy(a[:,i])
               # quit()
                println(size(collect(Iterators.flatten(zip(u_smooth[1,:],u_smooth[2,:])))))
                
                sample_new[1:Int(size(sample)[1]*(2/3))] = collect(Iterators.flatten(zip(u_smooth[1,:],u_smooth[2,:])))
                sample_repl = ScaleState.replace_it(cv,dh,dof_range(dh, :u),sample,sample_new)
                sample_repl = sample_new          
                a_smooth[:,i] .= deepcopy(sample_repl)

                apply_zero!(a_smooth[:,i], ch)

            end
            pred_mean = vec(mean(a_smooth,dims=2))


            ### Covariance Inflation
            a_inflated = deepcopy(a_smooth)
            let pred_mean = pred_mean
            @tasks for i in range(start=1,stop=length(y_samps))
                sample = deepcopy(a_smooth[:,i])
                sample_inflated = pred_mean .+ 1.05 .* (sample .- pred_mean)  ### was 1.15!
                a_inflated[:,i] .= deepcopy(sample_inflated)
            end
            end

            a_red = P'*a_inflated
            pred_mean = vec(mean(a_inflated,dims=2))
            pred_cov_red = cov(a_red,dims=2)
            pred_var = vec(var(a_inflated,dims=2))
            N = size(a_inflated, 2)
            mean_a = mean(a_inflated, dims=2)
            A′ = (a_inflated .- mean_a) / sqrt(N - 1)
            Y = P' * A′    # size m × N
            cross_cov = A′ * Y'


            
            ###
            # for each ensemble member, perform the Kalman shift and regularization steps
            ###
            let pred_mean = pred_mean, pred_cov_red = pred_cov_red, cross_cov = cross_cov, states_samps = states_samps, states_old_samps = states_old_samps, cvd = cvd, cvu = cvu, a_old = a_old, a = a, Δt = Δt
            @allow_boxed_captures @tasks for i in range(start=1,stop=length(y_samps))
                println("copying")
                sample = deepcopy(a_smooth[:,i])
                println("start filter")
                shifted_sample = KalmanFilter.kalmanShift(sample,pred_mean,pred_cov_red,cross_cov,sum_y,n_obs,P,pars,repeat(sensor_locs,2),dof_coords)

                println("stagger")
                #### Staggered regularization
                cvu, cvd = cv
                cv_local = copy(cvu), copy(cvd)
                @views sample_pp_recalc_1 = SeparateComputation.solve_phasefield!(material, cv_local, dh, ch, shifted_sample, shifted_sample, a_old[:,i], Δt, states_samps[i], states_old_samps[i],4,1)#2,1
                @views sample_disp_repl = SeparateComputation.solve_displacement!(material, cv_local, dh, ch, sample_pp_recalc_1, shifted_sample, a_old[:,i], Δt, states_samps[i], states_old_samps[i])
                @views sample_pp_recalc_2 = deepcopy(sample_disp_repl)#SeparateComputation.solve_phasefield!(material, cv_local, dh, ch, sample_disp_repl, a_old[:,i], Δt, states_samps[i], states_old_samps[i],1,1)  
                for it in 1:4
                    @views sample_pp_recalc_2 = SeparateComputation.solve_phasefield!(material, cv_local, dh, ch, sample_disp_repl, shifted_sample, a_old[:,i], Δt, states_samps[i], states_old_samps[i],1,1)
                    @views sample_disp_repl = SeparateComputation.solve_displacement!(material, cv_local, dh, ch, sample_pp_recalc_2, shifted_sample, a_old[:,i], Δt, states_samps[i], states_old_samps[i])
                end
                ####
                println("assemble states")
                a[:,i] .= deepcopy(sample_disp_repl)  #this!

            end
            print("update done")
            end
            
            end # elapsed time end
            println("Time taken for update: ", elapsed_time, " seconds")
            

        end
        
        println("Data times done")


        ## save current state here
        if timestep in save_times
            jldsave("./results/a_fine_Vol_review_$timestep.jld2";a)
            jldsave("./results/a_old_fine_Vol_review_$timestep.jld2";a_old)
            jldsave("./results/states_samps_fine_Vol_review_$timestep.jld2";states_samps)
            jldsave("./results/states_old_samps_fine_Vol_review_$timestep.jld2";states_old_samps)
        end  
        ##

        println("Save times done")

        copyto!(a_old, a)
        let states_samps = states_samps, states_old_samps = states_old_samps, a=a
        @tasks for i in range(start=1,stop=length(y_samps))
            states_old_samps[i] .= deepcopy(states_samps[i])
            if i in [1,2,3,4,20,21,22]#30
                vtk_grid("./results/phasefield_kalman_noup_VOL_5samps-$i-$step", dh) do vtk
                    vtk_point_data(vtk, dh, a[:,i])
                    vtk_save(vtk)
                    pvd_list[i][step] = vtk
                end
            end  
        end
        end
        post_mean, post_cov = vec(mean(a,dims=2)), vec(var(a,dims=2))
        vtk_grid("./results/phasefield_kalman_noup_VOL_5samps-$step", dh) do vtk
            vtk_point_data(vtk, dh, post_mean)
            vtk_save(vtk)
            pvd_mean[step] = vtk
        end  
        vtk_grid("./results/phasefield_kalman_cov_noup_VOL_5samps-$step", dh) do vtk
            vtk_point_data(vtk, dh, post_cov)
            vtk_save(vtk)
            pvd_cov[step] = vtk
        end  

        println("vtk save done")


    end # for timesteps

    
    for pvd in pvd_list
        vtk_save(pvd);
    end
    vtk_save(pvd_mean)
    vtk_save(pvd_cov)
    
end

n_sens = 150
Random.seed!(1213123)
sensor_locs = Tensors.Vec{2,Float64}[]
for i in range(start=1, step=1, stop=n_sens)
    rand_x = rand(0.5:0.001:0.875)
    rand_y = rand(-0.2:0.001:0.2) #both 0.15
    push!(sensor_locs,Vec((rand_x,rand_y)))
end
rotgrid(grd, θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)] * grd
P0 = [0.5,0]
sensor_locs_rot = rotgrid.(sensor_locs .- Ref(P0), -1.2*π/4) .+ Ref(P0)
sensor_locs_rot_vec = Tensors.Vec{2,Float64}[]
for coord in sensor_locs_rot
    push!(sensor_locs_rot_vec,Vec((coord[1],coord[2])))
end
sensor_locs = sensor_locs_rot_vec
P = ObsOperator.generateP(sensor_locs,dh,grid,cvs,ip_geo)
P_data = ObsOperator.generateP(sensor_locs,dh_data,grid_data,cvs,ip_geo)



print("P done")
jldsave("sensor_locs.jld2";sensor_locs)
jldsave("P.jld2";P)

struct Run
    a_samp::Vector{Float64}
end

struct AllRuns
    runs::Vector{Run}
end
all_a = AllRuns(Vector{Run}())
a_samps = Vector{Float64}[]
Random.seed!(7)#4
y_dist_1 = Normal(-0.065,0.032)
Random.seed!(9)
n_samps = 100

y_dist_2 = Beta(8,8)
x_dist_2 = Beta(8,8)
y_samps_2 = rand(y_dist_2, n_samps).*(-0.02-(-0.11)).+(-0.11) 
x_samps_2 = rand(x_dist_2, n_samps).*(0.62-(0.51)).+(0.51)

y_samps = y_samps_2
x_samps = x_samps_2
n_samps = length(y_samps)

time_samples = load("./results/data_time_series_fine_test_VOL.jld2", "a_data")
solve(dh, ch, material, cvs);

using ProgressMeter
using LinearSolve
using Ferrite, FerriteMeshParser
using FerriteGmsh, Gmsh
using LinearAlgebra
using SparseArrays
using StaticArrays
using Tensors
using JLD2
include("../methods/ReactionForce.jl")
using .ReactionForce
include("../methods/ErrorNorm.jl")
using .ErrorNorm
using DelimitedFiles
using ForwardDiff
using TimerOutputs
using IterativeSolvers
using Statistics
using Random, Distributions
using Plots


# Energy splits 
struct NoSplit end
struct SpectralSplit end # Not implemented
struct VolumetricSplit end
struct HybridSplit end

# Fracture models
struct AT1_FM end
get_cw(::AT1_FM) = 8 / 3
struct AT2_FM end
get_cw(::AT2_FM) = 2.0

struct QuasiBrittle_FM{T} # Not fully implemented yet
    ft::T # Tensile strength
    p::T
    a2::T
    a3::T
end

# Convexifications
struct NoConvexification end
struct RateConvexification end

# Main material and state variables
struct MicroMorphicElasticPhaseField{T,ST,FM,C}
    G::T
    K::T
    Gc::T
    l::T
    α::T
    cw::T
    g_resid::T
    energy_split::ST
    fracture::FM
    convexification::C
end
function MicroMorphicElasticPhaseField(;
    E=210e3, ν=0.3, Gc=2.7, l=1.5e-2, β=200.0,
    g_resid=1.e-10, # Same as in Ritu's code https://github.com/ritukeshbharali/falcon/blob/696ea6dd220927db8a138d0fbdfb94dc627d8169/src/fem/solidmech/MicroPhaseFractureExtModel.cpp#LL790C30-L790C36
    energy_split=VolumetricSplit(),
    fracture=AT2_FM(),
    convexification=NoConvexification())

    G = E / (2 * (1 + ν))
    K = E / (3 * (1 - 2ν))
    cw = get_cw(fracture)
    α = β * Gc / l
    return MicroMorphicElasticPhaseField(G, K, Gc, l, α, cw, g_resid, energy_split, fracture, convexification)
end

struct MicroMorphicElasticPhaseFieldState{T}
    ϕ::T
    d_rate::T #
end

# Macaulay brackets
function macaulay(x)
    return 0.5*(x+abs(x))#max(0.0, x)
end
function macaulay_neg(x)
    return 0.5*(x-abs(x))#return max(0.0, -x)
end


function spectral_decomposition(ε::SymmetricTensor{2,dim,T,N}) where {dim,T,N}
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = eigen(Symmetric(ε.data))  # Ensure symmetric for numerical stability

    # Compute positive and negative strain parts
    ε_plus  = zero(ε)  # Initialize with correct type
    ε_minus = zero(ε)

    for i in 1:dim
        λi = eigvals[i]
        ni = eigvecs[:, i]

        # Macaulay brackets
        λ_plus  = max(λi, 0)
        λ_minus = max(-λi, 0)

        # Reconstruct tensors using outer product
        ε_plus  += λ_plus  * (ni ⊗ ni)  # Ensure type compatibility
        ε_minus += λ_minus * (ni ⊗ ni)
    end

    return ε_plus, ε_minus
end



function split_strain(ϵ::SymmetricTensor{2,2,T}) where T

    #  eigen_decomp = eigen(Symmetric(ϵ_mat))
    λ, V = eigen(MArray{Tuple{2,2}}(ϵ))

    # Project onto positive and negative parts
    tolerance = 0.0
    λ_pos = max.(λ, tolerance)  # Keep only positive eigenvalues above tolerance
    λ_neg = min.(λ, -tolerance) # Keep only negative eigenvalues below -tolerance

    # Reconstruct the tensors
    ϵ_pos = V * Diagonal(λ_pos) * V'
    ϵ_neg = V * Diagonal(λ_neg) * V'

   ϵ⁺ = SymmetricTensor{2,2,T}(Symmetric(ϵ_pos))
   ϵ⁻ = SymmetricTensor{2,2,T}(Symmetric(ϵ_neg))


    return ϵ⁺, ϵ⁻
end



# Energy splits: NoSplit, VolumetricSplit, or SpectralSplit
split_energy(m::MicroMorphicElasticPhaseField, ϵ) = split_energy(m.energy_split, m, ϵ)
function split_energy(::NoSplit, m::MicroMorphicElasticPhaseField, ϵ)
    λ = m.K - 2 * m.G / 3
    Ψ⁺ = 0.5 * λ * tr(ϵ)^2 + m.G * (ϵ ⊡ ϵ)
    Ψ⁻ = zero(Ψ⁺)
    return Ψ⁺, Ψ⁻
end
function split_energy(::VolumetricSplit, m::MicroMorphicElasticPhaseField, ϵ)
    ϵdev = dev(ϵ)
    Ψ⁺ = 0.5 * m.K * macaulay(tr(ϵ))^2 + m.G * tr((ϵdev ⊡ ϵdev))
    Ψ⁻ = 0.5 * m.K * macaulay_neg(tr(ϵ))^2
    return Ψ⁺, Ψ⁻
end

function split_energy(::HybridSplit, m::MicroMorphicElasticPhaseField, ϵ)
    ϵdev = dev(ϵ)
    λ = m.K - 2 * m.G / 3
    Ψ⁺ = 0.5 * m.K * macaulay(tr(ϵ))^2 + m.G * tr((ϵdev ⊡ ϵdev))
    Ψ⁻ = 0.5 * m.K * macaulay_neg(tr(ϵ))^2
    Ψ⁰  = 0.5 * λ * tr(ϵ)^2 + m.G * (ϵ ⊡ ϵ)
    return Ψ⁺, Ψ⁻, Ψ⁰
end

function split_energy(::SpectralSplit, m::MicroMorphicElasticPhaseField, ϵ)
    λ = m.K - 2 * m.G / 3   
    ϵ⁺, ϵ⁻ = split_strain(ϵ)
   
    Ψ⁺ = 0.5 * λ * macaulay(tr(ϵ⁺))^2 + m.G * tr((ϵ⁺ ⊡ ϵ⁺))
    Ψ⁻ = 0.5 * λ * macaulay_neg(tr(ϵ⁻))^2+ m.G * tr((ϵ⁻ ⊡ ϵ⁻))
    return Ψ⁺, Ψ⁻, ϵ⁺, ϵ⁻
end

calculate_degraded_stress(m::MicroMorphicElasticPhaseField, args...) = calculate_degraded_stress(m.energy_split, m, args...)
function calculate_degraded_stress(::VolumetricSplit, m::MicroMorphicElasticPhaseField, ϵ, gϕ)
    return (gϕ * 2 * m.G) * dev(ϵ) + (m.K * (gϕ * macaulay(tr(ϵ)) - macaulay(-tr(ϵ)))) * one(ϵ)
end
function calculate_degraded_stress(::NoSplit, m::MicroMorphicElasticPhaseField, ϵ, gϕ)
    return (gϕ * 2 * m.G) * dev(ϵ) + (3 * gϕ * m.K) * vol(ϵ)
end
function calculate_degraded_stress(::HybridSplit, m::MicroMorphicElasticPhaseField, ϵ, gϕ)
    return (gϕ * 2 * m.G) * dev(ϵ) + (3 * gϕ * m.K) * vol(ϵ)
end

function calculate_degraded_stress(::SpectralSplit, m::MicroMorphicElasticPhaseField, ϵ, ϵ⁺,ϵ⁻, gϕ)
    return  (gϕ *2 * m.G) * dev(ϵ⁺) + (3 * gϕ * m.K) * vol(ϵ⁺)  + (2 * m.G) * dev(ϵ⁻) + (3 * m.K) * vol(ϵ⁻)
end

degradation_function(m::MicroMorphicElasticPhaseField, ϕ) = degradation_function(m.fracture, m, ϕ)
degradation_function(::Union{AT1_FM,AT2_FM}, ::MicroMorphicElasticPhaseField, ϕ) = (1 - ϕ)^2
function degradation_function(f::QuasiBrittle_FM, m::MicroMorphicElasticPhaseField, ϕ)
    a1 = 4 * m.E * m.Gc / (π * m.l * f.ft^2)
    one_minus_phi_power_p = (1 - ϕ)^m.p
    return one_minus_phi_power_p / (one_minus_phi_power_p + a1 * ϕ * (1 + a2 * ϕ * (1 + a3 * ϕ)))
end

calculate_phasefield(m::MicroMorphicElasticPhaseField, args...) = calculate_phasefield(m.fracture, m, args...)
function calculate_phasefield(::AT1_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T, ⁿϕ) where {T}
    return min(max((2 * Ψ⁺ + m.α * d - 3 * m.Gc / (8 * m.l)) / (2 * Ψ⁺ + m.α), convert(T, ⁿϕ)), one(T))
end
function calculate_phasefield(::AT2_FM, m::MicroMorphicElasticPhaseField, Ψ⁺, Ψ⁻, d::T, ⁿϕ) where {T}
    return min(max((2 * Ψ⁺ + m.α * d) / (2 * Ψ⁺ + m.α + m.Gc / m.l), convert(T, ⁿϕ)), one(T))
end


grid = mktempdir() do dir
    path = joinpath(@__DIR__, "../meshing/SENS_data.msh")
    togrid(path)
end


ip_tet = Lagrange{2,RefTetrahedron,1}()
ip_quad = Lagrange{2,RefCube,1}()

# Interpolations
ipu = Lagrange{2,RefTetrahedron,1}() # quadratic
ipd = Lagrange{2,RefTetrahedron,1}() # linear
ip_geo = Lagrange{2, RefTetrahedron, 1}() # interpolation for the geometry

dh = DofHandler(grid)
add!(dh, :u, 2, ipu)
add!(dh, :d, 1, ipd)
close!(dh)
renumber!(dh, DofOrder.FieldWise())

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


qr = QuadratureRule{2,RefTetrahedron}(1)
cvu = CellVectorValues(qr, ipu, ip_geo)
cvd = CellScalarValues(qr, ipd, ip_geo)
cvs = (cvu,cvd)

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
    σe = zeros(n)
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
    # Setup cellvalues and give easier names
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
        #println(typeof(ϵ))
        d = function_value(cvd, q_point, ae, dofrange_d)
        ∇d = function_gradient(cvd, q_point, ae, dofrange_d)
        old_qp_state = state_old[q_point]

        ⁿϕ = old_qp_state.ϕ
       # Ψ⁺, Ψ⁻, ϵ⁺, ϵ⁻ = split_energy(material, ϵ) # Spectral (not implemented)
       # Ψ⁺, Ψ⁻, Ψ⁰ = split_energy(material, ϵ) # Hybrid
        Ψ⁺, Ψ⁻ = split_energy(material, ϵ) # VolDev/No
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

        #σ = calculate_degraded_stress(material, ϵ, ϵ⁺, ϵ⁻, gϕ_fix) # Spectral
        σ = calculate_degraded_stress(material, ϵ, gϕ_fix) # all others

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


d_peak = 0.9
r_cut = 0.025
d_cut = d_peak/100 # d when r=r_cut
x0 = Vec((0.57, -0.23))
f(x::Vec) = f(norm(x-x0))
k = (1/d_cut-1/d_peak)/r_cut
f(r::Number) = r<r_cut ? (0.95*exp(-r*r/(2*0.011^2))) : 0.0



function solve(dh, ch, material, cv)
    pvd = paraview_collection("./results/phase_field_results.pvd");
    K = create_sparsity_pattern(dh); # tangent stiffness matrix
    r = zeros(ndofs(dh))
    a = zeros(ndofs(dh))
    Δa = zeros(ndofs(dh))
    ΔΔa = zeros(ndofs(dh))

    apply_analytical!(a, dh, :d, f)
    a_old = deepcopy(a)
    apply!(a_old, ch)
    apply!(a, ch)

    cvu, cvd = cv # Previously, the global cvd was used. 
    nqp = getnquadpoints(cvd)

    ### Initial damage field
    StateType   = typeof(MicroMorphicElasticPhaseFieldState(0.0,0.0))
    states      = Matrix{StateType}(undef, nqp, getncells(grid)) 
    states_old  = Matrix{StateType}(undef, nqp, getncells(grid)) 

    n = ndofs_per_cell(dh)
    ae_old = zeros(n)
    ae = zeros(n)

    for (i, cell) in enumerate(CellIterator(dh))
        # copy values from a to ae
        map!(i->a[i], ae, celldofs(cell))
        map!(i->a_old[i], ae_old, celldofs(cell))
        for q_point in 1:nqp
            d = function_value(cvd, q_point, ae, dof_range(dh, :d))
            d_old = function_value(cvd, q_point, ae_old, dof_range(dh, :d))
            states_old[q_point,i] = MicroMorphicElasticPhaseFieldState(d, zero(d))
            states[q_point,i] = MicroMorphicElasticPhaseFieldState(d_old, zero(d_old))
        end
    end


    refine_at = 70.0
    
    time_vector = append!(collect(1.0:1:refine_at), collect(refine_at+0.1:0.1:124.2))
    reaction_forces = ReactionForce.ReactionForcePars(time_vector,10,dh,faceset="top",dim=2)
    error_norms = ErrorNorm.ErrorNormPars(time_vector,1,0,0,0,0)
    cracktips_mat = zeros(length(time_vector), 3)
 
    for (step,timestep) in enumerate(time_vector)
        print("Timestep number ")
        print(timestep)
        print("\n")
        t = timestep
        if timestep <= refine_at
            Δt = 1.0 * 1e-4
        else
            Δt = 0.1 * 1e-4#0.1
        end
        Ferrite.update!(ch, t)
        apply!(a, ch)
        apply!(a_old, ch)
        
        # Perform Newton iterations ### I believe this is the slowest part!
        newton_itr = -1
        NEWTON_TOL = 1e-6
        NEWTON_MAXITER = 200
        prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")
        Δa = zeros(ndofs(dh))
        ni = 0
        normr_0 = 0
        while true; newton_itr += 1          
            # Construct the current guess
            apply_zero!(Δa, ch)
            a .= a_old .+ Δa
            # Compute residual and tangent for current guess
            doassemble!(K, r, material, cv, dh, a, a_old, Δt, states, states_old)
            # Apply boundary conditions
            apply_zero!(K, r, ch)
            # Compute the residual norm and compare with tolerance
            normr = norm(r[Ferrite.free_dofs(ch)])
            if newton_itr == 0
                normr_0 = normr
            end
            norm_rel = normr/normr_0
            #print(normg)
            ProgressMeter.update!(prog, normr; showvalues = [(:iter, newton_itr)])
            #if normr < NEWTON_TOL
            if norm_rel < 1e-3 #-3
                break
            elseif newton_itr > NEWTON_MAXITER
                ni += 1
                newton_itr = -1
                Δa .*=  0.9993
                if ni > 3
                    error("Reached maximum Newton iterations, aborting")
                end
            end

            # Compute increment
            Δa .-= K \ r
        end
        copyto!(a_old, a)
        states_old .= states

        push!(a_data,deepcopy(a)) # just for saving

        vtk_grid("./results/phase_field_results_fine_test_fineend-$step", dh) do vtk
            vtk_point_data(vtk, dh, a)
            vtk_save(vtk)
            pvd[step] = vtk
        end    

        ReactionForce.postprocess(1, step, doassemble!, reaction_forces, K, r, material, cv, dh, a, a_old, Δt, states, states_old, dim=2)
        ErrorNorm.postprocessOnlyL2(1, step, error_norms, a)
        writedlm("./results/2D_reaction_forces_reference.csv",  reaction_forces.rf_vals, ',')
        writedlm( "./results/norm_ref_pp.csv",  error_norms.error_vals_pp, ',')
        writedlm( "./results/norm_ref_u1.csv",  error_norms.error_vals_u1, ',')
        writedlm( "./results/norm_ref_u2.csv",  error_norms.error_vals_u2, ',')
    end
    vtk_save(pvd);
    
end


a_data = Vector{Float64}[]
data_generator_par = -0.058


global x0 = Vec((0.57, data_generator_par)) # initial value for the data generating process, used later on in the kalman update
solve(dh, ch, material, cvs);
jldsave("data_time_series_fine_test_VOL.jld2";a_data)



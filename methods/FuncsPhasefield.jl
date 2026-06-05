### some function definitions
module FuncsPhasefield
using LinearAlgebra
using LinearSolve
using Ferrite, FerriteMeshParser
using FerriteGmsh, Gmsh
using SparseArrays
using ForwardDiff
using IterativeSolvers
using Statistics
using Optim, NLSolversBase, LineSearches
using Random, Distributions
using KernelFunctions
using CovarianceEstimation

using ..TypesPhasefield



# Macaulay brackets
export macaulay
function macaulay(x)
    return max(0.0, x)
end
export macaulay_neg
function macaulay_neg(x)
    return max(0.0, -x)
end


# Energy splits: NoSplit, VolumetricSplit, or SpectralSplit
export split_energy
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
    ϵdev = dev(ϵ)
    Ψ⁺ = 0.5 * λ * macaulay(tr(ϵ))^2 + m.G * (ϵ⁺ ⊡ ϵ⁺)
    Ψ⁻ = 0.5 * λ * macaulay_neg(tr(ϵ))^2+ m.G * (ϵ⁻ ⊡ ϵ⁻)
    return Ψ⁺, Ψ⁻
end

export calculate_degraded_stress
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

export degradation_function
degradation_function(m::MicroMorphicElasticPhaseField, ϕ) = degradation_function(m.fracture, m, ϕ)
degradation_function(::Union{AT1_FM,AT2_FM}, ::MicroMorphicElasticPhaseField, ϕ) = (1 - ϕ)^2# + 1e-4
function degradation_function(f::QuasiBrittle_FM, m::MicroMorphicElasticPhaseField, ϕ)
    a1 = 4 * m.E * m.Gc / (π * m.l * f.ft^2)
    one_minus_phi_power_p = (1 - ϕ)^m.p
    return one_minus_phi_power_p / (one_minus_phi_power_p + a1 * ϕ * (1 + a2 * ϕ * (1 + a3 * ϕ)))
end


end # module

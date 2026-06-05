### type definitions for the phasefield Ferrite.jl code

module TypesPhasefield

# Energy splits 
export NoSplit
export SpectralSplit
export VolumetricSplit
export HybridSplit
struct NoSplit end
struct SpectralSplit end # Not implemented
struct VolumetricSplit end
struct HybridSplit end

# Fracture models
export AT1_FM
struct AT1_FM end
get_cw(::AT1_FM) = 8 / 3
export AT2_FM
struct AT2_FM end
get_cw(::AT2_FM) = 2.0

export QuasiBrittle_FM
struct QuasiBrittle_FM{T} # Not fully implemented yet
    ft::T # Tensile strength
    p::T
    a2::T
    a3::T
end

# Convexifications
export NoConvexification
struct NoConvexification end
export RateConvexification
struct RateConvexification end


export MicroMorphicElasticPhaseField

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
    E=210e3, ν=0.3, Gc=2.7, l=1.5e-2, β=200.0, #l=1.5e-2 #5.5e-2
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

export MicroMorphicElasticPhaseFieldState
struct MicroMorphicElasticPhaseFieldState{T}
    ϕ::T
    d_rate::T #
end


end # module 

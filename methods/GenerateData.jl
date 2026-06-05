module GenerateData

using Ferrite, SparseArrays
using LinearAlgebra
using Statistics
using Random, Distributions
using KernelFunctions



function mvn_sample(K)
    #Random.seed!(3)
    num_samples = 1
    v = randn(size(K)[1], num_samples)
    L = cholesky(K + 1e-12*I)
    f = L.L * v
    return f
end


function computeData(solution,sensor_locs,P,n_obs,constraint=false)
    rho = 1.0
    X = stack(sensor_locs)
    c_d = 0.00005^2*with_lengthscale(Matern52Kernel(), 0.3) 
    C_d = kernelmatrix(c_d, ColVecs(X))
    σ_e = 0.0001
    c_e = σ_e^2*with_lengthscale(Matern52Kernel(), 0.5) 
    n_sens = length(sensor_locs)
    C_e = σ_e^2* Matrix(I, n_sens, n_sens)
    y = Vector{Float64}[]
    for obs in range(start=1, step=1, stop=n_obs)
        eta = mvn_sample(C_e)
        d_samp = mvn_sample(C_d)
        println(size(P))
        println(size(solution))
        yi = rho * P'*solution + d_samp + eta
        push!(y,vec(yi))
    end

    return y
end


end # end module

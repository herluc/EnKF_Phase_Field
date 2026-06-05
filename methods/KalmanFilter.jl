module KalmanFilter

using Ferrite, SparseArrays
using LinearAlgebra
using LinearSolve
using Statistics
using Random, Distributions
using KernelFunctions
using CovarianceFunctions
using GaussianProcesses
using Distances
using DSP
using Flux
using Optim
using FiniteDifferences
include("./ObsOperator.jl")
using .ObsOperator


function computeMoments(samples)
    sample_mean = mean(samples)
    sample_cov = cov(samples)
    return sample_mean, sample_cov
end


function logLikelihood(sensor_locs,data,prediction_mean,prediction_cov,n_sens,P,hyperpars_log)
    ρ = exp(hyperpars_log[1])
 #   ρ = 1.0
    σ_d = exp(hyperpars_log[2])
    l_d = exp(hyperpars_log[3])
    P = P'
    Σ_pred = prediction_cov
    print(ρ)
    print(" ")
    print(σ_d)
    print(" ")
    print(l_d)
    print("\n")

    try
        X = stack(sensor_locs)
        c_d = σ_d^2*with_lengthscale(Matern52Kernel(), l_d)
        C_d = kernelmatrix(c_d, ColVecs(X))
        σ_e = 0.0004
        c_e = σ_e^2*with_lengthscale(Matern52Kernel(), 0.5) 
        n_sens = length(sensor_locs)
        C_e = σ_e^2* Matrix(I, n_sens, n_sens)
        sigtrans = Σ_pred
        sigtrans = 0.5*(sigtrans+sigtrans') # make sure it's symmetric
        K_y = C_d + C_e + ρ^2*sigtrans
        K_y = 0.5*(K_y+K_y')
        L = cholesky(Symmetric(K_y + 1e-10*I))
        
        log_Ky_det = 2*sum(log.(diag(L.factors)))
        loglik = 0

        print(size(data[1]))
        print(size(P))
        print(size(prediction_mean))
        res = data[1]-ρ*P*prediction_mean
        inv_prob = LinearProblem((K_y+1e-12*I), vec(res))
        linsolve = init(inv_prob)
        solve!(linsolve)
        for obs in data
            res = obs-ρ*P*prediction_mean
            linsolve.b = vec(res)
            sol = solve!(linsolve)
            inv_part = sol.u
            loglik += 0.5*(-res'*inv_part) - log_Ky_det - n_sens*log(2*pi)
        end
        print("end iteration \n")
        return -loglik
    catch
        return Inf
    end
end



function rbf_kernel(x1,x2,l)
    sqdist = pairwise(SqEuclidean(),x1',x2';dims=2)
    K = 0.01^2*exp.(-sqdist / (2 * l^2))
    return K
end

function matern52_kernel(x1,x2,l)
    sqdist = pairwise(SqEuclidean(),x1',x2';dims=2)
    r = pairwise(Euclidean(),x1',x2';dims=2)
    term1 = 1.0 .+ sqrt(5) .* r ./ l .+ (5 * r.^2) ./ (3 * l^2)
    term2 = 0.01^2*exp.(-sqrt(5) .* r ./ l)
    K = term1 .* term2

    return K
end


function matern52_kernel2D(x1,l)
    r = pairwise(Euclidean(),x1')
    term1 = 1.0 .+ sqrt(5) .* r ./ l .+ (5 * r.^2) ./ (3 * l^2)
    term2 = 0.01^2*exp.(-sqrt(5) .* r ./ l)
    K = term1 .* term2

    return K
end

function kernel_ridge_regression(k, X, y, Xstar, lambda)
    K = kernelmatrix(k, X)
    kstar = kernelmatrix(k, Xstar, X)
    return kstar * ((K + lambda * I) \ y)
end;

function kernelized_fit(kernel, x_train, y_train, x_test, lambda)
    y_pred = kernel_ridge_regression(kernel, x_train, y_train, x_test, lambda)
    return y_pred
end

function gaussianSmoothing(mean,sample,l,coords)
    coords_test = coords
    coords_test = [x[1] for x in coords_test]
    coords_train = coords[1:1:end]
    coords_full = coords[1:1:end]
    coords_full = [x[1] for x in coords_full]
    coords_train = [x[1] for x in coords_train]
    data_full = sample[1:1:end]
    data = sample[1:1:end]
    mean_values = mean[1:1:end]*0.0
    mean = mean[1:1:end]*0.0
    mean = [x[1] for x in mean]
    sig = 0.45
    l = 0.34
    K_train_train = sig^2*matern52_kernel(coords_train, coords_train, l) + 1e-8*I
    K_test_train = sig^2*matern52_kernel(coords_test, coords_train, l)
    inv_prob =  LinearProblem(K_train_train, vec(data-mean_values))
    sol = solve(inv_prob)
    inv_part = sol.u
    mean_smoothed_old = mean .+ K_test_train*inv_part  


    #Select mean and covariance function
    mZero = MeanZero()                   #Zero mean function
    #kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
    kern = Matern(5/2, log(0.45), log(0.05)) #0.45,1.2
    logObsNoise = -9                # log standard deviation of observation noise (this is optional)
    gp = GP(coords_train,data,mZero,kern,logObsNoise)       #Fit the GP
    mean_smoothed, var_smoothed = predict_y(gp,coords_full);


    return mean_smoothed_old
end


function smooth2D(mean,sample,l,coords,high_pp)
    coords_test = coords
    data = vec(sample[high_pp[1:3:end],:])
    coords_train = mapreduce(permutedims, vcat, coords[high_pp[1:3:end]])'
    coords_full = mapreduce(permutedims, vcat, coords[1:1:end])'
    sig = 0.45
    l = 0.34

    mZero = MeanZero()                   #Zero mean function
    kern = Matern(5/2, log(0.34), log(0.45)) #0.45,1.2
    logObsNoise = -9              # log standard deviation of observation noise (this is optional)
    gp = GP(coords_train,data,mZero,kern,logObsNoise)       #Fit the GP
    mean_smoothed, var_smoothed = predict_y(gp,coords_full);
    mean_smoothed = mean_smoothed #.+ mean


    return mean_smoothed
end



function statFEMupdate(prediction_mean,prediction_cov,sum_y,n_obs,P,hyperpars,sensor_locs)
    # alternative approach (not ensemble based)
    X = stack(sensor_locs)

    ρ = hyperpars[1]
    σ_d = hyperpars[2]
    l_d = hyperpars[3]
    σ_e = 0.00005
    print(ρ)
    print(σ_d)
    print(l_d)
    print("\n")

    c_d = σ_d^2*with_lengthscale(Matern52Kernel(), l_d)
    C_d = kernelmatrix(c_d, ColVecs(X))
    c_e = σ_e^2*with_lengthscale(Matern52Kernel(), 0.5) # l war 1...
   # C_e = kernelmatrix(c_e, ColVecs(X))
    n_sens = length(sensor_locs)
    C_e = σ_e^2* Matrix(I, n_sens, n_sens)

    ### STANDARD GP REGR ####
    P = P'
    inv_prob =  LinearProblem(ρ^2*n_obs*P*prediction_cov*P'+(C_d .+ C_e)+1e-12*I, vec(sum_y-ρ*n_obs*P*prediction_mean))
    sol = solve(inv_prob)
    inv_part = sol.u
    mean_shifted = prediction_mean .+ prediction_cov*P'*inv_part  

    sol_cov = (P*prediction_cov*P'+(C_d + C_e)+1e-12*I)\(P*prediction_cov)
    inv_part_cov = sol_cov
    cov_shifted = prediction_cov - prediction_cov*P'*inv_part_cov
    ###########################


    return mean_shifted, cov_shifted
end


function kalmanShiftMix(sample,prediction_mean,prediction_cov,sum_y,n_obs,P,hyperpars,sensor_locs,coords)
    # compute the shift for each sample
    X = stack(sensor_locs)

    ρ = 1.0
    σ_d = hyperpars[1]
    l_d = hyperpars[2]
    σ_e = 0.0004
    print(ρ)
    print(σ_d)
    print(l_d)
    print("\n")
    n_input = size(prediction_cov)[1]
    c_corr = 0.0004^2*with_lengthscale(Matern52Kernel(), 0.2)
    coords = [x for x in coords for _ in 1:2]
    C_corr = kernelmatrix(c_corr, ColVecs(stack(coords)))
    function preserve_diagonal(matrix)
        n = size(matrix, 1)
        result = zeros(eltype(matrix), n, n)
        for i in 1:n
            result[i, i] = matrix[i, i]
        end
        return result
    end
    varmat = preserve_diagonal(C_corr)
    prediction_cov = prediction_cov #.+ C_corr# .- varmat

    c_d = σ_d^2*with_lengthscale(Matern52Kernel(), l_d)
    C_d = kernelmatrix(c_d, ColVecs(X))
    c_e = σ_e^2*with_lengthscale(Matern52Kernel(), 0.0001) # l war 1...
   # C_e = kernelmatrix(c_e, ColVecs(X))
    n_sens = length(sensor_locs)
    C_e = σ_e^2* Matrix(I, n_sens, n_sens)

    ### STANDARD GP REGR ####
    # P = P'
    # inv_prob = LinearProblem(ρ^2*n_obs*P*prediction_cov*P'+(C_d + C_e)+1e-12*I, vec(sum_y-ρ*n_obs*P*prediction_mean))
    # sol = solve(inv_prob)
    # inv_part = sol.u
    # sample_shifted = prediction_mean .+ prediction_cov*P'*inv_part  
    ###########################
    
    ### SAMPLE UPDATE, ENSEMBLE KALMAN ###
    P = P'
    inv_prob = LinearProblem(ρ^2*n_obs*P*prediction_cov*P'+(C_d .+ C_e)+1e-10*I, vec(sum_y-ρ*n_obs*P*prediction_mean))
    sol = solve(inv_prob)
    inv_part = sol.u
    sample_shifted = prediction_mean .+ prediction_cov*P'*inv_part
    #print(prediction_cov)
    ##################################

    return sample_shifted
end


function total_variation_1d(x)
    return diff(x)
end

function tot_var_custom(sample)
    return gstv(vec(sample),80, 0.5)
end

function derivative_matrix(n, h)
    # Create an (n x n) matrix filled with zeros
    D = zeros(Float64, n, n)

    # Fill in the central difference approximation
    for i in 2:n-1
        D[i, i-1] = -1 #/ (2*h)
        D[i, i+1] = 1 #/ (2*h)
    end

    # Forward difference for the first row (first point)
    D[1, 1] = -1 #/ h
    D[1, 2] = 1 #/ h

    # Backward difference for the last row (last point)
    D[n, n-1] = -1 #/ h
    D[n, n] = 1 #/ h

    return D
end


function central_diff(f, h)
    n = length(f)
    df = zeros(n)
    for i in 2:n-1
        df[i] = (f[i+1] - f[i-1]) / (2*h)
    end
    df[1] = (f[2]-f[1] / h)
    df[n] = (f[n]-f[n-1] / h)
    return df
end


function tv_matrix_1d(n::Int)
    D = zeros(Float64, n-1, n)  # Difference operator matrix
    for i in 1:(n-1)
        D[i, i] = -1.0
        D[i, i+1] = 1.0
    end
    return D
end

function fourth_order_derivative_matrix(n, h)
    # Create an (n x n) matrix filled with zeros
    D = zeros(Float64, n, n)

    # Fill in the fourth-order central difference approximation for interior points
    for i in 3:n-2
        D[i, i-2] = 1 #/ (12*h)
        D[i, i-1] = -8 #/ (12*h)
        D[i, i+1] = 8 #/ (12*h)
        D[i, i+2] = -1 #/ (12*h)
    end

    # Use second-order differences at the boundaries
    # Forward difference for the first two rows (first two points)
    D[1, 1] = -3 #/ (2*h)
    D[1, 2] = 4 #/ (2*h)
    D[1, 3] = -1 #/ (2*h)

    D[2, 1] = -1 #/ (2*h)
    D[2, 3] = 1 #/ (2*h)

    # Backward difference for the last two rows (last two points)
    D[n, n] = 3 #/ (2*h)
    D[n, n-1] = -4 #/ (2*h)
    D[n, n-2] = 1 #/ (2*h)

    D[n-1, n-1] = 1 #/ (2*h)
    D[n-1, n-3] = -1 #/ (2*h)

    return D
end


function nystromApprox(kernel, coords, m)
    landmark_indices = round.(Int, collect(range(1,size(coords, 1), m)))
    println(landmark_indices)
    coords_landmarks = coords[landmark_indices, :]
    K_SS = kernelmatrix(kernel, vec(coords_landmarks))
    K_SR = kernelmatrix(kernel, vec(coords_landmarks), coords)

    #K_SS_inv = inv(K_SS)

    F = svd(K_SS)
    S_sqrt_inv = Diagonal(1 ./ sqrt.(F.S))
    K_SS_sqrt_inv = F.V * S_sqrt_inv * F.U'
    W = K_SR' * K_SS_sqrt_inv
    return W
end


function kalmanShift(sample,prediction_mean,prediction_cov_red,cross_cov,sum_y,n_obs,P,hyperpars,sensor_locs,coords)
    # compute the shift for each sample
    X = stack(sensor_locs) # this is for 1D!

    ρ = 1.0
    σ_d = hyperpars[2]
    l_d = hyperpars[3]
    σ_e = 0.0001
    print(ρ)
    print(σ_d)
    print(l_d)
    print("\n")

    n_input = size(prediction_cov_red)[1]

    c_loc = with_lengthscale(SqExponentialKernel(), 0.45)


    C_loc_small_sq = kernelmatrix(c_loc,ColVecs(X))
    C_loc_small_rect = kernelmatrix(c_loc,coords,ColVecs(X))

    c_d = σ_d^2*with_lengthscale(Matern52Kernel(), l_d)
    C_d = kernelmatrix(c_d, ColVecs(X))

    n_sens = length(sensor_locs)
    C_e = σ_e^2* Matrix(I, n_sens, n_sens)



    println("Start Kalman shift inverse")
    P = P'

    inv_prob = LinearProblem(ρ^2*n_obs*  C_loc_small_sq.*prediction_cov_red +(C_d .+ C_e), vec((sum_y-ρ*n_obs*P*sample))) 

    sol = solve(inv_prob)
    inv_part = sol.u

    sample_shifted = sample .+ C_loc_small_rect.*cross_cov *inv_part
    println("End Kalman shift inverse")


    return sample_shifted
end



function mod4dvar(analysis,forecast,prediction_cov,P,y,sensor_locs)
    n_sens = length(sensor_locs)
    σ_e = 0.0004
    C_e = σ_e^2* Matrix(I, n_sens, n_sens)
    P = P'
    bias = vec((y-P*analysis))
    inv_prob1 = LinearProblem((C_e), bias) 
    sol = solve(inv_prob1)
    inv_part = sol.u
    data_fit = 0.5 * bias' * inv_part 

    prior_bias = vec(analysis - forecast)
    inv_prob2 = LinearProblem((prediction_cov+1e-8*I), prior_bias) 
    sol2 = solve(inv_prob2)
    inv_part2 = sol2.u

    prior_diff = 0.5 * prior_bias' * inv_part2

    loss = data_fit + prior_diff
    println(loss)
    return loss
end

function gradmod4dvar(analysis,forecast,prediction_cov,P,y,sensor_locs)
    n_sens = length(sensor_locs)
    σ_e = 0.0004
    C_e = σ_e^2* Matrix(I, n_sens, n_sens)
    P = P'
    grad = (prediction_cov+1e-8*I) \ vec(analysis - forecast) - P'*(C_e \ vec((y-P*analysis)))
end




function optim4dvar(sample,prediction_cov,sum_y,n_obs,P,sensor_locs)
    y = sum_y./n_obs
    a_new = similar(sample)
    opt = Adam(1e-3)


    function loss4dvar(a)
        loss = mod4dvar(a,sample,prediction_cov,P,y,sensor_locs)
        return loss
    end

    function grad4dvar(a)
        gr = gradmod4dvar(a,sample,prediction_cov,P,y,sensor_locs)
        return gr
    end


    #  gs = Flux.gradient(a_new -> mod4dvar(a_new,sample,prediction_cov,P,y,sensor_locs),a_new)

    for i in 1:100
        #grads = Flux.gradient(x -> loss(x, y), b)
        grads = Flux.gradient(a_new -> loss4dvar(a_new), a_new)
        grad_an = grad4dvar(a_new)
        Flux.Optimise.update!(opt, a_new, grads[1])
      #  Flux.Optimise.update!(opt, a_new, grad_an)
 
       # println(norm(grads[1]))
        println(norm(grad_an))
    end

    # result = Optim.optimize(a -> mod4dvar(a,sample,prediction_cov,P,y,sensor_locs),
    #                   a_new, LBFGS(), Optim.Options(iterations=100))

    # a_new = result.minimizer

 

    return a_new
end


end # end module

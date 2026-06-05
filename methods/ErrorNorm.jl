# error norm post processing methods
module ErrorNorm

using Ferrite
using LinearAlgebra


struct ErrorNormPars{}
    error_vals_pp::Matrix{Float64}
    error_vals_u1::Matrix{Float64}
    error_vals_u2::Matrix{Float64}
    dh
    dh_data
    ph
    ph_data
end

function ErrorNormPars(time_vector, n_samps,dh,dh_data,ph,ph_data)
    return ErrorNormPars(zeros(length(time_vector),n_samps), zeros(length(time_vector),n_samps), zeros(length(time_vector),n_samps),dh,dh_data,ph,ph_data)    
end


function postprocess(sample_idx, timestep_idx, timestep_idx_data, enp::ErrorNormPars, time_samples, a, coords)
    sample = a
    true_solution = time_samples[timestep_idx_data]
    u_points = Ferrite.get_point_values(enp.ph, enp.dh, copy(sample), :u);
    pp_points = Ferrite.get_point_values(enp.ph, enp.dh, copy(sample), :d);
    u1_points = [v[1] for v in u_points]
    u2_points = [v[2] for v in u_points]
    error_u1 = true_solution[1:2:Int(size(true_solution)[1]*(2/3))-1] .- u1_points
    error_u2 = true_solution[2:2:Int(size(true_solution)[1]*(2/3))] .- u2_points
    error_pp = true_solution[Int(size(true_solution)[1]*(2/3))+1:1:end] .- pp_points


    norm_u1 = norm(error_u1)
    norm_u2 = norm(error_u2)
    norm_pp = norm(error_pp)

    enp.error_vals_pp[timestep_idx,sample_idx] = norm_pp
    enp.error_vals_u1[timestep_idx,sample_idx] = norm_u1
    enp.error_vals_u2[timestep_idx,sample_idx] = norm_u2
end


function postprocessOnlyL2(sample_idx, timestep_idx, enp::ErrorNormPars, a)
    sample = a
    

    norm_u1 = norm(sample[1:2:Int(size(sample)[1]*(2/3))-1])
    norm_u2 = norm(sample[2:2:Int(size(sample)[1]*(2/3))])
    norm_pp = norm(sample[Int(size(sample)[1]*(2/3))+1:1:end])

    enp.error_vals_pp[timestep_idx,sample_idx] = norm_pp
    enp.error_vals_u1[timestep_idx,sample_idx] = norm_u1
    enp.error_vals_u2[timestep_idx,sample_idx] = norm_u2
end


end #module

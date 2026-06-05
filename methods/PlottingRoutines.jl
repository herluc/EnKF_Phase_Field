module PlottingRoutines

using Plots, LinearAlgebra
using Statistics

function plotCorrelationMatrix(a,name)

    corr_mat = cor(a,dims=2)
    heatmap(corr_mat, yflip=true)
    savefig(name)

end

function plotCovarianceMatrix(a,name)

    cov_mat = cov(a,dims=2)
    heatmap(cov_mat, yflip=true)
    savefig(name)

end



end # end module

# A Regularized Ensemble Kalman Filter for Stochastic Phase Field Models of Brittle Fracture

This is the source code related to our CMAME paper "A Regularized Ensemble Kalman Filter for Stochastic Phase Field Models of Brittle Fracture". 


## Usage/Examples

To run the code, first launch a julia environment in the command line with
```bash
julia
```
Then, load the provided environment
```julia
pkg> activate EnkfEnv\\
```

and run the 2D SENS data generation code with
```julia
julia> include("./main_files/phasefield_mwe.jl")
```

This will create the ground truth data set.

With
```julia
julia> include("./main_files/phasefield_sampled_mwe.jl")
```
the ensemble solver will be run. Data assimilation happens automatically at the chosen time steps and results are written as .vtu files ready to be viewed in paraview.

Running the 1D code works similarly, simply with the files provided in the 1d folder.

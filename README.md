# Diffusion-based Condition Perturbations

Reposity for my capstone project. Much of the code is adapted from ECCC's repository for PARADIS (not the model, just the data acquisition and loading). 

## TODO
1. Train operator.
2. While it is training, write script for diffusion model. (monitor progess, MAKE SURE TO USE MOST OF THE DATA -> print start, end, dataset size)
3. When operator finishes, evaluate its autoregressive predictions.

**If all goes well and the diffusion model and operator fit well.**
1. Fit full operator, same size data as used for diffusion model.
2. Setup eval pipelines to determine best solver (DDIM, DPM++, etc.) along with parameters (eta, guidance, etc.)
  1. This requires setting up various metrics and LOT OF TIME.
3. Setup traditional techniques. (SV, random/transform)
4. Determine strengths and weaknesses.

**If all goes well...**
1. Possible extensions.
  1. Action-value function for guidance parameter.
  2. Tree-based forecasting by learning a state value function associated with uncertainty.
2. See how conformal prediction could be used autoregressively.

     

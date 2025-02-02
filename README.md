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

#### Usage
#### Configurations
There are two primary configuration files, the `config.yaml` and `accelerator.yaml`, both located in the `config/` directory.
The `config.yaml` handles the training (both deterministic and diffusion) and dataset parameters while the `accelerator.yaml` handles the parameters for ditributed training.

##### Training (deterministic model)
For training the deterministic model,
```
accelerate launch --config_file config/accelerator.yaml operator_trainer.py [override_args]
```
where `[override_args]` can override the inputs in the config file.

##### Training (diffusion model)
For training the deterministic model,
```
accelerate launch --config_file config/accelerator.yaml diffusion_trainer.py [override_args]
```
where `[override_args]` can override the inputs in the config file.

#### Dataset 
*Please note, these are the same instructions as those from PARADIS, if you have already done this there is no need to do it again.*

Download the original dataset from WeatherBench 2:

```
cd scripts
bash download_dataset.sh OUTPUT_DIR
```
where OUTPUT_DIR is the destination directory and then preprocess it

```
python scripts/preprocess_weatherbench_data.py -i /path/to/ERA5/5.625deg_wb2 -o /path/to/ERA5/5.65deg
```

     
#### Acknowledgements

This project draws significant inspiration and makes entenstive use of the data pipeline, from the paper [PARADIS - yet to be released].

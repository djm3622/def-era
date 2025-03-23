# Diffusion-augmented Ensemble Forecasting

Reposity for my capstone project. Much of the code is adapted from ECCC's repository for PARADIS (not the model, just the data acquisition and loading). 

## Dependencies
Necessary python packages can be installed using **pip**:

```
pip install -r requirements.txt --break-system-packages
```

## Usage
### Configurations
There are two primary configuration files, the `config.yaml` and `accelerator.yaml`, both located in the `config/` directory.
The `config.yaml` handles the training (both deterministic and diffusion) and dataset parameters while the `accelerator.yaml` handles the parameters for ditributed training.

In the following sections, each command will allow for `[override_args]`, which overrides the inputs in the config file.


### Training
The following section details how to train each model in a distributed evironment.

##### deterministic model

For training the deterministic model:
```
accelerate launch --config_file config/accelerator.yaml deterministic_trainer.py [override_args]
```

##### diffusion model

For training the diffusion model:
```
accelerate launch --config_file config/accelerator.yaml diffusion_trainer.py [override_args]
```

### Evaluation
The following section explains how to run our built-in evaluation scripts.

#### deterministic model

For evaluating the deterministic model:
```
python deterministic_eval.py [override_args]
```

This script will write trajectory .gifs and RMSE plots for selected variables and timesteps within the config.

#### diffusion model

For evaluating a single configuration of the diffusion model:
```
python diffusion_eval.py [override_args]
```

For evaluating an array of configurations, such as `guidance, eta, solver steps`:
```
./scripts/tune.sh
```

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

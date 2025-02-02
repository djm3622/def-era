import argparse
import yaml
from types import SimpleNamespace
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--train-config', type=str, required=True,
                      help='Path to training YAML configuration file')
    parser.add_argument('--dataset-config', type=str, required=True,
                      help='Path to dataset YAML configuration file')
    parser.add_argument('--override', nargs='*', default=[],
                      help='Override config values. Format: config_type.key=value where config_type is train or dataset')
    return parser.parse_args()

def load_config(config_path, overrides=None):
    """Load and process a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides if any
    if overrides:
        for override in overrides:
            key, value = override.split('=')
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current[k]
            try:
                value = eval(value)  # For numbers, booleans, etc.
            except:
                pass  # Keep as string if eval fails
            current[keys[-1]] = value
    
    return config

def dict_to_namespace(d):
    """Convert a dictionary to a namespace recursively."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        elif isinstance(value, list):
            setattr(namespace, key, [dict_to_namespace(x) if isinstance(x, dict) else x for x in value])
        else:
            setattr(namespace, key, value)
    return namespace

def main():
    args = parse_args()
    
    # Load configurations
    train_overrides = [o[6:] for o in args.override if o.startswith('train.')]
    dataset_overrides = [o[8:] for o in args.override if o.startswith('dataset.')]
    
    train_config = load_config(args.train_config, train_overrides)
    dataset_config = load_config(args.dataset_config, dataset_overrides)
    
    # Convert to namespaces for easier access
    train_config = dict_to_namespace(train_config)
    dataset_config = dict_to_namespace(dataset_config)
    
    # Your training code here
    print(f"Training for {train_config.training.epochs} epochs")
    
    # ... rest of your training code ...

if __name__ == "__main__":
    main()
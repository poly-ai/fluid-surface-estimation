import yaml
import os
from definitions import DATA_RAW_DIR
from pathlib import Path

def load_config(config_path, default_config_path):
    with open(default_config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            print(f"Loaded default config from '{default_config_path}'")
        except yaml.YAMLError as exc:
            print(exc)

    with open(config_path, 'r') as stream:
        try:
            custom_config = yaml.safe_load(stream)
            # Update defaults with custom configurations only if custom configurations not empty
            if custom_config != None:
                config.update(custom_config)
                print(f"Loaded user config from '{config_path}'")
            else:
                print(f"Could not load user config from '{config_path}'")
        except yaml.YAMLError as exc:
            print(exc)

    dataset_paths_dict = {name:Path(os.path.join(DATA_RAW_DIR, f"wave-{name}.npy")) for name in config['datasets'] }
    return config, dataset_paths_dict
import yaml
from bunch import Bunch

def yaml2namespace(yaml_path: str) -> Bunch:
    """Load the yaml file and convert it into a namespace

    Args:
        yaml_path: path to the yaml file

    Returns:
        Namespace config for the model
    """
    with open(yaml_path, 'r') as f:
        model_config_dict = yaml.load(f, yaml.FullLoader)

    model_config = Bunch(model_config_dict)
    return model_config

import ruamel.yaml
from pathlib import Path
import argparse
import sys

def read_cli(help):
    """
    Parse the command line interface of the script, with short and long options w/o arguments.
    It relies on argparse, the built-in argument parser of Python
    :return: parsing results from argv
    """
    # Display a thorough description of the script
    _parser = argparse.ArgumentParser(description=help)

    # Get the parameter file using the option -c|--config
    _parser.add_argument("-c", "--config", dest="config_file", type=str, required=True,
                         metavar="config/config_local.yml",
                         help="Path to a config file in YAML format. Check the config/ subdirectory for examples")

    _parser.add_argument("-f", "--fold", dest="fold", type=int, required=False,
                         metavar=None, help="Fold ID")

    return _parser.parse_args()


def read_config(config_file: str):
    with open(Path(config_file), "r") as yaml_stream:

        # Safely open the config file
        # --------------------------------------------------------------------------------------------------------------
        try:
            yaml = ruamel.yaml.YAML(typ='safe', pure=True)
            _config = yaml.load(yaml_stream)

        except yaml.YAMLError as yaml_error:
            # If the parameter file is not found, the execution must be aborted
            print(f"Cannot read the parameter file: {config_file}\n{yaml_error}", 'red')
            sys.exit(1)

        return _config
from configparser import ConfigParser
import os

config_path = 'config.ini'
parser = ConfigParser()
parser.read([config_path])

input_path_dict = dict(parser.items("INPUT_PATH"))
output_path_dict = dict(parser.items("OUTPUT_PATH"))

DATA_ROOT_PATH = input_path_dict["local_data_root_path"]
# DATA_ROOT_PATH = input_path_dict["leo_data_root_path"]
# DATA_ROOT_PATH = input_path_dict["wynton_data_root_path"]
SHORT_CHAIN_FATTY_ACID_DATA_FILENAME = input_path_dict["short_chain_fatty_acid_data_filename"]

OUTPUT_PATH = output_path_dict["output_path"]
FIGURE_DIR = os.path.join(OUTPUT_PATH, "figures")

__all__ = [
    "DATA_ROOT_PATH",
    "SHORT_CHAIN_FATTY_ACID_DATA_FILENAME",
    "OUTPUT_PATH",
    "FIGURE_DIR"
]
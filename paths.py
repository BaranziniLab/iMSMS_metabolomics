from configparser import ConfigParser
import os

# config_path = 'config.ini'
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.ini'))
parser = ConfigParser()
parser.read([config_path])

input_path_dict = dict(parser.items("INPUT_PATH"))
output_path_dict = dict(parser.items("OUTPUT_PATH"))
neo4_path_dict = dict(parser.items("NEO4J"))

DATA_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), input_path_dict["local_data_root_path"]))
# DATA_ROOT_PATH = input_path_dict["leo_data_root_path"]
# DATA_ROOT_PATH = input_path_dict["wynton_data_root_path"]
SHORT_CHAIN_FATTY_ACID_DATA_FILENAME = input_path_dict["short_chain_fatty_acid_data_filename"]
GLOBAL_SERUM_DATA_FILENAME = input_path_dict["global_serum_data_filename"]
GLOBAL_STOOL_DATA_FILENAME = input_path_dict["global_stool_data_filename"]

OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), output_path_dict["output_path"]))
FIGURE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(OUTPUT_PATH, "figures")))

URI = neo4_path_dict["uri"]
SPOKE_USER = neo4_path_dict["spoke_user"]
SPOKE_PASSWORD = neo4_path_dict["spoke_password"]

__all__ = [
    "DATA_ROOT_PATH",
    "SHORT_CHAIN_FATTY_ACID_DATA_FILENAME",
    "GLOBAL_SERUM_DATA_FILENAME",
    "GLOBAL_STOOL_DATA_FILENAME",
    "OUTPUT_PATH",
    "FIGURE_DIR",
    "URI",
    "SPOKE_USER",
    "SPOKE_PASSWORD"    
]
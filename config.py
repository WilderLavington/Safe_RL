
"""
CONFIG FILE TO LOAD IN ALL MY DATA:
TAKES THE YAML FILE LOCATION AND THEN LOADS ALL MY VARIABLES AS GLOBALS
"""

# read in yaml written by the main function
import yaml
import os
import shutil
from os import path
import gym
import numpy
import pprint

# function to read meta data
def load_my_yaml(file_location):
    # something to load in yaml file
    with open(file_location) as info:
        info_dict = yaml.load(info, Loader=yaml.FullLoader)
    # return the dictionary
    return info_dict

# function to write out meta data
def write_my_yaml(my_dict, file_location):
    # write it back
    with open(file_location, 'w') as yaml_file:
        yaml_file.write(yaml.dump(my_dict, default_flow_style=False))
    # nothing to return
    return None

# somethig to create a new directory to store results
def create_new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)           # Removes all the subdirectories!
        os.makedirs(path)
    return None

# something to print
def print_dict(dct):
    pprint.pprint(dct)

from transformers import (set_seed)
import yaml
import os
from munch import Munch
import pandas as pd
import numpy as np
from typing import Dict

# Loading the configuration file into cfg
with open("config.yaml", "r") as file:
    cfg = yaml.safe_load(file)
# Converting dictionary to object
cfg = Munch(cfg)
# set some params
set_seed(cfg.params["seed"])
# MLflow setup
os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow["exp_name"]
os.environ["MLFLOW_FLATTEN_PARAMS"] = cfg.mlflow["params"]
PROJECT_NAME = cfg.mlflow["exp_name"] + cfg.mlflow["params"]


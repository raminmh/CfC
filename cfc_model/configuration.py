import json
import glob

# Load all tensorflow configurations
tf = {
    "default":{"backbone_activation": "gelu",
    "backbone_dr": 0.0,
    "forget_bias": 3.0,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 0,
    "use_lstm": False,
    "no_gate": False,
    "minimal": False}
}

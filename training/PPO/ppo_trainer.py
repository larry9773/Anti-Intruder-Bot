#!/usr/bin/env python

import os
import random
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import tqdm
import gzip
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
from omegaconf import OmegaConf
from typing import Dict, List
from omegaconf.dictconfig import DictConfig

import habitat_sim
from habitat_sim.agent import AgentConfiguration, ActionSpec, ActuationSpec
from habitat_sim.sensor import SensorSpec, SensorType, SensorSubType
from habitat import VectorEnv
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.construct_vector_env import construct_envs
from habitat_baselines.config.default import get_config
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
)
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from pg.base_pg import BasePolicyGradient
from pg.base_pg_trainer import BasePolicyGradientTrainer

sensor_settings = {
    "height": 256,
    "width": 256,
    "sensor_height": 0.15,
}

# Caméra RGB
rgb_sensor_spec = habitat_sim.CameraSensorSpec()
rgb_sensor_spec.uuid = "color_sensor"
rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor_spec.resolution = [sensor_settings["height"], sensor_settings["width"]]
rgb_sensor_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

# Capteur de profondeur
depth_sensor_spec = habitat_sim.CameraSensorSpec()
depth_sensor_spec.uuid = "depth_sensor"
depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
depth_sensor_spec.resolution = [sensor_settings["height"], sensor_settings["width"]]
depth_sensor_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

# Capteur GPS simulé (placeholder)
gps_sensor_spec = habitat_sim.CameraSensorSpec()
gps_sensor_spec.uuid = "gps_sensor"
gps_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR  # placeholder
gps_sensor_spec.resolution = [1, 1]
gps_sensor_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
gps_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

# 2D Lidar sensor
lidar2d_spec = SensorSpec()
lidar2d_spec.uuid = "lidar_2d"
lidar2d_spec.sensor_type = SensorType.DEPTH
lidar2d_spec.sensor_subtype = SensorSubType.EQUIRECTANGULAR
lidar2d_spec.resolution = [1, 360]          # 1 vertical line, 360 horizontal rays
lidar2d_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]      # mounted 15cm above ground
lidar2d_spec.hfov = 360                     # full horizontal sweep
lidar2d_spec.vfov = 0.1                     # minimal vertical field of view
lidar2d_spec.near = 0.1
lidar2d_spec.far = 10.0

sensor_specs = [rgb_sensor_spec, depth_sensor_spec, gps_sensor_spec]

# -------------------------
# Configuration de l'agent
# -------------------------
agent_settings = {
    "action_space": {
        "move_forward": 0.25,  # déplacement fixe (mètres)
        "rotate": 1.0,         # facteur de rotation (angle en degrés)
    }
}

agent_cfg = AgentConfiguration()
agent_cfg.action_space = {}
for action, value in agent_settings["action_space"].items():
    agent_cfg.action_space[action] = ActionSpec(action, ActuationSpec(amount=value))
agent_cfg.sensor_specifications = sensor_specs

# -------------------------
# Configuration du simulateur
# -------------------------
sim_settings = {
    "default_agent": 0,
    "scene_id" : "data/scene_datasets/gibson/Cantwell.glb",
    #"scene_id": "data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb",  # à modifier si nécessaire
    "enable_physics": False,
    "seed": 42,
}

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = sim_settings["scene_id"]
sim_cfg.enable_physics = sim_settings["enable_physics"]

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-4.69643, 0.15825, -2.90618])
agent.set_state(agent_state)


base_config_dict = {
    "habitat": {
        "simulator": {
            "scene_id": sim_cfg.scene_id,
            "enable_physics": sim_cfg.enable_physics,
            "seed": sim_settings["seed"],
        },
        "agent": {
            "default_agent": sim_settings["default_agent"],
            "sensor_settings": sensor_settings,
            "sensors": [
                {
                    "uuid": "color_sensor",
                    "sensor_type": "COLOR",
                    "resolution": [sensor_settings["height"], sensor_settings["width"]],
                    "position": [0.0, sensor_settings["sensor_height"], 0.0],
                    "sensor_subtype": "PINHOLE",
                },
                {
                    "uuid": "depth_sensor",
                    "sensor_type": "DEPTH",
                    "resolution": [sensor_settings["height"], sensor_settings["width"]],
                    "position": [0.0, sensor_settings["sensor_height"], 0.0],
                    "sensor_subtype": "PINHOLE",
                },
                {
                    "uuid": "gps_sensor",
                    "sensor_type": "COLOR",  # placeholder
                    "resolution": [1, 1],
                    "position": [0.0, sensor_settings["sensor_height"], 0.0],
                    "sensor_subtype": "PINHOLE",
                },
                {
                    "uuid": "semantic",
                    "sensor_type": "SEMANTIC",  # placeholder
                    "position": [0.0, sensor_settings["sensor_height"], 0.0],
                }
            ],
        },
    }
}

base_config = OmegaConf.create(base_config_dict)

def compress_json_file(input_path: str, output_path: str) -> None:
    """
    Lit le fichier JSON situé à input_path et en crée une version compressée gzip à output_path.
    """
    if os.path.exists(output_path):
        print(f"Le fichier compressé {output_path} existe déjà.")
        return

    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Fichier compressé créé : {output_path}")

input_file = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
output_file = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json.gz"
compress_json_file(input_file, output_file)

def build_PPO_config():
    # Change for REINFORCE
    config = get_config("objectnav/ddppo_objectnav.yaml")
    OmegaConf.set_readonly(config, False)
    config.habitat_baselines.load_resume_state_config = False
    config.habitat_baselines.launch_eval_afterwards = False
    config.habitat_baselines.checkpoint_folder = "data/PPO_checkpoints"
    config.habitat_baselines.tensorboard_dir = "tb/PPO"
    config.habitat_baselines.num_updates = -1
    config.habitat_baselines.num_environments = 2
    config.habitat_baselines.verbose = False
    config.habitat_baselines.num_checkpoints = -1
    config.habitat_baselines.checkpoint_interval = 1000000
    config.habitat_baselines.total_num_steps = 150 * 1000
    config.habitat_baselines.force_blind_policy = True
    config.habitat.dataset.data_path="data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json.gz"
    OmegaConf.set_readonly(config, True)

    return config

config = build_PPO_config()
final_config = OmegaConf.merge(base_config, config)

# Set randomness
random.seed(config.habitat.seed)
np.random.seed(config.habitat.seed)
torch.manual_seed(config.habitat.seed)
if (
    config.habitat_baselines.force_torch_single_threaded
    and torch.cuda.is_available()
):
    torch.set_num_threads(1)

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# Build the trainer and start training


if __name__ == "__main__":
    trainer = PPOTrainer(final_config)
    trainer.train()

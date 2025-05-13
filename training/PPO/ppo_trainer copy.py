#! python3
import multiprocessing as mp

import os
import random
import json
import gzip
import numpy as np
import torch
from omegaconf import OmegaConf

from habitat.core.vector_env import ThreadedVectorEnv as _VectorEnv, _make_env_fn as base_make_env_fn
from habitat_baselines.common.construct_vector_env import construct_envs
from helpers import register_ball as _register_ball


import habitat_sim
from habitat_sim import attributes, SensorType, SensorSubType, CameraSensorSpec
from habitat_sim.agent import AgentConfiguration
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat.core.env import RLEnv

from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)

# chemin et params globaux
_ball_file   = os.path.join(os.getcwd(), "data", "objects", "sim_ball_target.glb")
_scale       = 0.5
_semantic_id = 0

def make_env_with_ball(config, dataset=None, rank=0):
    # 1) création « standard »
    env = base_make_env_fn(config, dataset, rank)
    # 2) on y injecte la balle
    _register_ball(env, _ball_file, _scale, _semantic_id, idx=rank)
    return env


def _get_sim(env):
    return env.unwrapped._sim


# --- Utility to create a mini ObjectNav dataset for Cantwell, with sensors ---
def create_cantwell_dataset(out_dir, scene_path, semantic_id, n_episodes=20):
    print(">>>>>>>> CREATING CANTWELL DATASET")

    os.makedirs(out_dir, exist_ok=True)
    dataset_file = os.path.join(out_dir, "cantwell.json.gz")
    # Always recreate the dataset to avoid stale or invalid files
    # Remove existing file if present
    if os.path.exists(dataset_file):
        os.remove(dataset_file)
        

    # Simulator configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    # Build agent configuration with the requested sensors
    agent_cfg = AgentConfiguration()
    # sensor settings as per user
    sensor_settings = {"height": 544, "width": 720, "sensor_height": 1.5}
    sensor_specs = []
    # color sensor
    color_spec = CameraSensorSpec()
    color_spec.uuid = "color_sensor"
    color_spec.sensor_type = SensorType.COLOR
    color_spec.resolution = [sensor_settings["height"], sensor_settings["width"]]
    color_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
    color_spec.sensor_subtype = SensorSubType.PINHOLE
    sensor_specs.append(color_spec)
    # depth sensor
    depth_spec = CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = SensorType.DEPTH
    depth_spec.resolution = [sensor_settings["height"], sensor_settings["width"]]
    depth_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
    depth_spec.sensor_subtype = SensorSubType.PINHOLE
    sensor_specs.append(depth_spec)
    # gps placeholder sensor
    gps_spec = CameraSensorSpec()
    gps_spec.uuid = "gps_sensor"
    gps_spec.sensor_type = SensorType.COLOR  # placeholder
    gps_spec.resolution = [1, 1]
    gps_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
    gps_spec.sensor_subtype = SensorSubType.PINHOLE
    sensor_specs.append(gps_spec)
    # semantic sensor
    sem_spec = CameraSensorSpec()
    sem_spec.uuid = "semantic"
    sem_spec.sensor_type = SensorType.SEMANTIC
    sem_spec.resolution = [sensor_settings["height"], sensor_settings["width"]]
    sem_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
    sem_spec.sensor_subtype = SensorSubType.PINHOLE
    sensor_specs.append(sem_spec)
    agent_cfg.sensor_specifications = sensor_specs

    # Initialize simulator with one agent
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    episodes = []
    for i in range(n_episodes):
        start = sim.pathfinder.get_random_navigable_point()
        goal = sim.pathfinder.get_random_navigable_point()
        episodes.append({
            "episode_id": str(i),
            "scene_id": os.path.basename(scene_path),
            "start_position": [float(start[0]), float(start[1]), float(start[2])],
            "start_rotation": {
                                "@type": "quat",
                                "w": 1.0,
                                "x": 0.0,
                                "y": 0.0,
                                "z": 0.0
                                },

            # **ici** on indique la catégorie de l’objet à chercher
            "object_category": "red_ball",

            "goals": [
                {
                    "position": [float(goal[0]), float(goal[1]), float(goal[2])],
                    "radius": 0.1,

                    # **ici** on nomme exactement object_category
                    "object_category": "red_ball",
                }
            ],
        })
    sim.close()


    # Construire les mappings
    unique_cats = sorted({ep["object_category"] for ep in episodes})
    category_to_task_category_id = {cat: i for i, cat in enumerate(unique_cats)}
    category_to_mp3d_category_id = {cat: semantic_id for cat in unique_cats}

    # Injecter object_id, object_name et object_name_id dans chaque goal
    # … après avoir construit category_to_* et before writing JSON …
    for ep in episodes:
        cat = ep["object_category"]
        task_id = category_to_task_category_id[cat]
        mp3d_id = category_to_mp3d_category_id[cat]
        for g in ep["goals"]:
            g["object_id"]       = task_id
            g["object_name"]     = cat
            g["object_name_id"]  = mp3d_id
            g["view_points"]     = []   # <-- obligatoire !

    data = {
        "episodes": episodes,
        "category_to_task_category_id": category_to_task_category_id,
        "category_to_mp3d_category_id": category_to_mp3d_category_id,
    }

    # Écrire le JSON compressé
    with gzip.open(dataset_file, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return dataset_file


# --- 1) Trainer qui charge et enregistre le template “ball” ---
class TrainerWithBall(PPOTrainer):
    def __init__(self, config):
        print(">>>>>> INITIATING TRAINING")
        super().__init__(config)
        # on stocke ici nos paramètres une seule fois
        self._ball_file   = os.path.join(os.getcwd(),
                                         "data", "objects", "sim_ball_target.glb")
        self._scale       = 0.5
        self._semantic_id = 0


    
    def _init_envs(self, config=None, is_eval: bool = False):
        """
        Construire manuellement le VectorEnv en injectant la balle
        dès la création de chaque env, en passant dataset=None.
        """
        cfg = config or self.config
        start_method = mp.get_start_method()

        # Wrapper : signature (config, dataset, rank)
        def make_env_with_ball(cfg_inner, dataset, rank):
            # 1) création standard (dataset=None → base_make_env_fn charge depuis cfg)
            env = construct_envs(
                config=cfg,
                workers_ignore_signals=False,
                enforce_scenes_greater_eq_environments=True,
            )
            
            # 2) injection de la balle
            register_ball(
                env,
                self._ball_file,
                self._scale,
                self._semantic_id,
                idx=rank,
            )
            return env

        num_envs = cfg.habitat_baselines.num_environments
        # ► IMPORTANT : dataset=None, pas un string ! Sinon Env() reçoit un str et plante
        env_args = [(cfg, None, rank) for rank in range(num_envs)]

        self.envs = _VectorEnv(
            make_env_with_ball,
            env_args,
            auto_reset_done=True,
            multiprocessing_start_method=start_method,
        )
        print(f"[TrainerWithBall] Created {num_envs} envs with ball injection")


    def _init_train(self, resume_state=None):
        # On force resume_state à None pour ne jamais charger de state_dict
        print(">>>>> LAUNCHING TRAINING")
        print(">>>>> RESUME_STATE : ", resume_state)
        super()._init_train(resume_state)


# --- 2) Construction de la config PPO + Gibson/Cantwell + dataset ---
def build_cfg(dataset_path: str):
    """
    Charge la config ddppo_objectnav.yaml et la patch pour ObjectNav+Cantwell.
    """
    cfg = get_config("objectnav/ddppo_objectnav.yaml")
    OmegaConf.set_readonly(cfg, False)

    # Dataset uniquement Cantwell
    cfg.habitat.dataset.type = "ObjectNav-v1"
    cfg.habitat.dataset.data_path = dataset_path
    cfg.habitat.dataset.split = "train"
    cfg.habitat.dataset.scenes_dir = os.path.join(
        "data", "scene_datasets", "gibson"
    )

    # Logging et performance
    cfg.habitat_baselines.verbose = True
    cfg.habitat_baselines.log_interval = 1

    # Nombre d'envs et de processus ≥ 1
    cfg.habitat_baselines.num_environments = 1
    cfg.habitat_baselines.num_processes = 1

     # 1) Activer au moins un worker
    cfg.habitat_baselines.num_processes = 1

    # 2) Forcer le chargement de la scène Cantwell
    cfg.habitat.simulator.scene = os.path.join(
        os.getcwd(), "data", "scene_datasets", "gibson", "Cantwell.glb"
    )
    cfg.habitat.simulator.scene_dataset = "default"

    OmegaConf.set_readonly(cfg, True)
    return cfg


# --- 3) Main ---
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # créer le mini-dataset Cantwell
    repo_root = os.getcwd()
    dataset_dir = os.path.join(repo_root, "data", "datasets", "objectnav")
    cantwell_glb = os.path.join(repo_root, "data", "scene_datasets", "gibson", "Cantwell.glb")
    dataset_file = create_cantwell_dataset(
        dataset_dir, cantwell_glb, semantic_id=0, n_episodes=20
    )
    print(">>>>>>>> CANTWELL DATASET CREATED")

    # construire la config et lancer
    cfg = build_cfg(dataset_file)
    print(">>>>>>>> BUILT CONFIG")
    trainer = TrainerWithBall(cfg)
   # 4) Debug : init + reset mono-proc
    try:
        trainer._init_envs(cfg, is_eval=False)
        obs = trainer.envs.reset()
        print("✅ trainer.envs.reset() a réussi, obs keys :", obs.keys())
    except Exception:
        import traceback; traceback.print_exc(); raise
    trainer.train()

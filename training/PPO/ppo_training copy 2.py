#!/usr/bin/env python
"""
ppo_habitat_target_nav.py
-------------------------

Ce script démontre l'entraînement d'un agent RL avec PPO dans Habitat pour une tâche
de navigation vers un objectif. La configuration des capteurs inclut :
  • Une caméra RGB
  • Un capteur de profondeur
  • Un capteur GPS simulé (placeholder)
  • Un capteur Lidar simulé (placeholder)

L'agent exécute des actions constituées d'un mouvement avant fixe et d'une commande de rotation continue
(en degrés). La politique fusionne les entrées visuelles et non-visuelles et est entraînée via une boucle PPO vectorisée.

Note : Cet exemple définit des fonctions d'aide pour les opérations sur les quaternions à la place des implémentations de Habitat.
"""

import os
import random
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
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
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer


class HabitatSimulator():
    def __init__(self):
        self.simSettings = {
            "default_agent": 0,
            "scene_id": "data/scene_datasets/gibson/Cantwell.glb",  # à modifier si nécessaire
            "enable_physics": False,
            "seed": 42,
        }
        self.cfg = self.setConfig(self.simSettings)
        self.sim = habitat_sim.Simulator(self.cfg)
        self.agent = self.sim.initialize_agent(self.simSettings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([-4.69643, 0.15825, -2.90618])
        self.agent.set_state(agent_state)

    def setConfig(self, sim_settings):
        agent_cfg = AgentConfiguration()
        sim_cfg = habitat_sim.SimulatorConfiguration()
        
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

        # Depth sensor
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [sensor_settings["height"], sensor_settings["width"]]
        depth_sensor_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # Define an IMU Sensor Spec
        imu_spec = SensorSpec()
        imu_spec.uuid = "imu_sensor"
        imu_spec.sensor_type = SensorType.IMU
        imu_spec.sensor_subtype = SensorSubType.NONE
        imu_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]  # IMU mounted 1m above ground

        # 2D Lidar sensor
        lidar2d_spec = SensorSpec()
        lidar2d_spec.uuid = "lidar_2d"
        lidar2d_spec.sensor_type = SensorType.DEPTH
        lidar2d_spec.sensor_subtype = SensorSubType.EQUIRECTANGULAR
        lidar2d_spec.resolution = [1, 360]          # 1 vertical line, 360 horizontal rays
        lidar2d_spec.position = [0.0, 0.15, 0.0]      # mounted 15cm above ground
        lidar2d_spec.hfov = 360                     # full horizontal sweep
        lidar2d_spec.vfov = 0.1                     # minimal vertical field of view
        lidar2d_spec.near = 0.1
        lidar2d_spec.far = 10.0

        sensor_specs = [rgb_sensor_spec, depth_sensor_spec, imu_spec, lidar2d_spec]

        # -------------------------
        # agent configuration
        # -------------------------
        agent_settings = {
            "action_space": {
                "move_forward": 0.25,  # déplacement fixe (mètres)
                "rotate": 1.0,         # facteur de rotation (angle en degrés)
            }
        }
    

        # Create an agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.action_space = {
            k: habitat_sim.agent.ActionSpec(
                k, habitat_sim.agent.ActuationSpec(amount=v)
            ) for k, v in agent_settings["action_space"].items()
        }
        agent_cfg.sensor_specifications = sensor_specs

        # -------------------------
        # Simulator configuration
        # -------------------------

        sim_cfg.scene_id = sim_settings["scene_id"]
        sim_cfg.enable_physics = sim_settings["enable_physics"]
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
class PPO_Trainer():
    def __init__(self):
        self.sample_config = self.build_pretrained_config("data/datasets/pointnav/gibson/v1/val/val_cantwell.json.gz")
        self.sample_env = self.build_env(config=self.sample_config)
        self.sample_device = torch.device("cpu")
        self.sample_actor_critic, self.ample_agent = self.build_agent(config=self.sample_config, env=self.sample_env, device=self.sample_device)
        self.sample_test_recurrent_hidden_states, self.sample_prev_actions, self.sample_not_done_masks = self.build_variables(
            config=self.sample_config, 
            actor_critic=self.sample_actor_critic, 
            device=self.sample_device
            )

    def build_pretrained_config(self, data_path: str):
        config = get_config("pointnav/ppo_pointnav.yaml")  # Extract config from yaml
        # Change for evaluation
        OmegaConf.set_readonly(config, False)
        config.habitat_baselines.eval_ckpt_path_dir="./data/checkpoints/gibson.pth"  # Choose checkpoint
        config.habitat_baselines.num_updates = -1
        config.habitat_baselines.num_environments = 1
        config.habitat_baselines.verbose = False
        config.habitat.dataset.data_path = data_path
        OmegaConf.set_readonly(config, True)

        return config
    
    def build_env(self, config: DictConfig, multiprocess=True):
        if not multiprocess:
            import os
            os.environ['HABITAT_ENV_DEBUG'] = '1'
        return construct_envs(
            config=config,
            workers_ignore_signals=False,
            enforce_scenes_greater_eq_environments=True,
        )

    # A function to load the pretrained agent
    def build_agent(self, config: DictConfig, env: VectorEnv, device: torch.device):
        ppo_cfg = config.habitat_baselines.rl.ppo  # Extract config for PPO

        policy = baseline_registry.get_policy(
            config.habitat_baselines.rl.policy.name
        )
        observation_space =  env.observation_spaces[0]
        policy_action_space = env.action_spaces[0]
        orig_policy_action_space = env.orig_action_spaces[0]
        
        actor_critic = policy.from_config(  # Build the actor-critic
            config,
            observation_space,
            policy_action_space,
            orig_action_space=orig_policy_action_space,
        )
        actor_critic.to(device)

        agent = PPO.from_config(  # Build the PPO agent
            actor_critic=actor_critic,
            config=ppo_cfg,
        )

        ckpt_dict = torch.load(config.habitat_baselines.eval_ckpt_path_dir, map_location="cpu")  # Load the checkpoint
        agent.load_state_dict(ckpt_dict["state_dict"])

        actor_critic.eval()
        agent.eval()

        return actor_critic, agent
    
    # A function to build auxiliary variables for the policy
    def build_variables(self, config: DictConfig, actor_critic: nn.Module, device: torch.device):
        test_recurrent_hidden_states = torch.zeros(  # Hidden recurrent state
            config.habitat_baselines.num_environments,
            actor_critic.num_recurrent_layers, 
            config.habitat_baselines.rl.ppo.hidden_size, 
            device=device
        )
        prev_actions = torch.zeros(  # Previous action
            config.habitat_baselines.num_environments,
            1,
            device=device,
            dtype=torch.long,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            1,
            device=device,
            dtype=torch.bool,
        )

        return test_recurrent_hidden_states, prev_actions, not_done_masks
    
    # A function to map observations to actions using the pretained agent
    def step_agent(self, actor_critic: nn.Module, batch: Dict[str, torch.Tensor], test_recurrent_hidden_states: torch.Tensor, prev_actions: torch.Tensor, not_done_masks: torch.Tensor):
        # TODO: Please enter your code here to replace ...
        # HINT: You can refer to the resnet policy at https://github.com/facebookresearch/habitat-lab/blob/v0.2.3/habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py
        with inference_mode():
            (
                _,
                actions,
                _,
                test_recurrent_hidden_states,
            ) = actor_critic.act(
                batch,                      
                test_recurrent_hidden_states,
                prev_actions,                
                not_done_masks,
                deterministic=False,
            )

            prev_actions.copy_(actions)  # type: ignore

        return actions, test_recurrent_hidden_states


    # A function to excecute simulation within the environment
    def step_env(self, env: VectorEnv, actions: torch.Tensor):
        # TODO: Please enter your code here to replace ...
        # HINT: You can refer to the doc of VectorEnv at https://aihabitat.org/docs/habitat-lab/habitat.core.vector_env.VectorEnv.html
        step_data = [a.item() for a in actions]
        outputs = env.step(step_data)

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]
        
        return observations, rewards_l, dones, infos


    # A utility function to post-process the results
    def post_process(self, observations: List[Dict], dones: List[bool], device: torch.device):
        batch = batch_obs(  # type: ignore
            observations,
            device=device,
        )

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=device,
        )

        return batch, not_done_masks


    # A utility function to collect rewards in an episode
    def collect_rewards(self, rewards_l: List[float], ep_rewards: List[float]):
        rewards = torch.tensor(
            rewards_l, dtype=torch.float, device="cpu"
        ).unsqueeze(1)
        ep_rewards.append(rewards[0].item())


    # A utility function to collect frames in an episode
    def collect_frames(self, batch: Dict[str, torch.Tensor], infos: List[Dict], not_done_masks: torch.Tensor, ep_frames: List[np.ndarray]):
        frame = observations_to_image(
            {k: v[0] for k, v in batch.items()}, infos[0]
        )
        if not not_done_masks[0].item():
            # The last frame corresponds to the first frame of the next episode
            # but the info is correct. So we use a black frame
            frame = observations_to_image(
                {k: v[0] * 0.0 for k, v in batch.items()}, infos[0]
            )
        frame = overlay_frame(frame, infos[0])
        ep_frames.append(frame)


    # A utility function to generate video from collected frames
    def generate_video(self, ep_frames: List[np.ndarray], video_name: str, output_dir: str = "output_video", fps: int = 10):
        output_dir = output_dir.lower().replace(" ", "_")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        video_name = video_name.lower().replace(" ", "_")
        
        writer = imageio.get_writer(
            os.path.join(output_dir, video_name),
            fps=fps
        )

        frames_iter = tqdm.tqdm(ep_frames)
        for fm in frames_iter:
            writer.append_data(fm)
        writer.close()

    def loop_env(self, config: DictConfig, scene_name: str, agent_name: str, add_drift: bool = False, drift_func = None):
        # Set the randomness requried later
        random.seed(config.habitat.seed)
        np.random.seed(config.habitat.seed)
        torch.manual_seed(config.habitat.seed)

        # Choose device for cuda or cpu
        device = torch.device("cuda", config.habitat_baselines.torch_gpu_id) if torch.cuda.is_available() else torch.device("cpu")

        # Build the vectorized environment
        env = self.build_env(config=config)

        # Build the actor-critic and agent from a checkpoint
        actor_critic, agent = self.build_agent(config=config, env=env, device=device)

        # Build auxiliary variables
        test_recurrent_hidden_states, prev_actions, not_done_masks = self.build_variables(
            config=config, 
            actor_critic=actor_critic, 
            device=device
        )

        observations = env.reset()  # Reset the environment, e.g., move the agent back to its start location
        batch = batch_obs(observations, device=device)

        rng = np.random.default_rng()
        sign, scale = None, None

        n_step = 0
        episodes_stats = {}
        ep_rewards, ep_frames = [], []
        while len(episodes_stats) < env.number_of_episodes[0]:
            ep_name = env.current_episodes()[0].episode_id
            actions, test_recurrent_hidden_states = self.step_agent(  # Map observations to actions
                actor_critic=actor_critic,
                batch=batch,
                test_recurrent_hidden_states=test_recurrent_hidden_states,
                prev_actions=prev_actions,
                not_done_masks=not_done_masks,
            )

            observations, rewards_l, dones, infos = self.step_env(  # One step forward of simulation in the environment
                env=env,
                actions=actions,
            )
            if add_drift:
                if sign is None:
                    sign = rng.choice((-1, 1))  # Direction for drift
                    scale = 0.05  # Scale the drift w.r.t. steps
                drift_func(
                    observations=observations,
                    n_step=n_step,
                    sign=sign,
                    scale=scale
                )

            batch, not_done_masks = self.post_process(  # Post-process the results
                observations=observations,
                dones=dones,
                device=device,
            )

            self.collect_rewards(  # Collect rewards
                rewards_l=rewards_l,
                ep_rewards=ep_rewards,
            )

            self.collect_frames(  # Collect frames
                batch=batch,
                infos=infos,
                not_done_masks=not_done_masks,
                ep_frames=ep_frames
            )

            n_step += 1
            if not not_done_masks[0].item():  # Episode ended
                last_infos = infos.copy()

                episodes_stats[ep_name] = {
                    'success': last_infos[0]['success'],
                    'spl': last_infos[0]['spl'],
                    'return': sum(ep_rewards),
                }

                # Generate video
                self.generate_video(
                    ep_frames=ep_frames,
                    video_name=f"ep={ep_name}_success={last_infos[0]['success']}_spl={last_infos[0]['spl']}.mp4",
                    output_dir=f"output_video/{scene_name}/{agent_name}"
                )

                # Clean
                n_step = 0
                ep_rewards, ep_frames = [], []

                # Build auxiliary variables
                test_recurrent_hidden_states, prev_actions, not_done_masks = self.build_variables(
                    config=config, 
                    actor_critic=actor_critic, 
                    device=device
                )

        success_l, spl_l, return_l = [], [], []
        for ep_stat in episodes_stats.values():
            success_l.append(ep_stat['success'])
            spl_l.append(ep_stat['spl'])
            return_l.append(ep_stat['return'])

        avg_success = sum(success_l) / len(success_l)
        avg_spl = sum(spl_l) / len(spl_l)
        avg_return = sum(return_l) / len(return_l)

        print(f"In {scene_name}, {agent_name}")
        print(f"\t Average success rate: {avg_success}")
        print(f"\t Average SPL: {avg_spl}")
        print(f"\t Average return: {avg_return}")

        return {
            'success': avg_success,
            'spl': avg_spl,
            'return': avg_return
        }

if __name__ == "__main__":
    pass

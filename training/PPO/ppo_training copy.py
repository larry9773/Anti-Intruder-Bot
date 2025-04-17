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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

import habitat_sim
from habitat_sim.agent import AgentConfiguration, ActionSpec, ActuationSpec

# -------------------------
# Fonctions d'aide pour les quaternions
# -------------------------
def quat_from_angle_axis(angle, axis):
    """
    Calcule un quaternion à partir d'une rotation de 'angle' radians autour de 'axis'.
    Renvoie un tableau numpy [w, x, y, z].
    """
    axis = np.array(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("Axe de rotation de norme nulle.")
    axis = axis / axis_norm
    half_angle = angle / 2.0
    w = math.cos(half_angle)
    xyz = axis * math.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]])

def quat_multiply(q, r):
    """
    Multiplie deux quaternions q et r.
    Chaque quaternion est un tableau numpy [w, x, y, z].
    Renvoie leur produit sous forme d'un tableau numpy [w, x, y, z].
    """
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# -------------------------
# Configuration des capteurs et du simulateur
# -------------------------

sensor_settings = {
    "height": 256,
    "width": 256,
    "sensor_height": 1.5,
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

# Capteur Lidar simulé (placeholder)
lidar_sensor_spec = habitat_sim.CameraSensorSpec()
lidar_sensor_spec.uuid = "lidar_sensor"
lidar_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
# La résolution est attendue sous forme d'un tableau numpy de forme (2,1) : ici 360 lectures (une par degré)
lidar_sensor_spec.resolution = np.array([[360], [1]], dtype=np.int32)
lidar_sensor_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
lidar_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

sensor_specs = [rgb_sensor_spec, depth_sensor_spec, gps_sensor_spec, lidar_sensor_spec]

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
    "scene_id": "data/scene_datasets/gibson/Cantwell.glb",  # à modifier si nécessaire
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

# -------------------------
# Fonctions d'aide pour le prétraitement et la rotation
# -------------------------
def preprocess_observation(obs, device):
    """
    Prétraite les observations du simulateur.
    Convertit l'image RGB en tenseur et construit un vecteur de capteurs en concaténant
    les lectures du GPS (on en prend 2) et du Lidar (on en prend 360).
    """
    # Traitement de l'image RGB
    rgb_img = obs.get("color_sensor", None)
    if rgb_img is None:
        rgb_img = np.zeros((sensor_settings["height"], sensor_settings["width"], 3), dtype=np.uint8)
    else:
        if rgb_img.shape[-1] >= 4:
            rgb_img = rgb_img[..., :3]
    rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Traitement du GPS : on extrait les 2 premières valeurs
    gps_img = obs.get("gps_sensor", None)
    if gps_img is None:
        gps = np.zeros((2,), dtype=np.float32)
    else:
        gps = np.array(gps_img).flatten()[:2]
    # Traitement du Lidar : on prend les 360 premières valeurs
    lidar_img = obs.get("lidar_sensor", None)
    if lidar_img is None:
        lidar = np.zeros((360,), dtype=np.float32)
    else:
        lidar = np.array(lidar_img).flatten()[:360]
    sensor_vec = np.concatenate([gps, lidar])
    sensor_tensor = torch.from_numpy(sensor_vec).unsqueeze(0).float().to(device)
    return rgb_tensor, sensor_tensor

def rotate_agent(agent, angle_deg):
    """
    Met à jour l'orientation de l'agent en effectuant une rotation autour de l'axe vertical (Y).
    La rotation actuelle (un quaternion) est convertie en numpy, multipliée par le quaternion de rotation,
    puis l'état de l'agent est mis à jour.
    """
    state = agent.get_state()
    current_q = np.array([state.rotation.w, state.rotation.x, state.rotation.y, state.rotation.z])
    yaw = np.deg2rad(angle_deg)
    rot_quat = quat_from_angle_axis(yaw, [0, 1, 0])
    new_q = quat_multiply(current_q, rot_quat)
    try:
        from habitat_sim.math import Quaternion
        state.rotation = Quaternion(new_q[0], new_q[1], new_q[2], new_q[3])
    except ImportError:
        state.rotation = new_q
    agent.set_state(state)

# -------------------------
# Réseau de politique multimodale
# -------------------------
class MultiModalPolicy(nn.Module):
    def __init__(self):
        super(MultiModalPolicy, self).__init__()
        # Branche visuelle : CNN simple pour l'image RGB.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        # Pour une image 256x256 :
        # Après conv1 (kernel=5, stride=2) : 126x126, puis conv2 (kernel=3, stride=2) : 62x62.
        cnn_out_dim = 32 * 62 * 62

        # Branche capteurs : traitement du GPS (2) et du Lidar (360).
        self.sensor_mlp = nn.Sequential(
            nn.Linear(2 + 360, 64),
            nn.ReLU()
        )
        # Fusion des caractéristiques.
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 128),
            nn.ReLU()
        )
        # Têtes de sortie.
        self.move_head = nn.Linear(128, 1)           # Décision de mouvement avant.
        self.rotate_mean = nn.Linear(128, 1)         # Moyenne de la distribution de rotation (degrés).
        self.rotate_logstd = nn.Parameter(torch.zeros(1))  # Log-écart-type.
        self.value_head = nn.Linear(128, 1)          # Estimation de la valeur.
        
    def forward(self, rgb, sensor_vec):
        visual_feat = self.cnn(rgb)
        sensor_feat = self.sensor_mlp(sensor_vec)
        combined = torch.cat([visual_feat, sensor_feat], dim=1)
        features = self.fc(combined)
        move_logit = self.move_head(features)
        rotation_mean = self.rotate_mean(features)
        rotation_std = self.rotate_logstd.exp().expand_as(rotation_mean)
        value = self.value_head(features)
        return move_logit, (rotation_mean, rotation_std), value

# -------------------------
# Sélection d'action et stockage pour PPO
# -------------------------
def select_action(policy, observation, device):
    """
    Sélectionne une action à partir d'une observation via la politique.
    Le mouvement avant est déterminé par un seuil (0.5) sur sigmoid(move_logit).
    L'angle de rotation est échantillonné depuis une distribution normale.
    Renvoie : action (dictionnaire), log-probabilité totale et estimation de valeur.
    """
    rgb_tensor, sensor_tensor = preprocess_observation(observation, device)
    move_logit, (rot_mean, rot_std), value = policy(rgb_tensor, sensor_tensor)
    move_prob = torch.sigmoid(move_logit)
    move_action = 1 if move_prob.item() > 0.5 else 0

    rotation_dist = distributions.Normal(rot_mean, rot_std)
    rotation = rotation_dist.sample()
    log_prob_rot = rotation_dist.log_prob(rotation)

    bernoulli = distributions.Bernoulli(probs=move_prob)
    log_prob_move = bernoulli.log_prob(torch.tensor(move_action, dtype=torch.float, device=device))
    total_log_prob = log_prob_move + log_prob_rot

    action = {"move_forward": move_action, "rotate": rotation.item()}
    return action, total_log_prob, value, rgb_tensor, sensor_tensor

# -------------------------
# Étape environnementale
# -------------------------
def step_env(sim, agent, action):
    """
    Exécute une étape dans l'environnement.
    L'agent avance via sim.step (même si ici c'est une simple commande "move_forward")
    puis la rotation est appliquée.
    Renvoie les observations, la récompense (dummy) et le flag de fin d'épisode.
    """
    observations = sim.step("move_forward")
    rotate_agent(agent, action["rotate"])
    observations = sim.get_sensor_observations()
    reward = -0.01
    done = False
    return observations, reward, done

# -------------------------
# Boucle d'entraînement PPO vectorisée
# -------------------------
def train_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MultiModalPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    
    num_episodes = 100  # À ajuster
    max_steps = 200
    gamma = 0.99
    clip_param = 0.2
    ppo_epochs = 4  # Nombre de passes PPO sur le même batch

    print("Début de l'entraînement PPO...")
    for episode in range(num_episodes):
        # Stockage des observations, actions, log-prob, et valeurs pour le batch
        obs_rgb_list = []
        obs_sensor_list = []
        old_log_probs = []
        old_values = []
        old_move_actions = []
        old_rotations = []
        rewards = []

        sim.reset()
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([-4.69643, 0.15825, -2.90618])
        agent.set_state(agent_state)
        observation = sim.get_sensor_observations()

        for t in range(max_steps):
            action, log_prob, value, rgb_tensor, sensor_tensor = select_action(policy, observation, device)
            obs_rgb_list.append(rgb_tensor)
            obs_sensor_list.append(sensor_tensor)
            old_log_probs.append(log_prob)
            old_values.append(value)
            old_move_actions.append(action["move_forward"])
            old_rotations.append(action["rotate"])
            
            observation, reward, done = step_env(sim, agent, action)
            rewards.append(reward)
            if done:
                break

        # Calcul des retours (returns) discountés
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)

        old_values_tensor = torch.cat(old_values).detach()
        advantages = returns - old_values_tensor

        old_log_probs_tensor = torch.cat(old_log_probs).detach()
        # Stockage des actions sous forme de tenseurs
        old_move_actions_tensor = torch.tensor(old_move_actions, dtype=torch.float32, device=device).unsqueeze(1)
        old_rotations_tensor = torch.tensor(old_rotations, dtype=torch.float32, device=device).unsqueeze(1)

        # Création du batch d'observations
        batch_rgb = torch.cat(obs_rgb_list, dim=0)
        batch_sensor = torch.cat(obs_sensor_list, dim=0)

        # Mise à jour PPO sur plusieurs passes (re-évaluation de la politique sur le batch)
        for epoch in range(ppo_epochs):
            new_move_logit, (new_rot_mean, new_rot_std), new_values = policy(batch_rgb, batch_sensor)
            new_move_prob = torch.sigmoid(new_move_logit)
            bernoulli = distributions.Bernoulli(probs=new_move_prob)
            new_log_prob_move = bernoulli.log_prob(old_move_actions_tensor)
            
            rotation_dist = distributions.Normal(new_rot_mean, new_rot_std)
            new_log_prob_rot = rotation_dist.log_prob(old_rotations_tensor)
            new_log_probs = new_log_prob_move + new_log_prob_rot

            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(new_values, returns)
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_reward = sum(rewards)
        print(f"Episode {episode+1}/{num_episodes}, Récompense Totale: {total_reward:.2f}")

if __name__ == "__main__":
    train_ppo()

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from ball_finder_env import BallFinderEnv
from stable_baselines3.common.monitor import Monitor

# --- 1) Prépare l'env d'évaluation ---
def make_eval_env(max_steps=500, render_mode=None):
    # render_mode peut être "human" si tu veux voir en direct, sinon None
    return BallFinderEnv(max_steps=max_steps, render_mode=render_mode)

eval_env = make_eval_env(render_mode=None)
eval_env = Monitor(eval_env)
# --- 2) Recharge ton modèle entraîné ---
# Par défaut, SB3 enregistre le .zip : ici on suppose "models/ppo_ball_finder.zip"
model_path = os.path.join("models", "ppo_ball_finder.zip")
model = PPO.load(model_path, env=eval_env)  # on fixe l'env pour .predict()

# --- 3) Évaluation automatique avec SB3 -------------
# renvoie deux floats : moyenne et écart‐type des récompenses sur n_episodes
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=20,    # nombre d'épisodes d'éval
    deterministic=True,    # pas d'exploration
    return_episode_rewards=False
)
print(f"Mean reward over 20 episodes: {mean_reward:.2f} ± {std_reward:.2f}")

# --- 4) Facultatif : récupérer success_rate également ------------- 
# Si tu veux mesurer success_rate (is_success dans info), tu peux écrire une petite boucle :

n_episodes = 20
successes = 0
for ep in range(n_episodes):
    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
    successes += int(info.get("is_success", False))
    print(f"Episode {ep+1}: reward={total_reward:.1f}  success={info.get('is_success', False)}")
print(f"Success rate: {successes}/{n_episodes} = {successes/n_episodes:.2%}")

# --- 5) Facultatif : enregistrer une vidéo -------------
# Si tu souhaites une vidéo de l’agent en évaluation, utilise RecordVideo
from gym.wrappers import RecordVideo

video_env = RecordVideo(
    make_eval_env(render_mode="rgb_array"),
    video_folder="videos/",
    name_prefix="eval_ep"
)
obs, _ = video_env.reset()
for _ in range(eval_env.max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = video_env.step(action)
    if terminated or truncated:
        break
video_env.close()
print("Video saved in videos/")

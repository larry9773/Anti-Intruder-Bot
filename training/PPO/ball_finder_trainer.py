import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from ball_finder_env import BallFinderEnv
from stable_baselines3.common.monitor import Monitor
from success_rate_callback import SuccessRateCallback
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env():
    env = BallFinderEnv(max_steps=500, render_mode="rgb_array")
    return Monitor(env)

checkpoint_cb = CheckpointCallback(
    save_freq=5_000,        # tous les 10 000 pas
    save_path="./checkpoints",
    name_prefix="ppo"
)

vec_env = DummyVecEnv([make_env for _ in range(4)])

model = PPO(
    policy=MultiInputActorCriticPolicy,
    env=vec_env,
    verbose=1,
    tensorboard_log="./tb_sb3/",
    n_steps=256,
    batch_size=32,
    n_epochs=4,
    vf_coef=1.0,
    ent_coef=0.05,
    stats_window_size=20
)

#model.learn(total_timesteps=50_000, callback=SuccessRateCallback())
model.learn(total_timesteps=50_000, callback=checkpoint_cb)
model.save("models/ppo_ball_finder")
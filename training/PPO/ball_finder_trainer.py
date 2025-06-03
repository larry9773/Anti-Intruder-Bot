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
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
import multiprocessing as mp
import torch


total_time_steps = 1_200_000
n_env = 8
max_steps = 1000

def make_env():
    env = BallFinderEnv(max_steps=max_steps, render_mode=None, total_time_steps=total_time_steps, n_env=n_env, end_dist=15.0)
    return Monitor(env)


if __name__ == "__main__":
    #mp.set_start_method("fork", force=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=5_000,        
        save_path=str(
            os.path.join(os.getcwd(), "ball_nav/checkpoints")
        ),
        #"./checkpoints",
        name_prefix="Multi_scenes_phase_1",
    )

    initial_lr = 3e-4
    def lr_schedule(progress_remaining: float) -> float:
        #return initial_lr * (0.5 + 0.5 * progress_remaining)
        t_norm = 1.0 - progress_remaining  # fraction parcourue
        if t_norm < 0.75:
            return 3e-4
        else:
            # fraction dans la seconde moitié
            frac = (t_norm - 0.5) / 0.5  # ∈ [0, 1]
            return 3e-4 - frac * (3e-4 - 7.5e-5)

    # Clip range schedule: floor at 0.05
    def clip_schedule(progress_remaining: float) -> float:
        return max(0.05, 0.1 * progress_remaining)


    initial_ent = 0.02 
    def ent_coef_schedule(progress_remaining: float) -> float:
        # de 0.02 → 0.001
        return 0.001 + 0.019 * progress_remaining

    #vec_env = DummyVecEnv([make_env for _ in range(4)])
    vec_env = SubprocVecEnv([make_env for _ in range(n_env)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    #initial_lr = 3e-4
    #lr_schedule = lambda progress_remaining: progress_remaining * initial_lr
    
    model = PPO(
        policy=MultiInputActorCriticPolicy,
        env=vec_env,
        verbose=1,
        tensorboard_log="./tb_sb3/ball_finder_multi_scenes_phase_1/",
        n_steps=512,           
        batch_size=256,        
        n_epochs=3,            
        learning_rate=lr_schedule,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=clip_schedule,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        stats_window_size=100,
        device="cuda",
    )

    '''if torch.cuda.device_count() > 1:
    # Affichez le nombre de GPU pour debug :
        print(f"[Info] Using {torch.cuda.device_count()} GPUs for the model")
        # Enrobez la policy dans DataParallel
        model.policy = torch.nn.DataParallel(model.policy)
    else:
        print("[Warning] Only one GPU detected, DataParallel disabled")'''

    model.learn(total_timesteps=total_time_steps, callback=checkpoint_cb)

    model.save(str(
            os.path.join(os.getcwd(), "ball_nav/models/ball_finder_multi_scenes_phase_1")
        ))
    
    # Supposons que 'model' est ton PPO déjà entraîné
    policy = model.policy

    # 1) Récupère l'état du réseau sous forme de dict de tensors
    state_dict = policy.state_dict()

    # 2) Sauvegarde dans un fichier .pt ou .pth
    torch.save(state_dict, str(
            os.path.join(os.getcwd(), "ball_nav/models/ball_finder_multi_scenes_phase_1.pth")
        ))
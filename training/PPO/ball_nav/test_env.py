#!/usr/bin/env python3
# test_env.py

import os, sys
# 1) Assurez-vous que votre projet (le dossier courant) est bien dans PYTHONPATH
sys.path.insert(0, os.getcwd())

# 2) Imports corrects
from habitat_baselines.config.default import get_config
from habitat_baselines.common.construct_vector_env import construct_envs

if __name__ == "__main__":
    # 3) Charge la config que vous utilisez pour le training
    cfg = get_config("configs/objectnav/ddppo_ball.yaml", [])

    # 4) Construis un vector env en mono-processus
    #    (c'est exactement ce que fait PPOTrainer._init_envs())
    envs = construct_envs(
        config=cfg.habitat.dataset,  # le nœud dataset de votre config
        opts=[]                      # pas d'overrides supplémentaires
    )

    try:
        print("→ Tentative de reset de l'env …")
        obs = envs.reset()  # ici on devrait voir l'erreur Python complète
        print("✅ reset OK, clés d'observation :", list(obs[0].keys()))
    finally:
        envs.close()

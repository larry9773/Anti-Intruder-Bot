#!/usr/bin/env python3
# scripts/generate_ball_episodes.py

import os
import json
import gzip
import random
import numpy as np
import habitat_sim
from habitat_sim import Simulator, Configuration
from habitat_sim.simulator import SimulatorConfiguration, AgentConfiguration


def generate_episodes(
    scene_glb: str,
    output_path: str,
    n_episodes: int = 1000,
    n_samples: int = 500,
):
    """
    Génère un dataset ObjectNav-v1 au format attendu par Habitat-Baselines,
    avec mappings category_to_task_category_id et category_to_mp3d_category_id.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1) Initialise le simulateur pour charger la scène (navmesh déjà présent)
    sim_cfg = SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    # AgentConfiguration minimal (pas de capteurs nécessaires pour sampling)
    agent_cfg = AgentConfiguration()
    sim = Simulator(Configuration(sim_cfg, [agent_cfg]))

    # 2) Échantillonner un pool de points navigables
    nav_pts = [sim.pathfinder.get_random_navigable_point() for _ in range(n_samples)]

    # 3) Créer les épisodes bruts
    episodes = []
    for i in range(n_episodes):
        start = random.choice(nav_pts)
        goal = random.choice(nav_pts)
        while np.allclose(goal, start):
            goal = random.choice(nav_pts)
        episodes.append({
            "episode_id": str(i),
            "scene_id": os.path.basename(scene_glb),
            "start_position": [float(x) for x in start],
            "start_rotation": {"@type": "quat", "w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "goals": [{
                "position":        [float(x) for x in goal],
                "object_category": "ball",
                # placeholders, seront remplis ensuite
                "radius":          0.1,
                "object_id":       None,
                "object_name":     None,
                "object_name_id":  None,
                "view_points":     []
            }],
        })

    sim.close()

    # 4) Construire les mappings de catégories
    unique_cats = sorted({
        g["object_category"]
        for ep in episodes
        for g in ep["goals"]
    })
    category_to_task_category_id = {cat: idx for idx, cat in enumerate(unique_cats)}
    category_to_mp3d_category_id = {cat: idx for idx, cat in enumerate(unique_cats)}

    # 5) Remplir object_id, object_name, object_name_id
    for ep in episodes:
        for g in ep["goals"]:
            cat = g["object_category"]
            g["object_id"] = category_to_task_category_id[cat]
            g["object_name"] = cat
            g["object_name_id"] = category_to_mp3d_category_id[cat]
            # view_points reste []

    # 6) Assembler le dict final
    dataset_dict = {
        "episodes": episodes,
        "category_to_task_category_id": category_to_task_category_id,
        "category_to_mp3d_category_id": category_to_mp3d_category_id,
    }

    # 7) Sauvegarde compressée
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2)

    print(f"✅ Généré {len(episodes)} épisodes → {output_path}")


if __name__ == "__main__":
    SCENE = "data/scene_datasets/cantwell/Cantwell.glb"
    OUT_JSON = "data/scene_datasets/cantwell/ball_episodes.json.gz"
    generate_episodes(
        scene_glb=SCENE,
        output_path=OUT_JSON,
        n_episodes=1000,
        n_samples=500
    )

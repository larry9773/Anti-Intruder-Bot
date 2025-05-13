import seaborn as sns
import matplotlib
import random
import os

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import math

import habitat_sim
from habitat_sim.utils import common as ut
from habitat_sim.utils import viz_utils as vut
import magnum as mn
from habitat_sim.geo import Ray

class HabitatSimBuilder():
    def __init__(self):
        self.sim = None
        self.obj_attr_mgr = None
        self.prim_attr_mgr = None
        self.stage_attr_mgr = None
        self.sel_file_obj_handle = None
        self.sel_prim_obj_handle = None
        self.sel_asset_handle = None
        self.ball_id = None
        self.object_ids = []

    ######### Configuration Utility functions ######

    def make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.enable_physics = settings["enable_physics"]
        
        sensor_settings = {
            "height": 1080, "width": 1920,  # Spatial resolution of observations
            "sensor_height": 0.15,  # Height of sensors in meters, relative to the agent
        }

        # Create a RGB sensor configuration
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [sensor_settings["height"], sensor_settings["width"]]
        rgb_sensor_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # Create a depth sensor configuration
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [sensor_settings["height"], sensor_settings["width"]]
        depth_sensor_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # Create a Semantic sensor configuration
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [sensor_settings["height"], sensor_settings["width"]]
        semantic_sensor_spec.position = [0.0, sensor_settings["sensor_height"], 0.0]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        sensor_specs = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
        print(">>>>>> Added all sensors in array sensor_spec")
        # Here you can specify the amount of displacement in a forward action and the turn angle

        agent_settings = {
            "action_space": {
                "move_forward": 0.25, "move_backward": 0.25,  # Distance to cover in a move action in meters
                "turn_left": 30.0, "turn_right": 30,  # Angles to cover in a turn action in degrees
            }
        }

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        print(">>>>>> Creating agent cfg")
        agent_cfg.action_space = {
            k: habitat_sim.agent.ActionSpec(
                k, habitat_sim.agent.ActuationSpec(amount=v)
            ) for k, v in agent_settings["action_space"].items()
        }
        agent_cfg.sensor_specifications = sensor_specs
        print(">>>>>> Added sensor_spec to agent_cfg")

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def make_default_settings(self):
        settings = {
            "width": 720,  # Spatial resolution of the observations
            "height": 544,
            "scene": "./data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",  # Scene path
            "default_agent": 0,
            "sensor_height": 1.5,  # Height of sensors in meters
            "sensor_pitch": -np.pi / 8.0,  # sensor pitch (x rotation in rads)
            "color_sensor_1st_person": True,  # RGB sensor
            "color_sensor_3rd_person": False,  # RGB sensor 3rd person
            "depth_sensor_1st_person": False,  # Depth sensor
            "semantic_sensor_1st_person": True,  # Semantic sensor
            "seed": 1,
            "enable_physics": True,  # enable dynamics simulation
        }
        return settings

    def make_simulator_from_settings(self, sim_settings):


        # UI-populated handles used in various cells.  Need to initialize to valid
        # value in case IPyWidgets are not available.
        # Holds the user's desired file-based object template handle
        self.sel_file_obj_handle = self.obj_attr_mgr.get_file_template_handles()[0]
        # Holds the user's desired primitive-based object template handle
        self.sel_prim_obj_handle = self.obj_attr_mgr.get_synth_template_handles()[0]
        # Holds the user's desired primitive asset template handle
        self.sel_asset_handle = self.prim_attr_mgr.get_template_handles()[0]

    ############ Simulation Utility functions ############

    def remove_all_objects(self):
        for obj_id in self.object_ids:
            self.sim.remove_object(obj_id)
        self.object_ids.clear()


    def simulate(self, sim, dt=1.0, get_frames=True):
        # simulate dt seconds at 60Hz to the nearest fixed timestep
        print("Simulating " + str(dt) + " world seconds.")
        observations = []
        start_time = sim.get_world_time()
        while sim.get_world_time() < start_time + dt:
            sim.step_physics(1.0 / 60.0)
            if get_frames:
                observations.append(sim.get_sensor_observations())
        return observations

    # Set an object transform relative to the agent state
    def set_object_state_from_agent(
        self,
        sim,
        ball_id,
        offset=np.array([0, 2.0, -1.5]),
        orientation=mn.Quaternion(((0, 0, 0), 1)),
    ):
         # récupère l’objet instancié
        rigid_mgr = self.sim.get_rigid_object_manager()
        ball_obj = rigid_mgr.get_object_by_id(ball_id)   # ou add_object_by_template_id renvoie déjà l’objet
        
        # calcule la position dans le monde
        agent_tf    = self.sim.agents[0].scene_node.transformation_matrix()
        world_pos   = agent_tf.transform_point(offset)

        ball_obj.translation = world_pos
        ball_obj.rotation    = orientation

    # A utility function for displaying observations
    def display_obs(self, rgb_obs: np.ndarray, depth_obs: np.ndarray, sem_obs: np.ndarray):
        img_arr, title_arr = [], []
        
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
        img_arr.append(rgb_img)
        title_arr.append("rgb")
        
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        img_arr.append(depth_img)
        title_arr.append("depth")


        # Semantic : on fait notre propre palette
        # sem_obs : H×W d'entiers (0,1,…)
        H, W = sem_obs.shape
        sem_color = np.zeros((H, W, 4), dtype=np.uint8)

        # palette simple : 0 → transparent (background), 1 → rouge semi-opaque
        sem_color[sem_obs == 0] = [0,   0,   0,   0]    # background transparent
        sem_color[sem_obs == 1] = [255, 0,   0, 127]    # balle en rouge à 50% d'alpha

        sem_img = Image.fromarray(sem_color, mode="RGBA")
        
        # --- 2) Affichage 1×3 ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax in axes:
            ax.axis("off")

        axes[0].set_title("RGB")
        axes[0].imshow(rgb_img)

        axes[1].set_title("Depth")
        axes[1].imshow(depth_img, cmap="gray")

        axes[2].set_title("Semantic")
        axes[2].imshow(sem_img)

        plt.show(block=False)

    def get_360_lidar_scan(self,
                           sim,
                      num_rays: int = 360,
                      max_range: float = 12.0,
                      height: float = 0.15):
        # 1) On récupère le node et sa rotation complète (Quaternion)
        node = sim.agents[0].scene_node
        q    = node.rotation
        mat  = node.absolute_transformation()

        # 2) Origine du lidar
        p      = mat.translation
        origin = mn.Vector3(p.x, p.y + height, p.z)

        # 3) Angles locaux entre -π et +π
        angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
        dists  = np.empty(num_rays, dtype=np.float32)

        for i, θ in enumerate(angles):
            # 4) direction LOCALE 0→avant = (0,0,-1), tournée de θ autour de Y
            local_dir = mn.Vector3(np.sin(θ), 0.0, -np.cos(θ))

            # 5) on applique la rotation complète du robot
            #    c'est équivalent à q * local_dir * q^{-1}
            world_dir = q.transform_vector(local_dir)
            world_dir.y = 0.0
            world_dir = world_dir.normalized()

            # 6) cast
            ray    = Ray(origin, world_dir)
            result = sim.cast_ray(ray)
            if result.has_hits() and len(result.hits) > 0:
                distance = result.hits[0].ray_distance
            else:
                distance = max_range
            dists[i] = min(distance, max_range)

        return angles, dists
    

    def euler_from_quaternion(self, q: mn.Quaternion) -> float:
        x, y, z = q.vector     # mn.Quaternion stocke (x,y,z) dans q.vector
        w       = q.scalar     # et le scalaire dans q.scalar
        siny =  2 * (w*y + x*z)
        cosy =  1 - 2 * (y*y + z*z)
        return math.atan2(siny, cosy)
    
    def get_agent_position(self) :
        t = self.sim.agents[0].scene_node.absolute_transformation().translation
        return np.array([t.x, t.y, t.z], dtype=np.float32)
    
    def get_agent_to_ball_dist(self):
        t = self.ball_id.translation
        return np.linalg.norm(self.get_agent_position() - np.array([t.x, t.y, t.z], dtype=np.float32))
    
    def get_random_navigable_point(self):
        pf = self.sim.pathfinder
        return pf.get_random_navigable_point()
    

    def is_ball_detected(self, sem_obs: np.ndarray, min_pixel_count: int = 1) -> bool:
        """
        Renvoie True si la semantic observation contient au moins `min_pixel_count`
        pixels avec l’ID 1 (la balle).
        """
        # sem_obs est de shape (H, W) avec des entiers 0,1,...
        count = int((sem_obs == 1).sum())
        return count >= min_pixel_count
    
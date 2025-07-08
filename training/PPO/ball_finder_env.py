import gymnasium as gym
import numpy as np
from gymnasium import spaces
from habitat_sim_builder import HabitatSimBuilder
import habitat_sim
import os
from gym.wrappers import RecordVideo
import cv2
from habitat_sim.nav import ShortestPath
import magnum as mn
from habitat_sim.utils import viz_utils as vut
import glob
import random

class BallFinderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, max_steps = 1000, render_mode=None, n_env=8, total_time_steps=200_000, end_dist=10.0):
        super().__init__()
        self.habitat_sim = HabitatSimBuilder()
        self.max_steps = max_steps
        self.render_mode = render_mode

        scene_csv_path = os.path.join(
            os.getcwd(), "hm3d_scene_areas_sorted.csv"
        )

        with open(scene_csv_path, "r") as f:
            # 1) On lit et on jette la ligne d'en‐tête
            header = next(f)  

            # 2) Ensuite on ne garde que la partie "Scene" (avant la virgule) de chaque ligne
            self.sorted_scenes = [
                line.strip().split(",")[0]
                for line in f
                if line.strip()
            ]

        # RGB
        img_shape = (256, 256, 3)
        rgb_space = spaces.Box(0, 255, shape=img_shape, dtype=np.uint8)
        
        # Lidar
        max_range = 12
        min_range = 0.15
        n_lidar_beams = 360

        lidar_space = spaces.Box(
            low=min_range, high=max_range,
            shape=(n_lidar_beams,), dtype=np.float32
        )

        # IMU
        imu_low  = np.array([-50.0, -50.0, -50.0], dtype=np.float32)
        imu_high = np.array([ 50.0,  50.0,  50.0], dtype=np.float32)
        imu_space = spaces.Box(low=imu_low, high=imu_high, dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "rgb"  : rgb_space,
            "lidar": lidar_space,
            "imu"  : imu_space,
        })

        self.action_space = spaces.Discrete(4)
        
        self._step_count = 0
        self._last_dist = None
        self._ball_id  = None
        self.success_dist = 0.3

        self.train_scenes_path = os.path.join(os.getcwd(), "hm3d/train")

        self.all_scene_glbs = glob.glob(
            os.path.join(self.train_scenes_path, "*", "*.glb")
        )
        if len(self.all_scene_glbs) == 0:
            raise RuntimeError(f"No .glb file found in {self.train_scenes_path}.")

        
        self.distance_to_goal_at_start = 0
        self.collision_count = 0
        self.d_max = 100
        self.k     = 0.05
        self.gamma = 0.995

        self.total_episodes = (total_time_steps/max_steps)/n_env
        self.frac_scene = 0.0

        self.curr_start = 2.0
        self.curr_end   = end_dist
        self.curr_max = self.curr_start
        self.curr_eps  = self.total_episodes
        self.episode_count = 0

        self.alpha_scene = 4.0  # Exponent for the scene selection curve
        self.beta = 2.0   # Exponent for the distance selection curve
        self.start_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.habitat_sim.sim = None




    def step(self, action):
        #print(">>>>>>>>>>>>>>>>>>>>>> STEP ENV")
        # Perform action
        action_mapping = {
            0: "move_forward",
            1: "move_backward",
            2: "turn_left",
            3: "turn_right"
        }
        #print(">>>>>>>>>>>>>>>>>>>>>> MAPPED ACTIONS")
        
        action_str = action_mapping[int(action)]
        self.habitat_sim.sim.step(action_str)
        #print(">>>>>>>>>>>>>>>>>>>>>> STEP SIMULATOR")
        self.habitat_sim.sim.step_physics(1.0/60.0)
        #print(">>>>>>>>>>>>>>>>>>>>>> PEFORMED ACTION")

        # New observation
        observation = self.habitat_sim.sim.get_sensor_observations()
        rgb_obs   = observation["color_sensor"]
        rgb_obs = rgb_obs[..., :3]
        _, lidar_obs = self.habitat_sim.get_360_lidar_scan(self.habitat_sim.sim)
        imu_obs = self.habitat_sim.get_agent_position() - self.start_pos
        sem_obs = observation["semantic_sensor"]
        #print(">>>>>>>>>>>>>>>>>>>>>>  RETRVIED OBSERVATIONS")

        obs_dict = {"rgb": rgb_obs, "lidar": lidar_obs, "imu": imu_obs}
        truncated = False
        done = False
        success = False

        
        dist = self.compute_shortest_distance_to_goal(self.habitat_sim.sim, 
                                                      self.habitat_sim.sim.agents[0].scene_node.absolute_transformation().translation, 
                                                      self.habitat_sim.ball_id.translation)
        if dist == float('inf'):
            dist = self.d_max
            truncated = True

        reward = self.compute_reward(self._last_dist, dist, sem_obs)
        self._last_dist = dist

        
        if dist < self.success_dist :
            reward += 50.0
            done = True
            success = True
        
        self._step_count += 1
        if self._step_count >= self.max_steps:
            self._step_count = 0
            truncated = True
            success = False
        
        
        terminated = done
        #print(f"[DEBUG] step_count={self._step_count}  dist={dist:.3f}  success={success}")

        info = {
            "is_success" : success
        }

        
        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.episode_count +=1

        self.frac_scene = pow(self.episode_count / self.curr_eps, self.alpha_scene)  # fraction [0,1]
        self.frac_scene = min(max(self.frac_scene, 0.0), 1.0)

        max_index = int(self.frac_scene * (len(self.sorted_scenes) - 1))
        chosen_index = random.randint(0, max_index)

        matching = [path for path in self.all_scene_glbs if os.path.basename(path) == self.sorted_scenes[chosen_index]]
        if len(matching) == 0:
            raise RuntimeError(f"No matching scene found for {self.sorted_scenes[chosen_index]} in the dataset.")

        chosen_scene = matching[0]

        print(f"[DEBUG] Using scene: {chosen_scene}")

        sim_settings = self.habitat_sim.make_default_settings()
        '''sim_settings["scene"] = str(
            os.path.join(os.getcwd(), "ball_nav/data/scene_datasets/cantwell/Cantwell.glb")
        ) '''
        '''sim_settings["scene"] = str(
            os.path.join(os.getcwd(), "ball_nav/data/scene_datasets/edgemere/Edgemere.glb")
        )'''

        sim_settings["scene"] = chosen_scene
        #"data/scene_datasets/cantwell/Cantwell.glb"
        sim_settings["sensor_pitch"] = 0

        cfg = self.habitat_sim.make_cfg(sim_settings)

        if self.habitat_sim.sim is not None:
            self.habitat_sim.sim.close()
        # initialize the simulator
        self.habitat_sim.sim = habitat_sim.Simulator(cfg)

        # Add the ball to the scene
        # Managers of various Attributes templates
        self.habitat_sim.obj_attr_mgr = self.habitat_sim.sim.get_object_template_manager()
        self.habitat_sim.prim_attr_mgr = self.habitat_sim.sim.get_asset_template_manager()

        ball_template = habitat_sim.attributes.ObjectAttributes()
        ball_template.render_asset_handle = str(
            os.path.join(os.getcwd(), "ball_nav/data/objects/ball/ball.glb")
        )
        ball_template.scale = np.array([1.0, 1.0, 1.0])

        # set the default semantic id for this object template
        ball_template.semantic_id = 1  # @param{type:"integer"}
        ball_template_id = self.habitat_sim.obj_attr_mgr.register_template(ball_template, "ball")

        rigid_mgr = self.habitat_sim.sim.get_rigid_object_manager()
        self.habitat_sim.ball_id = rigid_mgr.add_object_by_template_id(ball_template_id)
        self.habitat_sim.object_ids.append(self.habitat_sim.ball_id)
        #print(">>>>>>>>>>>>>>>>>>>>>> RESET ENV")

        '''frac = min(self.episode_count / self.curr_eps, 1.0)
        curr_max = self.curr_start + (self.curr_end - self.curr_start) * frac'''

        # Dynamically adjust the maximum distance based on the current episode count,
        # gradually increasing it from `curr_start` to `curr_end` as episodes progress.
        
        #self.curr_max = self.curr_start + (self.curr_end - self.curr_start) * min(frac_scene, 1.0)

        #frac_dist = frac_scene ** 2
        #d_max = self.curr_start + (self.curr_end - self.curr_start) * frac_dist
        
        '''d_min = 2
        d_max = 4'''

        '''if self.frac_scene < 0.5:
            d_min, d_max = 1.5, 2.5
            self.max_steps = 500
        else:
            d_min, d_max = 2.0, 4.0
            self.max_steps = 1000'''
        
        d_min = 1.5
        d_max = 2.5
        self.max_steps = 500
        

        self.habitat_sim.sim.reset()

        # Set agent and ball position
        sp = ShortestPath()
        pf = self.habitat_sim.sim.pathfinder
        is_navigable = False
        agent_start = None
        ball_pos = None
        while is_navigable == False :
            agent_start = pf.get_random_navigable_point()
            ball_pos   = pf.get_random_navigable_point()
            sp.requested_start = mn.Vector3(agent_start)
            sp.requested_end = mn.Vector3(ball_pos)
            #is_navigable = pf.find_path(sp) and sp.geodesic_distance > self.curr_start and sp.geodesic_distance < d_max
            is_navigable = pf.find_path(sp) and sp.geodesic_distance > d_min and sp.geodesic_distance < d_max

        # Set agent state
        agent = self.habitat_sim.sim.agents[0]  # Get our default agent
        agent_state = habitat_sim.AgentState()
        agent_state.position = mn.Vector3(agent_start)  # Position in world coordinate
        agent.set_state(agent_state)

        self.start_pos = self.habitat_sim.get_agent_position()


        self.habitat_sim.ball_id.translation = mn.Vector3(ball_pos)       # setter on ManagedBulletRigidObject
        self.habitat_sim.sim.step_physics(1.0 / 60.0)

        observation = self.habitat_sim.sim.get_sensor_observations()
        rgb_obs = observation["color_sensor"]
        rgb_obs = rgb_obs[..., :3]
        position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        _, lidar_sensor = self.habitat_sim.get_360_lidar_scan(self.habitat_sim.sim)

        obs_dict = {
            "rgb" : rgb_obs,
            "lidar" : lidar_sensor,
            "imu" : position
        }

        info = {
            
        }
        self._step_count = 0
        self._last_dist = self.compute_shortest_distance_to_goal(self.habitat_sim.sim, agent_state.position, self.habitat_sim.ball_id.translation)
        self.distance_to_goal_at_start = self._last_dist
        self.collision_count = 0
        return obs_dict, info

    def render(self):
        #print(">>>>>>>>>>>>>>>>>>>>>> RENDERING ENV")
        # 1) récupère l’image RGB courante depuis le simulator
        obs = self.habitat_sim.sim.get_sensor_observations()
        img = obs["color_sensor"]  # np.ndarray H×W×3, dtype uint8

        # 2) selon le render_mode
        if self.render_mode == "rgb_array":
            # on renvoie simplement l'image
            return img

        elif self.render_mode == "human":
            # on affiche la fenêtre (nécessite un DISPLAY ou Xvfb)
            cv2.imshow("BallFinderEnv", img)
            cv2.waitKey(1)
            return None

        else:
            # pas de rendu pour les autres modes
            return None

    def close(self):
        self.habitat_sim.sim.close()

    
    def compute_shortest_distance_to_goal(self,
                                sim, 
                                start_vec: mn.Vector3, 
                                end_vec: mn.Vector3) -> float:
        """
        Calcule la distance géodésique (plus court chemin sur le navmesh)
        entre start_vec et end_vec, tous deux des magnum.Vector3.
        Retourne float('inf') si aucun chemin n'existe.
        """
        sp = ShortestPath()
        sp.requested_start = start_vec
        sp.requested_end   = end_vec

        success = sim.pathfinder.find_path(sp)
        '''if not success or not np.isfinite(sp.geodesic_distance):
        # on remplace inf ou un path introuvable par une distance max raisonnable
            return self.d_max'''
        return sp.geodesic_distance if success else float('inf')
    
    def compute_reward(self, 
                        old_distance: float,
                    new_distance: float,
                    sem_obs: np.ndarray,
                    time_step_penalty: float = -0.015):
            
            time_step_penalty = -0.005
            collision_penalty = -0.2
            detection_factor = 10.0
        
            '''if self.frac_scene < 0.5:
                time_step_penalty = -0.005
                collision_penalty = -0.2
                detection_factor = 10.0
            else:
                time_step_penalty = -0.015
                collision_penalty = -1.0
                detection_factor = 5.0'''

            collision = self.habitat_sim.sim.get_physics_num_active_contact_points() > 0

            detection_reward =(self.habitat_sim.number_of_pixels_ball_in_camera(sem_obs) / (256*256))

            reward = 0.0

            dist_cap = min(new_distance, self.d_max)
            old_cap  = min(old_distance, self.d_max)

            distance_reward = max(0.0, old_cap - dist_cap)

            phi_old   = self.k * (self.d_max - old_distance)
            phi_new   = self.k * (self.d_max - new_distance)
            shaping_r = self.gamma * phi_new - phi_old

            # 3) Pénalité collision
            if collision:
                #self.collision_count += 1
                #collision_penalty = -self.collision_count
                reward += collision_penalty
            
            reward += shaping_r * 0.5
            reward += distance_reward * 2.0
            reward += detection_reward * detection_factor
            reward += time_step_penalty
            
            #reward = detection_reward *5.0 + collision_penalty + time_penalty + shaping_r * 0.5 + distance_reward * 2.0
            
            return reward

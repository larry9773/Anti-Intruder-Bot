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

class BallFinderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, max_steps = 200, render_mode=None):
        super().__init__()
        self.habitat_sim = HabitatSimBuilder()
        self.max_steps = max_steps
        self.render_mode = render_mode

        # RGB
        img_shape = (1080, 1920, 3)
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

        sim_settings = self.habitat_sim.make_default_settings()
        sim_settings["scene"] = "data/scene_datasets/cantwell/Cantwell.glb"
        sim_settings["sensor_pitch"] = 0

        cfg = self.habitat_sim.make_cfg(sim_settings)

        if self.habitat_sim.sim != None:
            self.habitat_sim.sim.close()
        # initialize the simulator
        self.habitat_sim.sim = habitat_sim.Simulator(cfg)

        # Add the ball to the scene
        # Managers of various Attributes templates
        self.habitat_sim.obj_attr_mgr = self.habitat_sim.sim.get_object_template_manager()
        self.habitat_sim.prim_attr_mgr = self.habitat_sim.sim.get_asset_template_manager()

        ball_template = habitat_sim.attributes.ObjectAttributes()
        ball_template.render_asset_handle = str(
            os.path.join(os.getcwd(), "data/objects/ball/ball.glb")
        )
        ball_template.scale = np.array([1.0, 1.0, 1.0])

        # set the default semantic id for this object template
        ball_template.semantic_id = 1  # @param{type:"integer"}
        ball_template_id = self.habitat_sim.obj_attr_mgr.register_template(ball_template, "ball")

        rigid_mgr = self.habitat_sim.sim.get_rigid_object_manager()
        self.habitat_sim.ball_id = rigid_mgr.add_object_by_template_id(ball_template_id)
        self.habitat_sim.object_ids.append(self.habitat_sim.ball_id)


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
        imu_obs = self.habitat_sim.get_agent_position()
        sem_obs = observation["semantic_sensor"]
        #print(">>>>>>>>>>>>>>>>>>>>>>  RETRVIED OBSERVATIONS")

        obs_dict = {"rgb": rgb_obs, "lidar": lidar_obs, "imu": imu_obs}
        
        dist = self.compute_shortest_distance_to_goal(self.habitat_sim.sim, 
                                                      self.habitat_sim.sim.agents[0].scene_node.absolute_transformation().translation, 
                                                      self.habitat_sim.ball_id.translation)
        reward = self.compute_reward(self._last_dist, dist, sem_obs)
        self._last_dist = dist

        done = False
        success = False
        if dist < self.success_dist :
            reward += 200.0
            done = True
            success = True
        
        self._step_count += 1
        truncated = False
        if self._step_count >= self.max_steps:
            self._step_count = 0
            truncated = True
        
        
        terminated = done
        #print(f"[DEBUG] step_count={self._step_count}  dist={dist:.3f}  success={success}")

        info = {
            "is_success" : success
        }

        
        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        #print(">>>>>>>>>>>>>>>>>>>>>> RESET ENV")

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
            is_navigable = pf.find_path(sp)

        # Set agent state
        agent = self.habitat_sim.sim.agents[0]  # Get our default agent
        agent_state = habitat_sim.AgentState()
        agent_state.position = mn.Vector3(agent_start)  # Position in world coordinate
        agent.set_state(agent_state)

        self.habitat_sim.ball_id.translation = mn.Vector3(ball_pos)       # setter on ManagedBulletRigidObject
        self.habitat_sim.sim.step_physics(1.0 / 60.0)

        observation = self.habitat_sim.sim.get_sensor_observations()
        rgb_obs = observation["color_sensor"]
        rgb_obs = rgb_obs[..., :3]
        position = self.habitat_sim.get_agent_position()
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
        return sp.geodesic_distance if success else float('inf')
    
    def compute_reward(self, 
                    old_distance: float,
                   new_distance: float,
                   sem_obs: np.ndarray,
                   time_step_penalty: float = -0.1):
        # 1) Reward de progression vers la balle
        distance_reward = (old_distance - new_distance)*10

        collision = self.habitat_sim.sim.get_physics_num_active_contact_points() > 0

        # 2) Reward de détection (dense ou binaire)
        if self.habitat_sim.is_ball_detected(sem_obs):
            detection_reward = 15.0  # ou plus fort si tu veux
        else:
            detection_reward = 0.0

        # 3) Pénalité collision
        collision_penalty = -5.0 if collision else 0.0

        # 4) Pénalité de temps
        time_penalty = time_step_penalty

        # 5) Gros bonus à la prise de balle (à appeler quand tu considères l’épisode terminé par succès)
        # par exemple dans ton env.step(): if distance_to_ball < threshold: done=True et reward += 50
        final_reward = 0.0

        return distance_reward + detection_reward + collision_penalty + time_penalty + final_reward

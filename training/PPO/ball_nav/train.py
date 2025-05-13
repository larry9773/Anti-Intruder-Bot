import seaborn as sns
import matplotlib
import random
import os

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

import habitat_sim
from habitat_sim.utils import common as ut
from habitat_sim.utils import viz_utils as vut
import magnum as mn


class BallFinderTrainer():
    def __init__(self):
        self.sim = None
        self.obj_attr_mgr = None
        self.prim_attr_mgr = None
        self.stage_attr_mgr = None
        self.sel_file_obj_handle = None
        self.sel_prim_obj_handle = None
        self.sel_asset_handle = None
        self.object_ids = []

    ######### Configuration Utility functions ######

    def make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.enable_physics = settings["enable_physics"]

        # Note: all sensors must have the same resolution
        '''sensors = {
            "color_sensor_1st_person": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
                "orientation": [settings["sensor_pitch"], 0.0, 0.0],
            },
            "depth_sensor_1st_person": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
                "orientation": [settings["sensor_pitch"], 0.0, 0.0],
            },
            "semantic_sensor_1st_person": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
                "orientation": [settings["sensor_pitch"], 0.0, 0.0],
            },
            # configure the 3rd person cam specifically:
            "color_sensor_3rd_person": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"] + 0.2, 0.2],
                "orientation": np.array([-np.pi / 4, 0, 0]),
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                print(">>>>>> Adding sensor ", sensor_params["sensor_type"])
                sensor_spec = habitat_sim.SensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params["sensor_type"]
                sensor_spec.resolution = sensor_params["resolution"]
                sensor_spec.position = sensor_params["position"]
                sensor_spec.orientation = sensor_params["orientation"]
                print(">>>>>> Done sensor ", sensor_params["sensor_type"])
                sensor_specs.append(sensor_spec)'''
        
        sensor_settings = {
            "height": 256, "width": 256,  # Spatial resolution of observations
            "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
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
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        print(">>>>>> Creating agent cfg")
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
        ball_template_id,
        offset=np.array([0, 2.0, -1.5]),
        orientation=mn.Quaternion(((0, 0, 0), 1)),
    ):
        agent_transform = sim.agents[0].scene_node.transformation_matrix()
        ob_translation = agent_transform.transform_point(offset)
        '''sim.set_translation(ob_translation, ob_id)
        sim.set_rotation(orientation, ob_id)'''
        ball_obj = rigid_mgr.add_object_by_template_id(ball_template_id)
        ball_obj.translation = ob_translation       # setter on ManagedBulletRigidObject
        ball_obj.rotation    = orientation     # setter on ManagedBulletRigidObject

    # A utility function for displaying observations
    def display_obs(rgb_obs: np.ndarray, depth_obs: np.ndarray):
        img_arr, title_arr = [], []
        
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
        img_arr.append(rgb_img)
        title_arr.append("rgb")
        
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        img_arr.append(depth_img)
        title_arr.append("depth")
        
        plt.figure(figsize=(12, 8))
        for i, (img, title) in enumerate(zip(img_arr, title_arr)):
            ax = plt.subplot(1, 2, i + 1)
            ax.axis("off")
            ax.set_title(title)
            plt.imshow(img)
        plt.show(block=False)

if __name__ == "__main__":
    ball_finder = BallFinderTrainer()
    sim_settings = ball_finder.make_default_settings()
    sim_settings["scene"] = "data/scene_datasets/cantwell/Cantwell.glb"
    sim_settings["sensor_pitch"] = 0

    cfg = ball_finder.make_cfg(sim_settings)
    # clean-up the current simulator instance if it exists
    if ball_finder.sim != None:
        ball_finder.sim.close()
    # initialize the simulator
    ball_finder.sim = habitat_sim.Simulator(cfg)
    print(">>>>>>>> Instantiated sim")

    print(">>>>>>>>> ball_finder.sim = ", ball_finder.sim)
    ball_finder.remove_all_objects()
    observations = []

    # @markdown Set the initial object orientation via local Euler angle (degrees):
    orientation_x = 45  # @param {type:"slider", min:-180, max:180, step:1}
    orientation_y = 45  # @param {type:"slider", min:-180, max:180, step:1}
    orientation_z = 45  # @param {type:"slider", min:-180, max:180, step:1}


    # compose the rotations
    rotation_x = mn.Quaternion.rotation(mn.Deg(orientation_x), mn.Vector3(1.0, 0, 0))
    rotation_y = mn.Quaternion.rotation(mn.Deg(orientation_y), mn.Vector3(0, 1.0, 0))
    rotation_z = mn.Quaternion.rotation(mn.Deg(orientation_z), mn.Vector3(0, 0, 1.0))
    object_orientation = rotation_z * rotation_y * rotation_x
    print(object_orientation)

    # add a box with default semanticId configured in the template
    # Note: each face of this box asset is a separate component

    # Managers of various Attributes templates
    ball_finder.obj_attr_mgr = ball_finder.sim.get_object_template_manager()
    ball_finder.prim_attr_mgr = ball_finder.sim.get_asset_template_manager()
    #stage_attr_mgr = ball_finder.sim.get_stage_template_manager()
    
    ball_template = habitat_sim.attributes.ObjectAttributes()
    ball_template.render_asset_handle = str(
        os.path.join(os.getcwd(), "data/objects/ball/ball.glb")
    )
    ball_template.scale = np.array([0.2, 0.2, 0.2])

    # set the default semantic id for this object template
    ball_template.semantic_id = 0  # @param{type:"integer"}
    ball_template_id = ball_finder.obj_attr_mgr.register_template(ball_template, "ball")

    #ball_id = ball_finder.sim.add_object(ball_template_id)
    rigid_mgr = ball_finder.sim.get_rigid_object_manager()
    ball_id = rigid_mgr.add_object_by_template_id(ball_template_id)
    ball_finder.object_ids.append(ball_id)
    ball_finder.set_object_state_from_agent(
        ball_finder.sim, ball_template_id, mn.Vector3(0.0, 1.5, -0.75), orientation=object_orientation
    )
    
    ball_finder.make_simulator_from_settings(sim_settings)


    
    observations.append(ball_finder.sim.get_sensor_observations())
    print("observation = ", observations)
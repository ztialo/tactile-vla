import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils.math import subtract_frame_transforms

from ..robot.franka3_contact_cfg import FRANKA_3_CFG
import os
import numpy as np

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
BASE_DIR = os.path.dirname(__file__)
ASSET_DIR = os.path.join(BASE_DIR, "..", "assets")  
SCENE_USD_PATH = os.path.join(ASSET_DIR, "franka_scene.usd")


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """Configuration for a workstation scene with a table and a single piper robot arm."""

    # Scene
    scene = AssetBaseCfg(
        prim_path="/World/scene",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=SCENE_USD_PATH),
    )

    # Articulation
    franka = FRANKA_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka_arm")
    franka.init_state.pos = (0.0, 1.5, 1.0)
    # Cube
    # dice_cube = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Cube",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(root_dir, "assets", "Props", "dice", "dice_cube.usd"),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    # )
    # dice_cube.init_state.pos = (0.35, 0.19, 0.2)

    # Desk camera
    # desk_cam = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/desk_sensor",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     # data_types=["rgb", "distance_to_image_plane"],
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=16.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=24,
    #         clipping_range=(0.01, 1.0e5),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, 0.0, 0.72),
    #         rot=(0.1830127, -0.6830127, 0.6830127, -0.1830127),
    #     ),
    # )

    # # Hand camera
    # hand_cam = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Piper_arm/link6/hand_cam/camera_sensor",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     # data_types=["rgb", "distance_to_image_plane"],
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=16.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=24,
    #         clipping_range=(0.01, 1.0e5),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.03, 0.0, 0.0),
    #         rot=(-0.6916548, 0.1470158, 0.1470158, -0.6916548),
    #     ),
    # )

    # # wall obstacle
    # obstacle_wall = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Wall",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(
    #             root_dir, "assets", "Props", "obstacle", "obstacle_cube.usd"
    #         ),
    #         scale=(0.3, 0.05, 0.15),
    #         # rigid, collision, and mass are preset in USD
    #     ),
    # )
    # obstacle_wall.init_state.pos = (0.35, -0.1, 0.0)


@configclass
class FrankaEnvCfg(DirectRLEnvCfg):
    """Configuration for a single arm environment."""

    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=1, env_spacing=2.5)
    # Specifically for direct RL
    action_space = 7
    observation_space = 7
    state_space = 10
    # Init range
    # init_pos_ranges = {
    #     # "cube": [[0.2, -0.3, 0.0], [0.4, 0.3, 0.0]],
    #     # "target_goal": [[0.2, -0.3, -0.035], [0.4, 0.3, -0.035]],
    #     # reduce the range since we moved piper arm from origin to (0, -0.25, 0)
    #     "cube": [[0.2, -0.3, 0.0], [0.4, 0.1, 0.0]],  # max y from 0.3 -> 0.1
    #     "target_goal": [[0.2, -0.3, -0.035], [0.4, 0.1, -0.035]],
    #     "obstacle_wall": [[0.25, -0.3, 0.05], [0.4, 0.1, 0.05]],
    # }
    # init_rot_ranges = {"cube": [[0.0, 0.0, -np.pi / 3], [0.0, 0.0, np.pi / 3]]}
    # # Target goal
    # target_goal_size = 0.1
    # rerender_on_reset = True

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.physics_material = self.scene.ground.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        # self.viewer.origin_type = "asset_root"
        # self.viewer.asset_name = "robot"
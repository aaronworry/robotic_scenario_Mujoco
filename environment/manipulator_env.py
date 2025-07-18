import time

import mujoco
import numpy as np
from gym import utils
from gym.spaces import Box

from environment.baseEnv import BaseEnv
from utils.util import quat2angle, angle_dir_to_quat, csd_conjquatmat_wb_fn


class CubeManipulationEnv(BaseEnv):
    # One Manipulator and One Cube in a plane
    def __init__(self, **config):
    
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "single_rgb_array",
                "single_depth_array",
            ],
            "render_fps": 10,
        }

        self.frame_skip = 50

        # define observation and action space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(6+3+3,), dtype=np.float64)
        self.action_low = -0.03
        self.action_high = 0.03
        self.action_space = Box(low=self.action_low, high=self.action_high, shape=(6,), dtype=np.float64)
        self.n_qvel = 3
        self.n_qpos = 3
        self.control_dim = 6

        self.control_lb = self.action_low * np.ones(self.control_dim)
        self.control_ub = self.action_high * np.ones(self.control_dim)

        self.state_lb = np.array([-0.08, -0.08, -2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
        self.state_ub = np.array([0.08, 0.08, 2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        # initial environment
        # load model
        self.robot = 
        self.cube = 
        self.reset_model()
        
        #self.init_qpos = self.data.qpos.ravel().copy()
        #self.init_qvel = self.data.qvel.ravel().copy()

    def reset_model(self):

        # robot initialization
        self.robot.initialization()
        # set target to initial state
        # then move to initial state

        # object initialization
        self.cube.initialization()

        # move to initial state
        self.robot.set_initial_state()
        self.cube.set_initial_state()
        
        # set the objectives, such as the target of the cube
        self.cube.set_target_state()

        # do the forward
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()


    def step(self, action):
        
        self.robot.step(action)

        # self.renderer.render_step()
        mujoco.mj_forward(self.model, self.data)
        
        robot_state, cube_state = self._get_obs()
        
        reward, terminated = self.reward(robot_state, cube_state)

        return (
            robot_state,
            cube_state,
            reward,
            terminated
        )
        
    def reward(self, robot_state, cube_state):
        return 0


    def _get_obs(self):
        robot_state_obs = self.robot._get_obs()
        cube_state_obs = self.cube._get_obs()
        return robot_state_obs, cube_state_obs

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 1.0


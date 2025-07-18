import time

import mujoco
import numpy as np
from gym.spaces import Box

from environment.baseEnv import BaseEnv


class ContinumRobotEnv(BaseEnv):
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
        
        self.target_state = None


        # initial environment
        # load model
        self.robot = 
        self.reset_model()

    def reset_model(self):

        # robot initialization
        self.robot.initialization()
        # set target to initial state
        # then move to initial state

        
        # set the objectives, such as the target of the robot
        self.robot.set_target_state(self.target_state)

        # do the forward
        mujoco.mj_forward(self.robot.model, self.robot.data)
        return self._get_obs()


    def step(self, action):
        
        self.robot.step(action)

        # self.renderer.render_step()
        mujoco.mj_forward(self.robot.model, self.robot.data)
        # mujoco.mj_step(self.robot.model, self.robot.data)
        
        robot_state = self._get_obs()
        
        reward, terminated = self.reward(robot_state)

        return (
            robot_state,
            reward,
            terminated
        )
        
    def reward(self, robot_state):
        return 0


    def _get_obs(self):
        robot_state_obs = self.robot._get_obs()
        return robot_state_obs

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 1.0


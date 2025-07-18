import time

import mujoco
import numpy as np
from gym.spaces import Box

from environment.baseEnv import BaseEnv
from robots.continum_robot import ContinumRobot


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
        self.action_space = Box(low=self.action_low, high=self.action_high, shape=(2,), dtype=np.float64)
        self.n_qvel = 3
        self.n_qpos = 3
        self.control_dim = 6

        self.control_lb = self.action_low * np.ones(self.control_dim)
        self.control_ub = self.action_high * np.ones(self.control_dim)

        self.state_lb = np.array([-0.08, -0.08, -2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
        self.state_ub = np.array([0.08, 0.08, 2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        self.target_state = None
        
        super().__init__(
            "rope/2d_spring_pos.xml",
            self.frame_skip,
            observation_space=self.observation_space,
            **config
        )

        # initial environment
        # load model
        self.robot = ContinumRobot(self.model, self.data)
        self.reset_model(True)

    def reset_model(self, render = False):

        # robot initialization
        self.robot.initialization()
        # set target to initial state
        # then move to initial state

        
        # set the objectives, such as the target of the robot
        self.robot.set_target_state(self.target_state)

        # do the forward
        mujoco.mj_forward(self.model, self.data)
        
        if render:
            self.render()
        return self._get_obs()


    def step(self, current_time, action):
        
        # 施加的力清零
        self.data.xfrc_applied.fill(0.0)
        
        self.robot.step(current_time, self.data, action)

        # self.renderer.render_step()
        # mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        
        # 需要将 data 中的值 传输给 机器人对象
        
        robot_state = self._get_obs()
        
        reward, terminated = self.reward(robot_state)

        return (
            robot_state,
            reward,
            terminated
        )
        
    def reward(self, robot_state):
        return 0, False


    def _get_obs(self):
        robot_state_obs = self.robot._get_obs(self.data)
        return robot_state_obs

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 1.0


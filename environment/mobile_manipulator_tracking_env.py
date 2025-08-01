import time
import numpy as np

from gym.spaces import Box

from environment.baseEnv import BaseEnv
from simulator.mujoco_wrapper import MujocoConnect
from robots.kinova import Kinova
from robots.dingo import Dingo


class Target():
    def __init__(self):
        pass

class MobileManipulatorTrackingControlEnv(BaseEnv, MujocoConnect):
    def __init__(self, model_path):
        MujocoConnect.__init__(model_path)
        # define observation and action space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2+6,), dtype=np.float64)
        
        self.action_space = Box(low=self.action_low, high=self.action_high, shape=(2+6,), dtype=np.float64)
        
        self.arm_action_low = -0.03
        self.arm_action_high = 0.03
        self.n_arm_pos = 6
        self.n_arm_vel = 6
        self.arm_control_dim = 6
        
        self.chassis_action_low = -0.1
        self.chassis_action_high = 0.1
        self.n_chassis_vel = 2
        self.n_chassis_pos = 2
        self.chassis_control_dim = 2

        self.arm_control_lb = self.arm_action_low * np.ones(self.arm_control_dim)
        self.arm_control_ub = self.arm_action_high * np.ones(self.arm_control_dim)
        self.chassis_control_lb = self.chassis_action_low * np.ones(self.chassis_control_dim)
        self.chassis_control_ub = self.chassis_action_high * np.ones(self.chassis_control_dim)

        self.arm_state_lb = np.array([-0.08, -0.08, -2, -0.1, -0.1, -0.1])
        self.arm_state_ub = np.array([0.08, 0.08, 2, 0.1, 0.1, 0.1])
        self.chassis_state_lb = np.array([-5., -5.])
        self.chassis_state_ub = np.array([5., 5.])
        
        
        self.target_mover = Target()
        

        # initial environment
        # load model
        self.initialization()
        self.reset()
        
    def initialization(self):
        """
        arange name in mojuco to robot and arm
        then initallize the environment
        """
        
        self.arm = Kinova()
        self.chassis = Dingo()

    def reset(self, render = False):

        # robot initialization
        
        self.chassis.reset()
        self.arm.reset()
        
        # set the objectives, such as the target of the robot
        self.arm.set_target_state(self.target_state)

        # do the forward
        self.forward()
        
        return self._get_obs()


    def step(self, current_time, action):
        
        
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
        
    
    def change_mode(self, name, mode: Literal["position", "torque"]) -> None:
        """Change the control mode of the Kinova arm."""
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if mode == "position":
            self.model.actuator_biasprm[idx][1] = self.default_biasprm[idx][1]
            self.model.actuator_gainprm[idx][0] = self.default_gainprm[idx][0]
        elif mode == "torque":
            self.model.actuator_biasprm[idx][1] = 0
            self.model.actuator_gainprm[idx][0] = 0

        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if mode == "position":
            self.model.actuator_biasprm[idx][2] = self.default_biasprm[idx][2]
            self.model.actuator_gainprm[idx][0] = self.default_gainprm[idx][0]
        elif mode == "torque":
            self.model.actuator_biasprm[idx][2] = 0
            self.model.actuator_gainprm[idx][0] = 0
        




    

    
        
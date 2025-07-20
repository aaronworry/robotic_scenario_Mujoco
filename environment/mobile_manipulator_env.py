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

class MobileManipulatorTrackingControl(BaseEnv, MujocoConnect):
    def __init__(self):
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
        self.reset_model()
        
    def initialization(self):
        """
        arange name in mojuco to robot and arm
        then initallize the environment
        """
        
        self.arm = Kinova()
        self.chassis = Dingo()

    def reset_model(self, render = False):

        # robot initialization
        self.arm.reset()
        self.chassis.reset()
        
        # set the objectives, such as the target of the robot
        self.robot.set_target_state(self.target_state)

        # do the forward
        mujoco.mj_forward(self.model, self.data)
        
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
        
        
        
        
        
        
        
"""Provides the mujoco simulation of the robot."""

    def __init__(self) -> None:
        xml = str(os.path.dirname(models.__file__) + "/" + MODEL)
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.define_robots()

        self.x0 = self.y0 = self.rotz0 = None

        self.active = True
        

        self.default_biasprm = self.model.actuator_biasprm.copy()
        self.default_gainprm = self.model.actuator_gainprm.copy()

        pref = [np.deg2rad(pos) for pos in Position.pref.position]
        self.set_ctrl_value("Kinova", "position", pref)
        self.set_qpos_value("Kinova", "position", pref)

    def step(self) -> None:
        """Perform a simulation step."""
        mujoco.mj_step(self.model, self.data)
        
    def define_robots(self) -> None:
        """Define which robots are simulated."""
        names = str(self.model.names)
        self.name = re.search("b'(.*?)\\\\", names)[1]
        self.kinova = "Kinova" in names
        self.dingo = "Dingo" in names
        
    def start(self) -> None:
        """Start a mujoco simulation."""
        self.load_window_commands()
        viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.key_callback
        )
        sync = time.time()
        while self.active:
            step_start = time.time()
            self.step()
            if time.time() > sync + (1 / SYNC_RATE):
                viewer.sync()
                sync = time.time()
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def key_callback(self, key: int) -> None:
        """Key callback."""
        if key == 256:
            self.stop()

    def stop(self, *args: any) -> None:
        """Stop the simulation."""
        self.active = False

    def update_target(self, pos: np.ndarray) -> None:
        """Update the target."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.data.mocap_pos[body_id - 1] = pos

    def reset_target(self) -> None:
        """Reset the target."""
        self.update_target(self.end_effector)

    def change_mode(self, mode: Literal["position", "torque"], joint: int) -> None:
        """Change the control mode of the Kinova arm."""
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"Kinova_position_{joint}")
        if mode == "position":
            self.model.actuator_biasprm[idx][1] = self.default_biasprm[idx][1]
            self.model.actuator_gainprm[idx][0] = self.default_gainprm[idx][0]
        elif mode == "torque":
            self.model.actuator_biasprm[idx][1] = 0
            self.model.actuator_gainprm[idx][0] = 0

        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"Kinova_velocity_{joint}")
        if mode == "position":
            self.model.actuator_biasprm[idx][2] = self.default_biasprm[idx][2]
            self.model.actuator_gainprm[idx][0] = self.default_gainprm[idx][0]
        elif mode == "torque":
            self.model.actuator_biasprm[idx][2] = 0
            self.model.actuator_gainprm[idx][0] = 0

    def get_sensor_feedback(
        self,
        robot: Literal["Kinova", "Dingo"],
        prop: Literal["position", "velocity", "torque"],
    ) -> list[float]:
        """Return position, velocity or torque feedback for kinova arm or dingo base."""
        if robot == "Kinova":
            actuators = 6
        elif robot == "Dingo":
            actuators = 4

        feedback = []
        for n in range(actuators):
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{robot}_{prop}_{n}")
            feedback.append(self.data.sensordata[idx])
        return feedback

    def set_ctrl_value(
        self,
        robot: Literal["Kinova", "Dingo"],
        prop: Literal["position", "velocity", "torque"],
        values: list[float],
    ) -> None:
        """Set position, velocity or torque command for kinova arm or dingo base."""
        for n, value in enumerate(values):
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}_{prop}_{n}")
            self.data.ctrl[idx] = value
            
    def set_qpos_value(
        self,
        robot: Literal["Kinova", "Dingo"],
        prop: Literal["position", "velocity"],
        values: list[float],
    ) -> None:
        """Set the joint position or velocity for kinova arm or dingo base."""
        for n, value in enumerate(values):
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}_{n}")
            if prop == "position":
                idpos = self.model.jnt_qposadr[idx]
                self.data.qpos[idpos] = value
            elif prop == "velocity":
                idvel = self.model.jnt_dofadr[idx]
                self.data.qvel[idvel]

    def set_world_pos_value(
        self, x: float, y: float, quat_w: float, quat_z: float
    ) -> None:
        """Set the world position or for the dingo base."""
        self.x0 = x if not self.x0 else self.x0
        self.y0 = y if not self.y0 else self.y0
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "D_J_B")
        idpos = self.model.jnt_qposadr[idx]
        self.data.qpos[idpos] = x - self.x0
        self.data.qpos[idpos + 1] = y - self.y0
        self.data.qpos[idpos + 3] = quat_w
        self.data.qpos[idpos + 6] = quat_z

    def ctrl_increment(
        self,
        increments: list[float],
        robot: Literal["Kinova", "Dingo"] = "Kinova",
        prop: Literal["position", "velocity", "torque"] = "position",
    ) -> None:
        """Control increment."""
        for n, increment in enumerate(increments):
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}_{prop}_{n}")
            self.data.ctrl[idx] += increment
            
            
    def load_window_commands(self) -> None:
        """Load the window commands."""
        glfw.init()
        window_commands = WindowCommands(1)
        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        pose = [int(width / 3), 0, int(width * (2 / 3)), height]
        window_commands = WindowCommands(1)
        window_commands.add_window(self.name)
        window_commands.add_command(["replace", (self.name, *pose)])
        window_commands.add_command(["key", (self.name, "Tab")])
        window_commands.add_command(["key", (self.name, "Shift+Tab")])
        window_commands.start_in_new_thread()
        
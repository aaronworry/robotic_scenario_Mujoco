from __future__ import annotations
from typing import Literal
import os
import mujoco
import asset as models
import time
import re
from compliant_control.interface.window_commands import WindowCommands
import glfw
import numpy as np

SYNC_RATE = 60
MODEL = "arm_and_base.xml"


    """
    @property
    def end_effector(self) -> np.ndarray:
        # Get the position of the kinova end_effector.
        if not self.kinova:
            return [0, 0, 0]
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        return self.data.site_xpos[idx]
    """

class MujocoConnect():
    """Provides the mujoco simulation of the robot."""

    def __init__(self, model_path) -> None:
        
        if model_path.startswith("/"):
            self.fullpath = model_path
        else:
            self.fullpath = path.join(path.dirname(__file__), "../assets", model_path)
        if not path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")
        
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)
        self.define_robots()

        self.x0 = self.y0 = self.rotz0 = None

        self.active = True
        self.target_mover = TargetMover(self.get_target, self.update_target)

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

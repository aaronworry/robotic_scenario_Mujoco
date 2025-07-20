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
        
        self.world_pos_name = None

        self.x0 = self.y0 = self.rotz0 = None

        self.active = True

        self.default_biasprm = self.model.actuator_biasprm.copy()
        self.default_gainprm = self.model.actuator_gainprm.copy()
        
        self.create_environment()
        
    def create_environment(self):
        self.robots_state_dict = {}
        """
        key value
        {robot1_name : {body_names : [], joint_names: [], sensor_names: [], actuator_names: []}, robot2_name : {} }
        """
        self.objects_state_dict = []
        """
        list
        [object1_name, object2_name, ]
        """
        pass

    def step(self) -> None:
        """Perform a simulation step."""
        mujoco.mj_step(self.model, self.data)
        
    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        
        
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


    def set_body_pos(self, name, pos: np.ndarray) -> None:
        """Update the target."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.data.mocap_pos[body_id - 1] = pos
        
    def get_body_pos(self, name) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.mocap_pos[body_id - 1]

    def get_sensor_feedback(self, name) -> float:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return self.data.sensordata[idx]

    def set_actuator_value(self, name, value: float) -> None:
        """Set position, velocity or torque command for kinova arm or dingo base."""
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        self.data.ctrl[idx] = value
        
    def get_actuator_value(self, name) -> float:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return self.data.ctrl[idx]
            
    def set_joint_pos_value(self, name, value: float) -> None:
        """Set the joint position or velocity for kinova arm or dingo base."""
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        idpos = self.model.jnt_qposadr[idx]
        self.data.qpos[idpos] = value
        
    def get_joint_pos_value(self, name) -> float:
        """Set the joint position or velocity for kinova arm or dingo base."""
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        idpos = self.model.jnt_qposadr[idx]
        return self.data.qpos[idpos]
                
    def set_joint_vel_value(self, name, value: float) -> None:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        idvel = self.model.jnt_dofadr[idx]
        self.data.qvel[idvel] = value
        
    def get_joint_vel_value(self, name) -> float:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        idvel = self.model.jnt_dofadr[idx]
        return self.data.qvel[idvel]

    def set_world_pos_value(
        self, x: float, y: float, quat_w: float, quat_z: float
    ) -> None:
        """Set the world position or for the dingo base."""
        self.x0 = x if not self.x0 else self.x0
        self.y0 = y if not self.y0 else self.y0
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.world_pos_name)
        idpos = self.model.jnt_qposadr[idx]
        self.data.qpos[idpos] = x - self.x0
        self.data.qpos[idpos + 1] = y - self.y0
        self.data.qpos[idpos + 3] = quat_w
        self.data.qpos[idpos + 6] = quat_z

    def ctrl_increment(
        self,
        name,
        increment: float,
        robot: Literal["Kinova", "Dingo"] = "Kinova",
        prop: Literal["position", "velocity", "torque"] = "position",
    ) -> None:
        """Control increment."""
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
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

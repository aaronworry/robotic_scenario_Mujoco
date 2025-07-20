from functools import partial
from os import path
from typing import Optional

import gym
import numpy as np
from gym import error, spaces
from gym.spaces import Space

from .mujoco_renderer import Viewer
import mujoco

class MujocoEnv():
    """Superclass for all MuJoCo environments."""

    def __init__(
            self,
            model_path,
            frame_skip,
    ):
        
        if model_path.startswith("/"):
            self.fullpath = model_path
        else:
            self.fullpath = path.join(path.dirname(__file__), "../assets", model_path)
        if not path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")
        
        self._initialize_simulation()
        
        self.render_fps = 10.          # read from file
        
        self._viewers = {}          # view in global and view from sensor

        self.frame_skip = frame_skip

        self.viewer = None       # view in global

        
        assert (
                int(np.round(1.0 / self.dt)) == self.render_fps
        ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.render_fps}'
        


    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """
        pass

    def _initialize_simulation(self):
        """
        Initialize MuJoCo simulation data structures mjModel and mjData.
        """
        self.model = mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        """
        Reset MuJoCo simulation data structures, mjModel and mjData.
        """
        mujoco.mj_resetData(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        """
        Step over the MuJoCo simulaion.
        """
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)
        
    def render(self):
        self._get_viewer("global").render()
        

    def get_camera_data(
            self,
            name: str = "global",
    ):
        """
        Render a frame from the MuJoCo simulation and get_data.
        """
        if name == "rgb":
            width = 1
            height = 1
            data = self._get_viewer(name).read_pixels(width, height, depth=False)
            return data[::-1, :, :]
        elif name == "depth":
            width = 1
            height = 1
            data = self._get_viewer(name).read_pixels(width, height, depth=True)[1]
            return data[::-1, :]
        elif name == "global":
            width = self.model.vis.global_.offwidth
            height = self.model.vis.global_.offheight
            data = self._get_viewer(name).read_pixels(width, height, depth=False)
            return data[::-1, :, :]

    # -----------------------------

    def reset(
            self,
            return_info: bool = False,
    ):
        self._reset_simulation()

        ob = self.reset_model()
        if not return_info:
            return ob
        else:
            return ob, {}

    def set_state(self, qpos, qvel):
        """
        Set the joints position qpos and velocity qvel of the model. Override this method depending on the MuJoCo bindings used.
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        self._step_mujoco_simulation(ctrl, n_frames)



    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def get_body_com(self, body_name):
        """Return the cartesian position of a body frame"""
        return self.data.body(body_name).xpos

    def state_vector(self):
        """Return the position and velocity joint states of the model"""
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        
    def _get_viewer(self, name):
        if name == "global":
            self.viewer = self._viewers.get("global")
            if self.viewer is None:
                self.viewer = Viewer("global", self.model, self.data)
                self.viewer_setup()
                self._viewers[name] = self.viewer
            return self.viewer
        else:
            viewer = self._viewers.get(name)
            if viewer is None:
                viewer = Viewer(name, self.model, self.data)
            return viewer
            
        




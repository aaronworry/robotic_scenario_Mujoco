import mujoco
import numpy as np
from os import path


# 根据大小、作用点和指向点求力的向量
def force_vec_target(magnitude, tail, target):  # 大小，作用点，指向的点
    direction_vec = np.array(target - tail)
    distance = np.linalg.norm(direction_vec)
    force = magnitude / distance * direction_vec
    return force


# 在某body上的某site施加力
def apply_force(data, force, body_id, site_pos):  # 力的向量（使用force_vec生成），作用body的id，该body上作用site的坐标
    body_pos = data.xipos[body_id]  # body的质心坐标
    r_vec = site_pos - body_pos  # 力的平移：转换为施加在质心的力和力矩
    torque = np.cross(r_vec, force)
    data.xfrc_applied[body_id, :3] += force
    data.xfrc_applied[body_id, 3:] += torque

class ContinumRobot():
    def __init__(self, model, data, initial_state = None, controller = None):
        
        self.initial_state = initial_state
        self.state = None
        self._create_robot(model, data, controller)
        
            
    def _create_robot(self, model, data, controller):
        self.set_controller(controller)
        
        self.s0L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "s0_L")
        self.s1L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "s1_L")
        self.s2L_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "s2_L")
        self.s0R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "s0_R")
        self.s1R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "s1_R")
        self.s2R_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "s2_R")

        self.b0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B0")
        self.b1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B1")
        self.b2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B2")
        
        self._get_obs(data)
        
        
    def set_target_state(self, target_state):
        self.target_state = target_state
        
    
    def initialization(self):
        if self.initial_state is not None:
            self.move_to_initial_state()
        
        
    def _get_obs(self, data):
        self.s0L_pos = data.site_xpos[self.s0L_id]
        self.s1L_pos = data.site_xpos[self.s1L_id]
        self.s2L_pos = data.site_xpos[self.s2L_id]
        self.s0R_pos = data.site_xpos[self.s0R_id]
        self.s1R_pos = data.site_xpos[self.s1R_id]
        self.s2R_pos = data.site_xpos[self.s2R_id]

        self.state = np.concatenate((self.s0L_pos, self.s1L_pos, self.s2L_pos, self.s0R_pos, self.s1R_pos, self.s2R_pos))
        return self.state
        
    def step(self, current_time, data, high_level_action = None):
        if high_level_action is not None:
            self.controller(high_level_action)
        else:
            # 两根绳子施加的力
            if current_time >= 0 and current_time <= 10:
                f_L = -0.5 * np.cos(0.2 * np.pi * current_time) + 0.5
                f_R = 0
            elif current_time > 10 and current_time <= 20:
                f_L = 0
                f_R = -0.5 * np.cos(0.2 * np.pi * current_time) + 0.5
                
            
            
            # 左侧
            apply_force(data, force_vec_target(f_L, self.s2L_pos, self.s1L_pos), self.b2_id, self.s2L_pos)
            apply_force(data, force_vec_target(f_L, self.s1L_pos, self.s0L_pos) + force_vec_target(f_L, self.s1L_pos, self.s2L_pos), self.b1_id, self.s1L_pos)

            # 右侧
            apply_force(data, force_vec_target(f_R, self.s2R_pos, self.s1R_pos), self.b2_id, self.s2R_pos)
            apply_force(data, force_vec_target(f_R, self.s1R_pos, self.s0R_pos) + force_vec_target(f_R, self.s1R_pos, self.s2R_pos), self.b1_id, self.s1R_pos) 
            
        
    def set_controller(self, controller):
        self.controller = controller
        
    def move_to_initial_state(self):
        pass
        
    def reset():
        pass
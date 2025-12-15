import mujoco
import numpy as np
from trajectory_control.utils.piper_forward import Arm_FK


class MujocoArmControl():
    def __init__(self, model, data, v_max, dt, T_ee):
        self.joint_names = ["item_1/joint1", "item_1/joint2", "item_1/joint3", "item_1/joint4", "item_1/joint5", "item_1/joint6"]
        self.gripper_name = "item_1/gripper"           # 0闭合 0.035张开最大
        self.target_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pid_controllers = [PIDController(1., 0, 0.) for _ in range(6)]
        self.model = model
        self.data = data
        self.v_max = v_max
        self.dt = dt
        self.T_ee = T_ee

        self.arm_fk = Arm_FK(6, [[0., 0., 0.123], [-np.pi/2, 0., 0.], [0., 0.285, 0.], [np.pi/2, -0.022, 0.25], [-np.pi/2, 0., 0.], [np.pi/2, 0., 0.091]])
        self.offset_theta_list = [0., -np.pi * 174.22 / 180, -100.78 / 180 * np.pi, 0., 0., 0.]

    def set_target_angles(self, target_angles):
        self.target_angles = target_angles

    def get_current_angles(self):
        current_angles = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            current_angles.append(self.data.qpos[joint_id])  # 获取每个关节的当前角度
        return current_angles

    def get_ee_position_and_rotation(self, current_q):
        q = [current_q[i] + self.offset_theta_list[i] for i in range(len(self.offset_theta_list))]
        T = self.arm_fk.forward_compute(q) @ self.T_ee
        pos = T[:3, 3]
        rotation = T[:3, :3]
        return pos, rotation

    def control(self):
        current_angles = self.get_current_angles()
        control_signals = []
        for i in range(6):
            # control_signal = self.pid_controllers[i].calculate(self.target_angles[i], current_angles[i])
            control_signal = self.target_angles[i]
            control_signals.append(control_signal)
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.joint_names[i])
            self.data.ctrl[joint_id] = control_signal # + self.target_angles[i]

    def grasp_open(self):
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_name)
        self.data.ctrl[joint_id] = 0.105

    def grasp_close(self):
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_name)
        self.data.ctrl[joint_id] = 0.0

    def interpolate(self, waypoint_position):
        current_joint = self.get_current_angles()
        current_ee_position, _ = self.get_ee_position_and_rotation(current_joint)
        distance = np.linalg.norm(current_ee_position - waypoint_position)
        total_num = int(distance / self.v_max / self.dt)
        result = np.zeros((total_num, 3))
        for i in range(total_num):
            result[i, :] = current_ee_position + (i+1) / total_num * (waypoint_position - current_ee_position)
        return result

    def check_arrive(self, waypoint_position, tol):
        current_joint = self.get_current_angles()
        current_ee_position, _ = self.get_ee_position_and_rotation(current_joint)
        error_sum = np.linalg.norm(current_ee_position - waypoint_position)
        if error_sum <= tol:
            return True
        else:
            return False

    def check_joint(self, joint_state, tol):
        current_joint = self.get_current_angles()
        length = min(len(joint_state), len(current_joint))
        error_sum = np.linalg.norm(np.array(current_joint[:length]) - joint_state[:length])
        if error_sum <= tol:
            return True
        else:
            return False

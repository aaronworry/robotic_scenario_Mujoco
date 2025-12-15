import os
import time
import glob, shutil
from pathlib import Path
from casadi import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import transforms3d as t3d
import math

def hadamard_sum(A, B):
    n = len(A)
    m = len(A[0])
    result = 0.
    for i in range(n):
        for j in range(m):
            result += A[i][j] * B[i][j]
    return result

def save_log(train_id):
    # Copy file to log for version backup
    log_path = os.path.join(Path(__file__).resolve().parent, "logs/{}".format(train_id))
    if not os.path.exists(os.path.join(log_path, "logs")):
        os.makedirs(os.path.join(log_path, "logs"))
    for file in glob.glob(os.path.join(Path(__file__).resolve().parent, "*.py")):
        shutil.copy(file, os.path.join(log_path, "logs/"))


def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.
    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.
    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.
    """
    if timestep > .04 or i % (int(1 / (24 * timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i * timestep):
            time.sleep(timestep * i - elapsed)


def print_info():
    from inspect import currentframe, getframeinfo
    frame_info = getframeinfo(currentframe())
    print(frame_info.filename, frame_info.lineno)
    
    
def quat2angle(q):
    return 2.0 * math.acos(q[0]) * np.sign(q[-1])
    

# converter to quaternion from (radian angle, direction)
def angle_dir_to_quat(angle, direction):
    if type(direction) == list:
        direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    quat = np.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * direction
    return quat
    
def quaternion2euler(quat):
    """
    将四元数转化为欧拉角，描述3D空间中的旋转
    :param quat: a list with four elements:  [q_x, q_y, q_z, q_w]
    :return: a list with three angles, in radius: [rx, ry, rz]
    """
    r = R.from_quat(quat)
    rx, ry, rz = r.as_euler("xyz", degrees = False)
    return [rx, ry, rz]

def quaternion2rotation(quatndarray):
    """
    将四元数转化为欧拉角，描述3D空间中的旋转
    :param quatndarray: a list with four elements:  [q_x, q_y, q_z, q_w]
    :return: a 3x3 rotation matrix
    """
    quat_r = np.array([quatndarray[3], quatndarray[0], quatndarray[1], quatndarray[2]])
    rotation_matrix = t3d.quaternions.quat2mat(quat_r)
    return rotation_matrix

def euler2quaternion(euler):
    """
    将欧拉角转化为四元数
    :param euler: a list with three angles in radius: [rx, ry, rz]
    :return: a quaternion:   qx, qy, qz, qw
    """
    r = R.from_euler('xyz', euler, degrees=False)
    quat_xyzw = r.as_quat()
    return quat_xyzw

def rotation2eular(R):
    Pos = [0.] * 6
    if R[2, 0] < -1 + 0.0001:
        Pos[4] = np.pi / 2
        Pos[5] = 0
        Pos[3] = math.atan2(R[0, 1], R[1, 1])
    elif R[2, 0] > 1 - 0.0001:
        Pos[4] = -np.PI / 2
        Pos[5] = 0
        Pos[3] = -math.atan2(R[0, 1], R[1, 1])
    else:
        # General case for Euler angles computation
        _bt = math.atan2(-R[2, 0], math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])) # pitch (beta)
        Pos[4] = _bt
        Pos[5] = math.atan2(R[1, 0] / math.cos(_bt), R[0, 0] / math.cos(_bt))
        Pos[3] = math.atan2(R[2, 1] / math.cos(_bt), R[2, 2] / math.cos(_bt))
    return Pos[3:]

def axis_angle2rotation(joint_axis, angle):
    rot = R.from_rotvec(joint_axis * angle).as_matrix()
    return rot
    



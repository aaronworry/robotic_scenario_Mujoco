import os
import time
import glob, shutil
from pathlib import Path
from casadi import *

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
def angle_dir_to_quat(angle, dir):
    if type(dir) == list:
        dir = np.array(dir)
    dir = dir / np.linalg.norm(dir)
    quat = np.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat
    



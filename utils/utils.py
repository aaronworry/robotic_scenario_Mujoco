import os
import time
import glob, shutil
from pathlib import Path


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

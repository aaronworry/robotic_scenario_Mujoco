import sys
sys.path.append("..")

import time
from environment.mobile_manipulator_tracking_env import MobileManipulatorTrackingControlEnv

env = MobileManipulatorTrackingControlEnv()
state = env.reset()
while True:
    target = None
    state, _, _ = env.step(target)
    print(state)
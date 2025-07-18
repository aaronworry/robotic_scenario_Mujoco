import sys
sys.path.append("..")

import time
from environment.continum_robot_env import ContinumRobotEnv

env = ContinumRobotEnv()
start_time = time.time()
while True:
    current_time = time.time() - start_time
    state, _, _ = env.step(current_time, action = None)
    print(state)
    time.sleep(0.01)

import sys
sys.path.append("..")

import time
from environment.continum_LS2D_tracking_env import LS2D_tracking_Env

DT = 0.1 

env = LS2D_tracking_Env()
state = env.reset()
while True:
    action = None
    state = env.step(action = action, dt = DT)

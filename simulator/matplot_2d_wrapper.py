from __future__ import annotations
import os
from matplotlib import pyplot as plt
import time
from utils.window_commands import WindowCommands
import glfw
import numpy as np

# plt.draw()  :  改变元素数据，在原图上绘制  后面一般跟 plt.show()
    

class MatplotConnect():
    """Provides the matplot visualization of the robot."""

    def __init__(self) -> None:
        pass
            
            
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

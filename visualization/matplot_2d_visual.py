from __future__ import annotations
import os
from matplotlib import pyplot as plt
import time
import numpy as np


class MatplotViewer():
    """Provides the matplot visualization of the env."""
    def __init__(self, X_limit=[-5., 5.], Y_limit=[-5., 5.]):
    
        self.xlim = X_limit
        self.ylim = Y_limit
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([self.xlim[0], self.xlim[1]])
        self.ax.set_ylim([self.ylim[0], self.ylim[1]])
        
        self.dynamic_objects_plot_list = []
        self.static_objects_plt_list = []
        
        
    def init_static_object(objects, **kwargs):
        self.drawObjects(objects, dtype="static", **kwargs)
        
    
    def com_cla(self):
        if len(self.dynamic_objects_plot_list) > 0:
            for item in self.dynamic_objects_plot_list:
                item.remove()
        self.dynamic_objects_plot_list = []
        
    def pause(self, time=0.001):
        plt.pause(time)
        
    def show(self, flag=False, visible = True):
        plt.draw()
        if not visible:
            self.ax.grid(None)
            self.ax.axis("off")
        
        if flag:
            font = FontProperties(fname=r"c:\windows\fonts\times.ttf", size=14)
            
            # font.set_family('serif')
            # font.set_name('Times New Roman')  # Must be installed on your system
            # font.set_size(14)
            # font.set_weight('bold')

            self.ax.set_xlabel("x(m)", fontproperties = myfont)
            self.ax.set_ylabel("y(m)", fontproperties = myfont)
            self.ax.set_xticks(np.linspace(self.xlim[0], 1., self.xlim[1]), fontproperties = myfont)
            self.ax.set_yticks(np.linspace(self.ylim[0], 1., self.ylim[1]), fontproperties = myfont)
            self.ax.set_xticklabels(np.linspace(self.xlim[0], 1., self.xlim[1]), fontproperties = myfont)
            self.ax.set_yticklabels(np.linspace(self.ylim[0], 1., self.ylim[1]), fontproperties = myfont)
            
            
            # self.ax.lengend(prof = font)
            
            plt.savefig('result.pdf',dpi=300,bbox_inches = "tight")
            
    
    def add_object(self, item, dtype, **kwargs):
        """
        a closet set
        item = [[point], [point], [point]]:    point_plot
        item = [point_list, point_list, point_list]:   line_plot
        """
        for temp_object in item:
            plot_item = None
            if len(temp_object) == 1:
                plot_item = self.point_plot(temp_object[0], **kwargs)
            elif len(temp_object) == 2:
                plot_item = self.line_plot(temp_object[0], temp_object[1], **kwargs)
                
            if dtype == "dynamic":
                self.dynamic_objects_plot_list.append(plot_item[0])
            elif dtype == "static":
                self.static_objects_plt_list.append(plot_item[0])
        
    def drawRobots(self, robots, dtype, **kwargs):
        for robot in robots:
            self.add_object(robot, dtype, **kwargs)
            
    def drawObjects(self, objects, dtype, **kwargs):
        for item in objects:
            self.add_object(item, dtype, **kwargs)

    def line_plot(self, point1, point2, marker=',', markersize=1, linewidth=2, linestyle = '-', color="black"):
        return self.ax.plot([float(point1[0]), float(point2[0])], [float(point1[1]), float(point2[1])], marker=marker, markersize=markersize,  linewidth=linewidth, linestyle=linestyle, color=color)
        

    def point_plot(self, point, markersize=10, marker='.', color='red', linewidth=2, linestyle = '-'):
        return self.ax.scatter(point[0], point[1], c=color, s = markersize, marker = marker)
        
    def render(self, robots, objects, dt):
        self.com_cla()
        self.drawRobots(robots, dtype = "dynamic", linewidth=3)
        self.drawObjects(objects, dtype = "dynamic", linewidth=1)
        self.show()
        self.pause(dt)
import math
import time
import numpy as np

import rendering
from base_display import BaseDisplay

SUCCESS = 'success'
FAILURE_TOO_MANY_STEPS = 'too_many_steps'

# Custom failure states for navigation.
NAV_FAILURE_COLLISION = 'collision'
NAV_FAILURE_OUT_OF_BOUNDS = 'out_of_bounds'

class PygletDisplay(BaseDisplay):

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x_bounds = (0.0,1.0)
        self.y_bounds = (0.0,1.0)
        self.viewer = rendering.Viewer(width,height)
  
        self._reset()


    def _reset(self):
        self.static_obstacles_geoms = []
        self.static_obstacles_geoms_xform = []
        self.obstacles = {}

        self.goal_region_geoms = []
        self.goal_region_geoms_xform = []

        self.inbounds_region_geoms = []
        self.inbounds_region_geoms_xform = []

        self.robot = None
        self.exploded_robots_geoms = []
        self.exploded_robots_geoms_xform = []


    def setup(self, x_bounds, y_bounds,
                  in_bounds, goal_bounds,
                  margin, static_obstacles):

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.margin = margin
        self.viewer.set_bounds(self.x_bounds[0],self.x_bounds[1],
                               self.y_bounds[0]-0.2,self.y_bounds[1]+0.2)

        self._add_goal_region(goal_bounds)
        self._add_inbounds(in_bounds)
        self._add_static_obstacles(static_obstacles)
        self._add_robot()


    def _add_static_obstacles(self, static_obstacles):
        self.static_obstacles_geoms = []
        self.static_obstacles_geoms_xform = []

        poses = static_obstacles['pos']
        radius = static_obstacles['radius']
        for i in range(len(poses)):
            geom = rendering.make_circle(radius[i])
            xform = rendering.Transform()

            geom.set_color(0.5, 0.5, 0.5)
            geom.add_attr(xform)
            self.static_obstacles_geoms.append(geom)
            self.static_obstacles_geoms_xform.append(xform)

        for i in range(len(poses)):
          self.static_obstacles_geoms_xform[i].set_translation(*tuple(poses[i]))

    def _add_goal_region(self, bounds):
        vertices = []
        vertices.append([bounds.x_bounds[1], bounds.y_bounds[0]])
        vertices.append([bounds.x_bounds[1], bounds.y_bounds[1]])
        vertices.append([bounds.x_bounds[0], bounds.y_bounds[1]])
        vertices.append([bounds.x_bounds[0], bounds.y_bounds[0]])       

        geom = rendering.make_polygon(vertices)
        xform = rendering.Transform()

        geom.set_color(0.0, 1, 0.0) #  green
        geom.add_attr(xform)

        self.goal_region_geoms.append(geom)
        self.goal_region_geoms_xform.append(xform)


    def _add_inbounds(self, bounds):
        vertices = []
        vertices.append([bounds.x_bounds[1], bounds.y_bounds[0]])
        vertices.append([bounds.x_bounds[1], bounds.y_bounds[1]])
        vertices.append([bounds.x_bounds[0], bounds.y_bounds[1]])
        vertices.append([bounds.x_bounds[0], bounds.y_bounds[0]])       

        geom = rendering.make_polygon(vertices)
        xform = rendering.Transform()

        geom.set_color(1.0, 1.0, 1.0)
        geom.add_attr(xform)

        self.inbounds_region_geoms.append(geom)
        self.inbounds_region_geoms_xform.append(xform)

    def _add_robot(self):
        vertices = [[-0.02, 0], [0.02, 0], [0, 0.05]]      
        geom = rendering.make_polygon(vertices)
        xform = rendering.Transform()
        geom.set_color(0.0, 1.0, 1.0)
        geom.add_attr(xform)

        self.robot = (geom, xform)


    def begin_time_step(self, t):
        self.obstacles = {}

        pass

    def end_time_step(self, t):
        self.render()

    def obstacle_at_loc(self, i, x, y, nearest_obstacle = False, close_obstacle = False):
        if i not in self.obstacles:
            # create an obstacle
            geom = rendering.make_circle(self.margin)
            xform = rendering.Transform()
            geom.set_color(0.3, 0.3, 0.3)
            geom.add_attr(xform)
            self.obstacles[i] = (geom, xform)

        (geom, xform) = self.obstacles[i]
        xform.set_translation(x, y)
        pass

    def obstacle_set_color(self, i, color = 'grey' ):
        if color == 'grey':
            color_rgb = (0.3, 0.3, 0.3)
        elif color == 'blue':
            color_rgb = (0.0, 0.0, 1.0)

        for idx, obstacle in list(self.obstacles.items()):
            (geom, xform) = obstacle
            geom.set_color(0.3, 0.3, 0.3)

        if i in self.obstacles:
            (geom, xform) = self.obstacles[i]
            geom.set_color(*color_rgb)

    def robot_at_loc(self, x, y, h, is_ssa = False):
        if self.robot is not None:
            self.robot[1].set_translation(x, y)
            self.robot[1].set_rotation(h * 180/(math.pi))
        if is_ssa:
            self.robot[0].set_color(0.9, 0.0, 0.0)
        else:
            self.robot[0].set_color(0.1, 0.1, 0.1)

    def _explode_robot(self):
        (geom, xform) = self.robot
        explode_pos = (xform.translation[0], xform.translation[1])
        # print("_explode_robot:", explode_pos)
        geom = rendering.make_circle(0.05)
        xform = rendering.Transform()
        geom.set_color(1, 0.5, 0.0)
        geom.add_attr(xform)
        self.exploded_robots_geoms.append(geom)
        self.exploded_robots_geoms_xform.append(xform)

        self.robot = None
        self.exploded_robots_geoms_xform[-1].set_translation(*explode_pos)

    def collision(self):
        self._explode_robot()

    def out_of_bounds(self):
        self._explode_robot()

    def navigation_done(self, retcode, t):
        if retcode in (NAV_FAILURE_COLLISION,
                       NAV_FAILURE_OUT_OF_BOUNDS,
                       FAILURE_TOO_MANY_STEPS):
            self._explode_robot()


    def render(self):
        self.viewer.geoms = []
        # print("#. static_obstacles:", len(self.static_obstacles_geoms))
        for i, geom in enumerate(self.goal_region_geoms):
            self.viewer.add_geom(geom)

        for i, geom in enumerate(self.static_obstacles_geoms):
            self.viewer.add_geom(geom)

        for i, geom in enumerate(self.exploded_robots_geoms):
            self.viewer.add_geom(geom)

        for idx, obstacle in list(self.obstacles.items()):
            (geom, xform) = obstacle
            self.viewer.add_geom(geom)

        if self.robot:  
          (geom, xform) = self.robot
          self.viewer.add_geom(geom)


        ## Toy example for rendering
        # geom = rendering.make_circle(0.1)
        # xform = rendering.Transform()
        # geom.set_color(1, 0, 0) # (0,0,0) for black; (1,1,1) for white
 
        # geom.add_attr(xform)
        # xform.set_translation(0, 0)
        # self.viewer.add_geom(geom)
        
        pass
        self.viewer.render()
import math
import numpy as np

class DoubleIntegratorRobot():

    def __init__(self, x, y, vx, vy, max_speed):

        self.pos = np.array([x, y])
        self.vel = np.array([vx, vy])
        self.max_speed = max_speed

        #TODO: remove it after modifying train.py
        # but changing v_x, v_y to vel causes robot not moving :(
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.v_x = self.vel[0]
        self.v_y = self.vel[1]

    @property
    def position(self):
        heading_angle = math.atan2(self.vel[1], self.vel[0])
        return (self.pos[0], self.pos[1], heading_angle)

    def steer(self, vx_speed_change, vy_speed_change):

        self.v_x = max(min(self.v_x + vx_speed_change, self.max_speed), -self.max_speed)
        self.v_y = max(min(self.v_y + vy_speed_change, self.max_speed), -self.max_speed)

        new_x = self.pos[0] + self.v_x
        new_y = self.pos[1] + self.v_y
        # warp around
        if (new_x > 1):
            new_x = -1 + (new_x - 1) 
        if (new_x < -1):
            new_x = 1 + (new_x + 1) 
        if (new_y < -1):
            new_y = -1
        if (new_y > 1):
            new_y = 1
        return DoubleIntegratorRobot(x = new_x, y = new_y, vx = self.v_x, vy = self.v_y, max_speed = self.max_speed)

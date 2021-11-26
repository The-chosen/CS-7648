import random
import numpy as np
import math

def l2( xy0, xy1 ):
    ox = xy1[0]
    oy = xy1[1]
    dx = xy0[0] - xy1[0]
    dy = xy0[1] - xy1[1]
    dist = math.sqrt( (dx * dx) + (dy * dy) )
    if (xy1[0] < -0.9):
        warp_dx = xy0[0] - (1 + (xy1[0] + 1))
        dist1 = math.sqrt( (warp_dx * warp_dx) + (dy * dy) )
        if (dist1 < dist):
            ox = (1 + (xy1[0] + 1))
            dist = dist1
            #print(f"case1")
    elif (xy1[0] > 0.9):
        warp_dx = xy0[0] - (-1 + (xy1[0] - 1))
        dist1 = math.sqrt( (warp_dx * warp_dx) + (dy * dy) )
        if (dist1 < dist):
            ox = (-1 + (xy1[0] - 1))
            dist = dist1
            #print(f"case2")
    return dist, ox, oy


class Obstacle():
    # obstacles in 2d space [x, y]
    def __init__(self, acc_start, vel_start, pos_start, t_start, safety_dist, radius):
        self.acc = np.array(acc_start) / 5 # accelerate
        self.vel = np.array(vel_start) # velocitu
        self.pos = np.array(pos_start) # pos
        self.t = t_start
        self.safety_dist = safety_dist
        self.radius = radius

    @property
    def params(self):
        return { 'acc':     self.acc,
                 'vel':     self.vel,
                 'pos':     self.pos,
                 't_start': self.t,
                 'safety_dist': self.safety_dist,
                 'radius': self.radius }

    def update_loc(self, t):
        t_duration = t - self.t
        loc = (self.acc* t_duration * t_duration) + (self.vel * t_duration) + self.pos
        return loc

    def update_vel(self, t):
        t_duration = t - self.t
        vel = ((2 * self.acc * t_duration) + self.vel)
        return vel


FIELD_X_BOUNDS = (-0.95, 0.95)
FIELD_Y_BOUNDS = (-0.95, 1.0)

class ObstacleField(object):
    
    def __init__(self, static_obs_info):
        self.x_bounds = FIELD_X_BOUNDS
        self.y_bounds = FIELD_Y_BOUNDS
        self.static_obs_info = static_obs_info
        self.random_init()

    def random_init(self):
        # TODO: Add some static obs | safety distance(as attr) [attr1: pos, attr2: safety-dis]
        obstacles = []
        for i in range(50):
            obstacles.append(self.random_init_obstacle(t = -100))

        poses = self.static_obs_info['pos']
        radius = self.static_obs_info['radius']
        for i in range(len(poses)):
            obstacles.append(Obstacle([0, 0], [0, 0], poses[i], -100, 0.08, radius[i])) 
        self.obstacles = obstacles
        return 

    def random_init_obstacle(self, t, vehicle_x = 0, vehicle_y = -1, min_dist = 0.1):
        dist = -1
        x = y = -1
        while (dist < min_dist):
            x = random.uniform(FIELD_X_BOUNDS[0],FIELD_X_BOUNDS[1])
            y = random.uniform(FIELD_Y_BOUNDS[0],FIELD_Y_BOUNDS[1])
            pos = np.array([x, y])
            pos_vehicle = np.array([vehicle_x, vehicle_y])
            dist = np.linalg.norm(pos-pos_vehicle)
            #TODO (distance to car): add a dist from static obs
        vel = np.random.uniform(1e-3, 1e-2, 2) * random.choice([1,-1])
        acc = np.random.uniform(5e-6, 1e-4, 2) * random.choice([1,-1])
        return Obstacle(acc, vel, pos, t, 0.12, 0)

    def unsafe_obstacle_locations(self, t, cx, cy, min_dist):
        unsafe_obstacles = []
        for i, obstacle in enumerate(self.obstacles):
            x = obstacle.pos[0]
            y = obstacle.pos[1]
            if self.x_bounds[0] <= x <= self.x_bounds[1] and self.y_bounds[0] <= y <= self.y_bounds[1]:
                dist, ox, oy = l2([cx,cy], [x,y])
                if dist < min_dist:
                    [x_v,y_v] = obstacle.vel
                    [x_a,y_a] = obstacle.acc
                    unsafe_obstacles.append([i,(ox,oy,x_v,y_v,x_a,y_a)])
        return  unsafe_obstacles


    def obstacle_locations(self, t, vehicle_x, vehicle_y, min_dist):
        """
        Returns (i, x, y) tuples showing that the i-th obstacle is at location (x,y).
        """
        locs = []
        for i, a in enumerate(self.obstacles):
            pos = a.update_loc(t)
            if not (self.x_bounds[0] <= pos[0] <= self.x_bounds[1] and self.y_bounds[0] <= pos[1] <= self.y_bounds[1]):
                self.obstacles[i] = self.random_init_obstacle(t, vehicle_x, vehicle_y, min_dist)
                
            locs.append((i, pos[0], pos[1]))

        return locs

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
    def __init__(self,
                 a_x, v_x, p_x,
                 a_y, v_y, p_y, t_start, safety_dist, radius):
        self.a_x = a_x / 5 # acceleration
        self.v_x = v_x # velocity
        self.p_x = p_x # pos
        self.a_y = a_y / 5
        self.v_y = v_y
        self.p_y = p_y
        self.t_start = t_start
        self.safety_dist = safety_dist
        self.r = radius

    def pos_update(self, t):
        delta_t = t - self.t_start
        x = ((self.a_x * delta_t * delta_t)
             + (self.v_x * delta_t)
             + self.p_x)
        y = ((self.a_y * delta_t * delta_t)
             + (self.v_y * delta_t)
             + self.p_y)
        return x, y

    def vel_update(self, t):
        delta_t = t - self.t_start
        v_x = ((2 * self.a_x * delta_t) + self.v_x)
        v_y = ((2 * self.a_y * delta_t) + self.v_y)
        return v_x, v_y

FIELD_X_BOUNDS = (-0.95, 0.95)
FIELD_Y_BOUNDS = (-0.95, 0.95)

class ObstacleField(object):
    
    def __init__(self, static_obs_info):
        self.x_bounds = FIELD_X_BOUNDS
        self.y_bounds = FIELD_Y_BOUNDS
        self.static_obs_info = static_obs_info
        self.random_init()

    def random_init(self):
        # TODO: Add some static obs | safety distance(as attr) [attr1: pos, attr2: safety-dis]
        obstacles = []
        for i in range(5):
            obstacles.append(self.random_init_obstacle(t = -100))
        poses = self.static_obs_info['pos']
        radius = self.static_obs_info['radius']
        for i in range(len(poses)):
            obstacles.append(Obstacle(0, 0, poses[i][0], 0, 0, poses[i][1], -100, 0.08, radius[i])) 
        
        self.obstacles = obstacles
        return 

    def random_init_obstacle(self, t, vehicle_x = 0, vehicle_y = -1, min_dist = 0.1):
        dist = -1
        x = y = -1
        while (dist < min_dist):
            x = random.uniform(FIELD_X_BOUNDS[0],FIELD_X_BOUNDS[1])
            y = random.uniform(FIELD_Y_BOUNDS[0],FIELD_Y_BOUNDS[1])
            dist = math.sqrt((vehicle_x-x)**2 + (vehicle_y-y)**2) # distance to car TODO: add a dist from static obs
        v_x = random.uniform(1e-3, 1e-2) * random.choice([1,-1])
        v_y = random.uniform(1e-3, 1e-2) * random.choice([1,-1])
        a_x = random.uniform(5e-6, 1e-4) * random.choice([1,-1])
        a_y = random.uniform(5e-6, 1e-4) * random.choice([1,-1])
        return Obstacle(a_x, v_x, x, a_y, v_y, y, t, 0.12, 0)
    
    def unsafe_obstacle_locations(self, t, cx, cy, min_dist):    
        unsafe_obstacles = []
        for i, obst in enumerate(self.obstacles):
            x, y =  obst.pos_update(t)
            v_x, v_y = obst.vel_update(t)
            if self.x_bounds[0] <= x <= self.x_bounds[1] and self.y_bounds[0] <= y <= self.y_bounds[1]:
                dist, ox, oy = l2([cx,cy], [x,y])
                unsafe_obstacle_info = [i, (ox,oy,v_x,v_y,obst.a_x,obst.a_y,obst.safety_dist+obst.r)]      
                if obst.r==0 and dist < min_dist:
                    unsafe_obstacles.append(unsafe_obstacle_info)
                elif (obst.r > 0 and dist < obst.r+obst.safety_dist):
                    unsafe_obstacles.append(unsafe_obstacle_info) 		
                #if (obst.r == 0 and dist < min_dist) or (obst.r > 0 and dist < obst.safety_dist+obst.r):
                #    unsafe_obstacles.append(unsafe_obstacle_info)
        return  unsafe_obstacles
    '''

    def unsafe_obstacle_locations(self, t, cx, cy, min_dist):    
        locs =  [ (i, a.x(t), a.y(t), a.v_x(t), a.v_y(t), a.a_x, a.a_y, a.safety_dist, a.radius)
                  for i,a in enumerate(self.obstacles)]
        unsafe_obstacles = []
        for i,x,y,x_v,y_v,x_a,y_a, safe_dist, r  in locs:
            if self.x_bounds[0] <= x <= self.x_bounds[1] and self.y_bounds[0] <= y <= self.y_bounds[1]:
                dist, ox, oy = l2([cx,cy], [x,y])
                if r==0 and dist < min_dist:
                    unsafe_obstacles.append([i,(ox,oy,x_v,y_v,x_a,y_a,safe_dist+r)])
                elif (r > 0 and dist < r+safe_dist):
                    unsafe_obstacles.append([i,(ox,oy,x_v,y_v,x_a,y_a,safe_dist+r)])
        return  unsafe_obstacles
    '''

    def obstacle_locations(self, t, vehicle_x, vehicle_y, min_dist):
        """
        Returns (i, x, y) tuples indicating that the i-th obstacle is at location (x,y).
        """
        locs = []
        for i, obst in enumerate(self.obstacles):
            x, y =  obst.pos_update(t)
            loc = (i, x, y, obst.r)
            if not (self.x_bounds[0] <= x <= self.x_bounds[1] and self.y_bounds[0] <= y <= self.y_bounds[1]):
                # out of bound, re-initialize 
                self.obstacles[i] = self.random_init_obstacle(t, vehicle_x, vehicle_y, min_dist)
            
            locs.append(loc)
        return locs
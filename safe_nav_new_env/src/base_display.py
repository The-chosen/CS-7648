class BaseDisplay():

    def setup(self, x_bounds, y_bounds,
              in_bounds, goal_bounds,
              margin, static_obs):
        pass
    
    def begin_time_step(self, t):
        pass

    def obstacle_at_loc(self, i, x, y):
        pass

    def obstacle_estimated_at_loc(self, i, x, y, is_match=False):
        pass

    def obstacle_estimates_compared(self, num_matched, num_total):
        pass

    def obstacle_set_color(self, id, color):
        pass

    def robot_at_loc(self, x, y, h, is_safe):
        pass

    def robot_steers(self, dh, dv):
        pass

    def collision(self):
        pass

    def out_of_bounds(self):
        pass

    def goal(self):
        pass

    def navigation_done(self, retcode, t):
        pass

    def estimation_done(self, retcode, t):
        pass
    
    def end_time_step(self, t):
        pass

    def teardown(self):
        pass

    def render(self):
        pass

    def _reset(self):
        pass
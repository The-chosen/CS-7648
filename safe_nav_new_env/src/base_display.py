class BaseDisplay():

    def setup(self, x_bounds, y_bounds,
                  in_bounds, goal_bounds,
                  margin, static_obstacles):
        pass
       
    def begin_time_step(self, t):
        pass

    def end_time_step(self, t):
        pass

    def obstacle_at_loc(self, i, x, y, nearest_obstacle = False, close_obstacle = False):
        pass

    def obstacle_set_color(self, i, color = 'grey' ):
        pass

    def robot_at_loc(self, x, y, h, is_ssa = False):
        pass

    def navigation_done(self, retcode, t):
        pass

    def render(self):
        pass


from builtins import object
class BoundsRectangle(object):
    def __init__(self, x_bounds, y_bounds):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def contains(self, point):
        [x, y] = point
        return self.x_bounds[1] > x >= self.x_bounds[0] \
                and self.y_bounds[1] > y >= self.y_bounds[0]
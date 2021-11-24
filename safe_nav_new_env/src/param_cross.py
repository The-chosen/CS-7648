params = {
  "in_bounds": {
    "x_bounds": [
      -1.0, 
      1.0
    ], 
    "y_bounds": [
      -1, 
      1
    ]
  }, 
  "noise_sigma": 0.03, 
  "min_dist": 0.03, 
  "_args": {
    "noise_sigma": 0.03, 
    "min_dist": 0.03, 
    "asteroid_b_max": 0.01, 
    "robot_max_speed": 0.03, 
    "t_step": 2, 
    "asteroid_a_max": 0.0001, 
    "t_future": 1000, 
    "t_past": -100
  }, 
  "initial_robot_state": {
    "x": 0.0, 
    "y": -1.0, 
    "vx": 0, 
    "vy": 0, 
    "max_speed": 0.02
  }, 
  "goal_bounds": {
    "x_bounds": [
      -1.0, 
      -0.8
    ], 
    "y_bounds": [
      -0.6, 
      0.6
    ]
  },
  'static_obstacles': {
      'pos': [[0.8, 0.7], [-0.8, 0.7], [0.8, -.7], [-0.8, -.7]],
      'radius': [0.2, 0.2, 0.2, 0.2]
  }
}

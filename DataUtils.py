import numpy as np
from itertools import product

def generate_square_mask(shape, min_len, max_len):
    
    mask = np.ones(shape = shape)
    
    y_len = np.random.randint(min_len, max_len)
    x_len = np.random.randint(min_len, max_len)
    
    box_upper_left = (np.random.randint(0, shape[0] - y_len + 1), np.random.randint(0, shape[1] - x_len + 1))
    
    box_x = (box_upper_left[1], box_upper_left[1] + x_len)
    box_y = (box_upper_left[0], box_upper_left[0] + y_len)
    
    mask[box_y[0]:box_y[1], box_x[0]:box_x[1]] = 0
    
    return mask

def generate_circle_mask(shape, min_radius, max_radius):
    
    y_len = shape[0]
    x_len = shape[1]
    
    mask = np.ones(shape = shape)
    
    r = np.random.randint(min_radius, max_radius)
    
    (c_y, c_x) = (np.random.randint(r, shape[0] - r + 1), np.random.randint(r, shape[1] - r + 1))
    
    points = np.array([[y, x] for [y,x] in product(range(y_len), range(x_len)) if (x - c_x)**2 + (y - c_y)**2 <= r**2])
    mask[points[:, 0], points[:, 1]] = 0
    
    return mask
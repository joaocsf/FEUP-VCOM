import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from math import sqrt
from math import acos
import math

# Calculate the vector between two points
def get_vector(p1, p2):
    return (p2[0] - p1[0] , p2[1] - p1[1])

def vector_add(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])

def vector_to_int(v):
    return (int(v[0]), int(v[1]))

def vector_mult(v1, m):
    return (v1[0]*m, v1[1]*m)

# Subtract Vectors
def vector_sub(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1])

# Calculate the mean point
def mid_point(p1, p2):
    return ((p2[0] + p1[0])/2, (p2[1] + p1[1])/2)

# Optimized to Compare Distances
def distance_sqr(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    v1, v2 = (x1-x2, y1-y2)
    return v1*v1 + v2*v2

# Compute the dot product
def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

# Compute the vector Length
def vector_length(v):
    x, y = v
    return sqrt(x*x + y*y)


# Compute the vector length without the sqr (Used only to compare values)
def vector_length_sqr(v):
    x, y = v
    return x*x + y*y

# Compute the Angle of a Vector
def vector_angle(vector):
    angle = math.atan2(vector[1], vector[0])

def vectors_angle(v1, v2):
    ab = dot_product(v1,v2)    
    norm_ab = vector_length(v1) * vector_length(v2)
    res = ab/norm_ab

    if(res > 1.0): res = 1.0

    if(norm_ab == 0): return math.pi

    return acos(res)
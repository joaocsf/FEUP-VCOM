import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *


class Finger:
    def __init__(self, rect, contour):
        
        self.rect = rect
        self.rect_points = cv.boxPoints(rect)
        self.rect_points = np.int0(self.rect_points)
        self.width = 0
        self.height = 0
        self.index = 0
        self.debug_lines = []
        self.is_thumb = 0
        self.calculate_tangents_and_normals(self.rect_points, contour)
    
    def insideBounds(self, midpoint, tangent, contour):
        p1 = vector_add(midpoint, tangent)
        p2 = vector_sub(midpoint, tangent)

        r1 = cv.pointPolygonTest(contour, p1, False)
        if(r1 != 1): return False

        r2 = cv.pointPolygonTest(contour, p2, False)
        if(r2 != 1): return False

        return True

    def calculate_tangents_and_normals(self, rect_points, contour):
        p1,p2,p3,p4 = rect_points
        v1 = get_vector(p1, p2)
        v2 = get_vector(p2, p3)

        length1 = vector_length(v1) 
        length2 = vector_length(v2) 

        self.width = min(length1, length2)
        self.height = max(length1, length2)

        if(length1 > self.width):
            self.tangent = v1
            self.normal = v2
            self.width_points = [p2,p3,p4,p1]
        else:
            self.tangent = v2
            self.normal = v1
            self.width_points = [p1,p2,p3,p4]

        pa,pb,pc,pd = self.width_points 
        mid = mid_point(pa,pb)
        tangent = vector_mult(self.tangent, 0.25)

        if not self.insideBounds(mid, tangent, contour):
            p1, p2, p3, p4 = self.width_points
            self.width_points = p3,p4, p1,p2

        pa,pb,pc,pd = self.width_points 
        mid = mid_point(pa,pb)

        mid = vector_to_int(mid)

        self.bottom = mid
        self.top = vector_to_int(mid_point(pc,pd))
        self.tangent = vector_sub(self.top, self.bottom)

        tangent = vector_mult(self.tangent, 0.25)
        add = vector_add(mid, tangent)
        add = vector_to_int(add)

        self.debug_lines.append([mid, add])
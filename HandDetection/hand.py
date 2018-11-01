import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *

class Hand:

    THUMB_ANGLE = math.pi*35/180
    Y_ANGLE = math.pi*40/180
    L_ANGLE = math.pi*45/180

    def __init__(self, contour, hull, center, defects, points, rect):
        self.contour = contour
        self.hull = hull
        self.center = center
        self.defects = defects
        self.points = points
        self.fingers = len(defects) + 1
        self.rect = rect
        self.finger_list = []

        x,y,w,h = rect

        self.thumb = None
        self.width = w
        self.height = h
        self.big_radius = 0
        self.debug_points = []
        self.top_left = (x,y)
        self.bottom_right = (x + w,y + h)
        self.debug_lines = []
        self.pose = None

        min_rect = cv.minAreaRect(contour)
        points = cv.boxPoints(min_rect)
        points = np.int0(points)
        self.min_rect = min_rect
        self.min_rect_points = points

        self.palm_rect_points = []
    
    def set_local_palm_rect(self, local_contour):
        min_rect = cv.minAreaRect(local_contour)
        points = cv.boxPoints(min_rect)
        self.palm_rect_points = np.int0(points)
        p1, p2, p3, p4 = self.palm_rect_points
        
        v1 = vector_sub(p2,p1)
        v2 = vector_sub(p3,p2)

        if(vector_length_sqr(v1) > vector_length_sqr(v2)):
            self.tangent = v1
            self.normal = v2
        else:
            self.tangent = v2
            self.normal = v1

        p0 = vector_add(self.center, self.tangent) 
        p1 = self.center 
        self.debug_lines.append([p0, p1])
    
    def setLocalCenter(self, local_center):
        self.local_center = local_center
        x,y = self.top_left
        lx, ly = local_center
        self.center = (x + lx, y + ly)
    
    def calculate_thumb(self):
        tangent = self.tangent
        inverse = vector_sub((0,0), tangent)

        if len(self.finger_list) == 0: return
        angle = 0
        thumb = self.finger_list[0]

        for finger in self.finger_list:
            f_tangent = finger.tangent
            a1 = vectors_angle(f_tangent, tangent)
            a2 = vectors_angle(f_tangent, inverse)
            min_angle = min(a1, a2)
            if(min_angle > angle):
                angle  = min_angle
                thumb = finger
        if(angle > Hand.THUMB_ANGLE):
            thumb.is_thumb = True
            self.thumb = thumb
    
    def sort_fingers_by_thumb(self):
        if self.thumb == None: return
        thumb_pos = self.thumb.bottom
        self.finger_list = sorted(self.finger_list, key=lambda finger: distance_sqr(finger.bottom, thumb_pos), reverse=False)
        i = 0
        for finger in self.finger_list:
            finger.index = i
            i += 1

    def remove_small_fingers(self):
        if len(self.finger_list) == 0: return
        height = 0
        for finger in self.finger_list:
            height += finger.height
        
        height /= len(self.finger_list)

        sigma = height * 0.4

        self.finger_list = [finger for finger in self.finger_list if finger.height > height or height - finger.height < sigma ]
        self.fingers = len(self.finger_list)

    def processHandPose(self):
        self.remove_small_fingers()
        self.calculate_thumb()
        self.sort_fingers_by_thumb()

        if self.is_y_pose():
            self.pose = 'Y'
        elif self.is_l_pose():
            self.pose = 'L'
        elif self.is_all_right_pose():
            self.pose = 'ALL_RIGHT'
        elif self.is_ily_pose():
            self.pose = 'ILY'
        elif self.is_i_pose():
            self.pose = 'I'
        elif self.is_v_pose():
            self.pose = 'V'
        

        print('Pose: {0}'.format(self.pose))
    
    def is_ily_pose(self):
        if len(self.finger_list) != 3: return
        if self.thumb == None: return 
        thumb = self.thumb
        pinky = self.finger_list[2]
        pinky_dist = 0
        indicator = self.finger_list[1]
        sqr_width = thumb.width * 3
        sqr_width *= sqr_width

        if distance_sqr(pinky.bottom, indicator.bottom) < sqr_width: return False
        
        return vectors_angle(pinky.tangent, thumb.tangent) > Hand.THUMB_ANGLE


    def is_all_right_pose(self):
        if len(self.finger_list) != 1: return
        if self.thumb == None: return 

        return True


    def is_l_pose(self):
        if len(self.finger_list) != 2: return
        if self.thumb == None: return 
        pointer = None

        pointer = self.finger_list[1]

        distance = distance_sqr(pointer.bottom, self.thumb.bottom)

        sqr_dist = self.thumb.width * 6
        sqr_dist *= sqr_dist
        
        return distance < sqr_dist and vectors_angle(pointer.tangent, self.thumb.tangent) > Hand.L_ANGLE    


    def is_y_pose(self):
        if len(self.finger_list) != 2: return
        if self.thumb == None: return 
        pinky = None

        pinky = self.finger_list[1]

        thumb_pinky = get_vector(self.thumb.bottom, pinky.bottom) 

        hand_normal = vector_normalize(self.normal)
        horizontal_distance = dot_product(thumb_pinky, hand_normal)

        distance_threshold = self.width/2
        print(horizontal_distance, distance_threshold)
        if abs(horizontal_distance) < distance_threshold: return

        distance = distance_sqr(pinky.bottom, self.thumb.bottom)

        sqr_dist = self.thumb.width *4
        sqr_dist *= sqr_dist

        return distance > sqr_dist and vectors_angle(pinky.tangent, self.thumb.tangent) > Hand.Y_ANGLE


    def is_v_pose(self):
        if len(self.finger_list) != 2: return
        if self.thumb != None: return 
        
        # V must have 2 fingers (no thumb) and they must be next to each other (1 defect)
        first,second = self.finger_list
        distance = first.width * 3
        distance *= distance
        return distance_sqr(first.bottom,second.bottom) < distance
        return len(self.defects) == 1


    def is_i_pose(self):
        if len(self.finger_list) != 1: return
        if self.thumb != None: return

        # Assuming I requires only 1 finger lifted
        return True
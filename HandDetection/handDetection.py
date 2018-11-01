import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import math
from math import sqrt
from math import acos
DEBUG = False
mouseX = 0
mouseY = 0
real_time = False
hue_range=[0,50]
saturation_range=[30,240]
value_range=[0,255]

sample_points = [
    [0.5,0.5],
    [0.4,0.5],
    [0.6,0.5],
    [0.5,0.4],
    [0.5,0.6],
]


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
        #self.debug_lines.append([(p3[0],p3[1]),(p4[0], p4[1])])

        #Exception(e)

class Hand:

    THUMB_ANGLE = math.pi*35/180
    Y_ANGLE = math.pi*45/180
    L_ANGLE = math.pi*75/180

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

        sqr_dist = self.thumb.width *6
        sqr_dist *= sqr_dist
        
        return distance < sqr_dist and vectors_angle(pointer.tangent, self.thumb.tangent) > Hand.L_ANGLE    


    def is_y_pose(self):
        if len(self.finger_list) != 2: return
        if self.thumb == None: return 
        pinky = None

        pinky = self.finger_list[1]

        distance = distance_sqr(pinky.bottom, self.thumb.bottom)

        sqr_dist = self.thumb.width *4
        sqr_dist *= sqr_dist

        return distance > sqr_dist and vectors_angle(pinky.tangent, self.thumb.tangent) > Hand.Y_ANGLE and vectors_angle(pinky.tangent, self.thumb.tangent) < Hand.L_ANGLE


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

def clustering(image):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    return res2

# Change the calculation to somehow use the average position of the point
def add_point_to_hand(handPoints, newPoint, minDistance=20):
    canAdd = True
    x2, y2 = newPoint
    points_found = []
    for point in handPoints:
        x1, y1 = point[0]
        v1, v2 = (x1-x2, y1-y2)
        distance = sqrt(v1 * v1 + v2 * v2)
        if distance < minDistance:
            points_found.append(point)
            break
    sum = newPoint
    count = 1
    for point in points_found:
        x1, y1 = point[0]
        sum = (sum[0] + x1, sum[1] + y1)
        count = count + 1
        handPoints = [p for p in handPoints if p[0][0] != x1 and p[0][1] != y1]

    m = [sum[0]/count, sum[1]/count]
    handPoints.append(np.array([m]))


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

# Compute Angle Between 3 Points Where PC is the Center Point
def angle(pc, p1, p2):
    v1 = (p1[0] - pc[0], p1[1] - pc[1])
    v2 = (p2[0] - pc[0], p2[1] - pc[1])
    return vectors_angle(v1, v2)
    
# Calculates the convexity Defects Filtering Unecessary Points
def compute_convexity_defects(contour, hull, threshold=90):
    length = 0.005*cv.arcLength(contour, True)
    lengthSQR = length * length
    thresh = math.pi * threshold / 180
    result = []

    if(cv.isContourConvex(contour)):
            return np.array(result)

    defects = cv.convexityDefects(contour, hull)
    for i in range(defects.shape[0]):
        defect = defects[i]
        s, e, f, d = defect[0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        mDist = max( [distance_sqr(start,far), distance_sqr(end,far)] )

        ang = angle(far, start, end)

        #Work around to filter lower convexity points
        if far[1] < start[1] or far[1] < end[1]:
            continue

        if(mDist >= lengthSQR and ang <= thresh):
            result.append(defect)
    
    result = np.array(result)
    return result

# Calculate the center of the hull
def calculate_center(hull):
    M = cv.moments(hull)
    x = int(M["m10"] / M["m00"]) 
    y = int(M["m01"] / M["m00"])

    return (x,y)

# Calculates the hull and the respective hand points (Change the calculation to use the point average)
def calculate_hand_points(contours):
    hands = []

    for contour in contours:
        handPoints = []
        length = 0.01*cv.arcLength(contour, True)
        hull = cv.convexHull(contour)
        rect = cv.boundingRect(contour)

        defects = compute_convexity_defects(contour, cv.convexHull(contour, returnPoints=False))

        for i in range(defects.shape[0]) :
            s, e, _, _ = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            add_point_to_hand(handPoints, start, length)
            add_point_to_hand(handPoints, end, length)

        # if(len(handPoints) == 0):
        #     continue

        handPoints = np.array(handPoints, np.int32)
        hand = Hand(
            contour,
            hull, 
            calculate_center(hull), 
            defects, 
            handPoints, 
            rect)
        hands.append(hand)

    return hands

# Filters the Contour Size to limit small objects
def filter_contour_size(contours, imageArea):
    final_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= imageArea/16:
            final_contours.append(contour)
    
    return np.array(final_contours)

def show_svg(event, x, y, flrags, param):
    global mouseX, mouseY
    if(event == cv.EVENT_LBUTTONDBLCLK):
        [mouseX, mouseY] = [x, y]

def findMaxCoords(matrix):
    (x, y) = np.unravel_index(matrix.argmax(), matrix.shape) 
    return (y, x)


#Calculates the distance transform of a MASK (useful to find the center of the hand later)
def calculateDistanceTransform(mask):
    distance_transform = cv.distanceTransform(mask, cv.DIST_L2, cv.DIST_MASK_PRECISE) 
    cv.normalize(distance_transform, distance_transform, 0, 1.0, cv.NORM_MINMAX)
    return distance_transform

def checkCircle(center, radius):
    x = radius - 1
    y = 0
    dx = 1
    dy = 1
    err = dx - (radius << 1)

# Creates a new kernel correctly centered
def calculateKernel(kernelType, dimension):
    if dimension & 2 == 1: dimension -= 1
    dimension_center = int(dimension/2)
    return cv.getStructuringElement(kernelType, (dimension + 1, dimension + 1), (dimension_center, dimension_center))

def calculate_fingers(hand, contours, handContour):
    for contour in contours:
        rect = cv.minAreaRect(contour)
        finger = Finger(rect, handContour)
        hand.finger_list.append(finger)
    
    hand.fingers = len(hand.finger_list)

def processHands(hands, mask):
    global DEBUG
    i = 0
    for hand in hands:

        # Correct Hand Fill
        height, width = mask.shape[:2]

        blankImage = np.zeros((height, width, 1), np.uint8)

        mask = cv.fillPoly(blankImage, [hand.contour], (255))
        h, w = mask.shape[:2]
        minimum = int( min(w,h) * 0.01)
        cv.rectangle(mask, (0,0), (w,h), 0, thickness=minimum)

        i += 1

        x1, y1 = hand.top_left
        x2, y2 = hand.bottom_right

        # Returns only the pixels related to our hand
        local_mask = mask[y1:y2, x1: x2]
        if DEBUG:
            cv.imshow("LocalMask {0}".format(i), local_mask)
        _ , handContours , _ = cv.findContours(local_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        hand_contour = max(handContours, key=lambda x: cv.contourArea(x))

        # Calculate the distance for each pixel to the closest 0
        distance_transform = cv.distanceTransform(local_mask, cv.DIST_L2, cv.DIST_MASK_PRECISE) 

        # Using the max value we can determine the radius of the inner circle (UNUSED)
        radius = np.amax(distance_transform)

        # Normalize the distance_transform to work with propotions instead of pixels
        cv.normalize(distance_transform, distance_transform, 0, 1.0, cv.NORM_MINMAX)


        # Set the local center of the hand (This sets the global center correctly)
        hand.setLocalCenter(findMaxCoords(distance_transform))
        hand.radius = radius

        # A global threshold of 1/4 of the hand distance seems to work just fine.
        lower_threshold = 0.25

        # Radius of the outer circle, can be used to sample and filter the number of fingers (UNUSED)
        hand.big_radius = int(lower_threshold * radius)

        # Calculation of the Kernel Size using the hand propotions
        dilation = int(lower_threshold * radius)


        #Calculate the threshold of the hand's palm 
        _, thresh = cv.threshold(distance_transform, lower_threshold, 1.0, cv.THRESH_BINARY)

        # Debuging the distance transform
        if DEBUG:
            cv.mshow("distance_transform {0}".format(i), distance_transform)

        # Normalize and convert the threshhold to uint8 (to later subtract)
        cv.normalize(thresh, thresh, 0, 255, cv.NORM_MINMAX)
        thresh = thresh.astype(np.uint8)
        if DEBUG:
            cv.imshow("thresh {0}".format(i), thresh)

        # Correctly Define the kernel to have expected expansions
        kernel = calculateKernel(cv.MORPH_ELLIPSE, dilation)

        #Apply opening Manualy with 4 times the dilation
        #The reason to erode is to get rid of small artifacts (parts of the fingers)
        #And then dilate the area as much as possible to obtain an reasonably large palm
        kernel = calculateKernel(cv.MORPH_ELLIPSE, dilation)
        thresh = cv.morphologyEx(thresh, cv.MORPH_ERODE, kernel, iterations=1)

        thresh = cv.morphologyEx(thresh, cv.MORPH_DILATE, kernel, iterations=4)

        #Debug the theshold at this point
        if DEBUG:
            cv.imshow("thresh2 {0}".format(i), thresh)

        #Calculate the palm rect
        _ , local_palm_contour , _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        local_palm_contour = max(local_palm_contour, key=lambda x: cv.contourArea(x))
        hand.set_local_palm_rect(local_palm_contour)

        #To find the fingers subtract the original mask to the palm's mask
        subtraction = cv.subtract(local_mask, thresh)

        #Calculate a smaller kernel to erode the fingers
        #This Erosion is useful to separate fingers that were to close to each other
        kernel = calculateKernel(cv.MORPH_ELLIPSE, int(dilation/3))

        subtraction = cv.erode(subtraction, kernel, iterations=1)

        #Debug the resulting fingers
        if DEBUG:
            cv.imshow("SubTraction {0}".format(i), subtraction)

        #Find the contour for each finger
        _, contours, _ = cv.findContours(subtraction, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 

        area = hand.width * hand.height

        #Some times (depending on the image quality) some artifacts may appear
        #This filter removes the contours with a porpotion less than the area/15
        #15 was found as a good value via trial and error

        contours = filter_contour_size(contours, area/15)

        calculate_fingers(hand, contours, hand_contour)

        #Sets the current number of fingers to the number of filtered contours
        hand.processHandPose()
        #hand.fingers = len(contours)
        print("FINGERS:")
        print(hand.fingers)
        


def drawResultsInImage(mask, image, hsvImage, hands):
    global mouseX, mouseY, real_time, sample_points

    # Draw Realtime Sample Points
    if real_time:
        h, w = image.shape[:2]
        for p in sample_points:
            px, py = p
            point = (int(px*w), int(py*h))
            cv.circle(image, point, 3, (0,255,255), 1)

    for hand in hands:

        # Draw the original contours and their respective hulls
        cv.drawContours(image, hand.contour, -1, (200, 200, 200), 2)
        cv.drawContours(image, hand.hull, -1, (0, 255, 0), 2)

        # Draw a point in each POI
        for point in hand.points:
            x,y = point[0]
            cv.circle(image, (x, y), 5, (0, 255, 0), thickness=5)
            cv.line(image, hand.center, (x,y), (226,194,65), 2)

        # Draw Enclosing Rect
        cv.rectangle(image, hand.top_left, hand.bottom_right, (0,255,0))

        # Draw Inside Circles 
        cv.circle(image, hand.center, hand.radius, (255,0,0), 5)
        cv.circle(image, hand.center, hand.big_radius, (128,0,0), 5)

        # Draw Hand's Debug Lines
        for line in hand.debug_lines:
            cv.line(image, line[0], line[1], (0,255,0), 3)

        # Draw Debug Points 
        for point in hand.debug_points:
            cv.circle(image, point, (3), (0,255,255), -1)

        # Draw Defects Found
        defects = hand.defects
        center = hand.center
        contour = hand.contour
        text = "Defects: {0} Estimated: {1}".format(len(defects), 1 + len(defects))
        cv.putText(image, text, (center[0] - 0, center[1] + 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            cv.circle(image, far, 5, (0,255,255), -1)
    
        # Draw Possible Center
        cv.circle(image, center, 10, (226,194,65), -1)

        (x,y) = hand.top_left
        # Draw Palm Rect
        palm_points = [ [[lx+x, ly+y]] for [lx,ly] in hand.palm_rect_points ]
        palm_points = np.array(palm_points)
        cv.drawContours(image, [palm_points], 0, (200,100,0), 5)

        # Draw Finger Rectangles
        for finger in hand.finger_list:
            offsetedPoints = [ [[lx+x, ly+y]] for [lx,ly] in finger.rect_points ]
            offsetedPoints = np.array(offsetedPoints)
            #print(finger.index)
            color = (255, finger.index * 51, 0) if not finger.is_thumb else (128,255,0)
            cv.drawContours(image, [offsetedPoints], 0, color, 2)
            top = vector_add(finger.top, (x,y))
            text = "Index: {0}".format(finger.index)
            cv.putText(image, text, top, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
            bottom = vector_add(finger.bottom, (x,y))
            cv.circle(image, top, 5, (0,0,255), -1)
            cv.circle(image, bottom, 5, (255,255,0), -1)
            for line in finger.debug_lines:
                p1 = tuple(map(sum, zip(line[0], (x,y))))
                p2 = tuple(map(sum, zip(line[1], (x,y))))
                cv.line(image, p1, p2, (0,255,0), 3)
        
        # Draw Hand Rect
        cv.drawContours(image, [hand.min_rect_points], 0, (255,0,0), 2)

    # Debug Tool To log the HSV Values
    if (mouseX < image.shape[0] and mouseX > 0 and mouseY > 0 and mouseY < image.shape[1]):
        h,s,v = hsvImage[mouseY, mouseX]
        value = "H:{0} S:{1} V:{2}".format(h,s,v)
        cv.putText(image, value, (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv.LINE_AA)
        cv.circle(image, (mouseX, mouseY), 10, (0,255,255), 10, cv.LINE_4)

    distance_transform = calculateDistanceTransform(mask)
    maxPoint = findMaxCoords(distance_transform)
    cv.circle(image, maxPoint, 10, (0,0,255))
    # Show each mask used
    image = cv.resize(image, (500, 500)) 
    mask = cv.resize(mask, (500, 500)) 
    cv.imshow("Hand", image)
    # cv.imshow("Mask", mask)
    
    # cv.imshow("DT", distance_transform)


    #while(cv.waitKey(0) != 27): continue
    #cv.imshow("HSV", hsvImage)

def calculate_range(value, offset, min_v, max_v):
    return (max(value - offset, min_v), min(value + offset, max_v))

# Sample points from the HSV image and calculates the mean of h,s,v  components
def calibrateHSV(image):
    global hue_range, saturation_range, value_range
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    height, width = image.shape[:2]

    h_offset = 25
    s_offset = 60
    v_offset = 100

    sum_h = 0
    sum_s = 0
    sum_v = 0

    for p in sample_points:
        px, py = p
        (x,y) = (int(px*width), int(py*height))
        h,s,v = hsv[y,x]
        sum_h += h
        sum_s += s
        sum_v += v
    
    sum_h /= len(sample_points)
    sum_s /= len(sample_points)
    sum_v /= len(sample_points)

    sum_h = int(sum_h)
    sum_s = int(sum_s)
    sum_v = int(sum_v)
    
    hue_range = calculate_range(sum_h, h_offset, 0, 255)
    saturation_range = calculate_range(sum_s, s_offset, 0, 255)
    value_range = calculate_range(sum_v, v_offset, 0, 255)

def processImage(image):
    global mouseX, mouseY, hue_range, saturation_range, value_range, real_time
    # Apply Gaussian blur
    h,w = image.shape[:2]
    minP = int(min(h,w) * 0.005)
    if(minP % 2 == 0): minP += 1
    #image = cv.GaussianBlur(image, (3, 3), 1)
    #GaussianBlur based on image porpotions
    image = cv.GaussianBlur(image, (minP, minP), 4)
    #bwImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    imageArea = image.shape[0] * image.shape[1]
    # print('Area:', imageArea)

 
    # Convert to HSV color-space
    hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #hsvImage = clustering(hsvImage)
    # (h, s, v) = cv.split(hsvImage)
    hsvImage = cv.pyrMeanShiftFiltering(hsvImage, 2, 25, maxLevel = 1)
    #(h, s, v) = cv.split(hsvImage)
    
    # h3 = cv.Sobel(h,cv.CV_8U, 1, 1, ksize = 3)
    #_, bwImage = cv.threshold(bwImage, 128, 255, type=cv.THRESH_BINARY_INV)
    # cv.normalize(h3,h3, 0, 255, cv.NORM_MINMAX)
    # h3 = np.uint8(h3)
    #h = cv.multiply(h, bwImage)
    #hsvImage = cv.merge([h,s,v])

    #cv.imshow("Sobel", hsvImage)
    # Create a black image, a window and bind the function to window
    cv.namedWindow('Hand')
    cv.setMouseCallback('Hand',show_svg)

    # Calculate the lower and upper HS values
    minH = 0
    maxH = 30

    minS = 7
    maxS = 250

    minH = 0
    maxH = 50
    minS = 255 * 0.21
    maxS = 255 * 0.68
    maxV = 255
    minV = 255 * 0
    lower = (minH,minS, minV)
    upper = (maxH, maxS, maxV)

    if(real_time):
        lower = (hue_range[0], saturation_range[0], value_range[0])
        upper = (hue_range[1], saturation_range[1], value_range[1])
    print(lower, upper)

    # Mask HS Values
    mask = cv.inRange(hsvImage, lower, upper)

    mask = cv.erode(mask, (minP,minP), iterations=2)
    mask = cv.dilate(mask, (minP,minP), iterations=3)

    # Find the contours of different hands
    ret, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours = filter_contour_size(contours, imageArea/2)

    # Calculate points closest to a Point of Interest
    hands = calculate_hand_points(contours)
    hands.sort(key=lambda x: x.center[0])

    processHands(hands, mask)

    drawResultsInImage(mask, image, hsvImage, hands)

    res = []
    for hand in hands:
        res.append(hand.fingers)
    
    return res

def testRealTime():
    global real_time
    real_time = True
    video = cv.VideoCapture(0)
    while(True):
        _, img = video.read()
        processImage(img)
        key = cv.waitKey(1)
        if key == 27:
            break
        elif key == ord('s'):
            calibrateHSV(img)

def process_file(path):
    image = cv.imread(path, cv.IMREAD_COLOR)
    processImage(image)
    while(cv.waitKey(0) != 27): continue

def test():
    process_file('data-set/hand-signs/all_right/1.jpg')
    process_file('data-set/hand-signs/all_right/2.jpg')
    process_file('data-set/hand-signs/I/0.png')
    process_file('data-set/hand-signs/I/1.png')
    process_file('data-set/hand-signs/ILY/0.png')
    process_file('data-set/hand-signs/ILY/1.png')
    process_file('data-set/hand-signs/L/1.png')
    process_file('data-set/hand-signs/L/2.png')
    process_file('data-set/hand-signs/v/0.png')
    process_file('data-set/hand-signs/v/1.png')
    process_file('data-set/hand-signs/Y/1.png')
    #image = cv.imread('data-set/one-hand/5/five_fingers2.jpg', cv.IMREAD_COLOR)
    #print(result)

    # hsvImage = cv.cvtColor(result, cv.COLOR_BGR2HSV)

    # plt.xlabel('Hue')
    # plt.ylabel('Saturation')
    # (h, s, v) = cv.split(hsvImage)
    # plt.scatter(h, s, label = "test")
    # plt.legend()
    # plt.show()

    while(cv.waitKey(0) != 27): continue

#testRealTime()
test()


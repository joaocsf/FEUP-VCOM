import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import math
from math import sqrt
from math import acos

mouseX = 0
mouseY = 0

class Hand:
    def __init__(self, contour, hull, center, defects, points, rect ):
        self.contour = contour
        self.hull = hull
        self.center = center
        self.defects = defects
        self.points = points
        self.fingers = len(defects) + 1
        self.rect = rect

        x,y,w,h = rect

        self.width = w
        self.height = h
        self.big_radius = 0
        self.debug_points = []
        self.top_left = (x,y)
        self.bottom_right = (x + w,y + h)
    
    def setLocalCenter(self, local_center):
        self.local_center = local_center
        x,y = self.top_left
        lx, ly = local_center
        self.center = (x + lx, y + ly)

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

# Optimized to Compare Distances
def distance_sqr(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    v1, v2 = (x1-x2, y1-y2)
    return v1*v1 + v2*v2

# Compute the dot product
def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

# Computer the vector Length
def vector_length(v):
    x, y = v
    return sqrt(x*x + y*y)

# Compute Angle Between 3 Points Where PC is the Center Point
def angle(pc, p1, p2):
    v1 = (p1[0] - pc[0], p1[1] - pc[1])
    v2 = (p2[0] - pc[0], p2[1] - pc[1])
    ab = dot_product(v1,v2)    
    norm_ab = vector_length(v1) * vector_length(v2)
    res = ab/norm_ab

    if(res > 1.0): res = 1.0

    if(norm_ab == 0): return math.pi

    return acos(res)
    
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

        if(len(handPoints) == 0):
            continue

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

def claculatePalmContour():
    

    pass

def processHands(hands, mask):
    i = 0
    for hand in hands:

        # Correct Hand Fill
        mask = cv.fillPoly(mask, [hand.contour], (255))
        h, w = mask.shape[:2]
        minimum = int( min(w,h) * 0.01)
        cv.rectangle(mask, (0,0), (w,h), 0, thickness=minimum)

        i += 1

        x1, y1 = hand.top_left
        x2, y2 = hand.bottom_right


        local_mask = mask[y1:y2, x1: x2]
        distance_transform = cv.distanceTransform(local_mask, cv.DIST_L2, cv.DIST_MASK_PRECISE) 
        radius = np.amax(distance_transform)
        cv.normalize(distance_transform, distance_transform, 0, 1.0, cv.NORM_MINMAX)


        hand.setLocalCenter(findMaxCoords(distance_transform))
        hand.radius = radius

        lower_threshold = 0.3
        hand.big_radius = int(1.4 * radius)
        dilation = int(lower_threshold * radius)
        #cv.circle(related_mask, hand.local_center, 10, (0), -1)
        _, thresh = cv.threshold(distance_transform, lower_threshold, 1.0, cv.THRESH_BINARY)

        cv.normalize(thresh, thresh, 0, 255, cv.NORM_MINMAX)
        thresh = thresh.astype(np.uint8)

        im2 = cv.bitwise_xor(local_mask, thresh)

        im2 = calculateDistanceTransform(im2)

        _, thresh = cv.threshold(im2, 0.5, 1.0, cv.THRESH_BINARY)
        cv.normalize(thresh, thresh, 0, 255, cv.NORM_MINMAX)
        thresh = thresh.astype(np.uint8)


        cv.imshow("mask {0}".format(i), thresh)
        cv.imshow("sub {0}".format(i), im2)

        _, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 

        area = hand.width * hand.height

        contours = filter_contour_size(contours, area/50)
        #hand.fingers = len(contours)
        print("FINGERS:")
        print(hand.fingers)
        


def drawResultsInImage(mask, image, hsvImage, hands):

    for hand in hands:

        # Draw the original contours and their respective hulls
        cv.drawContours(image, hand.contour, -1, (200, 200, 200), 2)
        cv.drawContours(image, hand.hull, -1, (0, 255, 0), 2)

        # Draw a point in each POI
        for point in hand.points:
            x,y = point[0]
            cv.circle(image, (x, y), 5, (0, 255, 0), thickness=5)
            cv.line(image, hand.center, (x,y), (226,194,65), 2)

        pprint(hand.rect)
        # Draw Enclosing Rect
        cv.rectangle(image, hand.top_left, hand.bottom_right, (0,255,0))

        print(hand.radius)
        # Draw Inside Circles 
        cv.circle(image, hand.center, hand.radius, (255,0,0), 5)
        cv.circle(image, hand.center, hand.big_radius, (128,0,0), 5)


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
    
        cv.circle(image, center, 10, (226,194,65), -1)

    # Debug Tool To log the HSV Values
    if (mouseX < image.shape[0] and mouseX > 0 and mouseY > 0 and mouseY < image.shape[1]):
        h,s,v = hsvImage[mouseY, mouseX]
        value = "H:{0} S:{1} V:{2}".format(h,s,v)
        print(value)
        cv.putText(image, value, (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv.LINE_AA)
        cv.circle(image, (mouseX, mouseY), 10, (0,255,255), 10, cv.LINE_4)

    distance_transform = calculateDistanceTransform(mask)
    maxPoint = findMaxCoords(distance_transform)
    cv.circle(image, maxPoint, 10, (0,0,255))
    # Show each mask used
    image = cv.resize(image, (500, 500)) 
    mask = cv.resize(mask, (500, 500)) 
    cv.imshow("Hand", image)
    cv.imshow("Mask", mask)
    
    cv.imshow("DT", distance_transform)


    while(cv.waitKey(0) != 27): continue
    #cv.imshow("HSV", hsvImage)
    

def processImage(image):
    global mouseX, mouseY
    # Apply Gaussian blur
    image = cv.GaussianBlur(image, (9, 9), 4)
    imageArea = image.shape[0] * image.shape[1]
    # print('Area:', imageArea)

 
    # Convert to HSV color-space
    hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #hsvImage = clustering(hsvImage)
    
    # Create a black image, a window and bind the function to window
    cv.namedWindow('Hand')
    cv.setMouseCallback('Hand',show_svg)

    # Calculate the lower and upper HS values
    minH = 0
    maxH = 30

    minS = 7
    maxS = 250

    # maxH = 50
    # minS = 50
    # maxS = 255 * 0.68
    minH = 0
    maxH = 50

    minS = 255 * 0.21
    maxS = 255 * 0.68
    lower = (minH,minS, 0)
    upper = (maxH, maxS, 255)

    # Mask HS Values
    mask = cv.inRange(hsvImage, lower, upper)

    mask = cv.erode(mask, (3,3), iterations=2)
    mask = cv.dilate(mask, (3,3), iterations=3)

    # Find the contours of different hands
    ret, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours = filter_contour_size(contours, imageArea/2)

    # Calculate points closest to a Point of Interest
    hands = calculate_hand_points(contours)

    processHands(hands, mask)

    drawResultsInImage(mask, image, hsvImage, hands)

    res = []
    for hand in hands:
        res.append(hand.fingers)
    
    return res

def testRealTime():
    video = cv.VideoCapture(0)
    while(True):
        _, img = video.read()
        processImage(img)

        if(cv.waitKey(1) == 27):
            break

def test():
    image = cv.imread('hands2.jpg', cv.IMREAD_COLOR)
    result = processImage(image)
    print(result)

    # hsvImage = cv.cvtColor(result, cv.COLOR_BGR2HSV)

    # plt.xlabel('Hue')
    # plt.ylabel('Saturation')
    # (h, s, v) = cv.split(hsvImage)
    # plt.scatter(h, s, label = "test")
    # plt.legend()
    # plt.show()

    while(cv.waitKey(0) != 27): continue

#test()


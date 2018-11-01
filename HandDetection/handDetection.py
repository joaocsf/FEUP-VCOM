import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import math
from math import sqrt
from math import acos
from hand import *
from finger import *
from globals import *


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
    process_file('data-set/hand-signs/I/1.jpg')
    process_file('data-set/hand-signs/ILY/0.png')
    process_file('data-set/hand-signs/ILY/1.png')
    process_file('data-set/hand-signs/L/1.png')
    process_file('data-set/hand-signs/L/2.png')
    process_file('data-set/hand-signs/v/0.png')
    process_file('data-set/hand-signs/v/1.png')
    process_file('data-set/hand-signs/Y/1.png')

    while(cv.waitKey(0) != 27): continue

#testRealTime()
test()


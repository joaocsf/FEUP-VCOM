import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import math

from math import sqrt
from math import acos

mouseX = 0
mouseY = 0

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
def compute_convexity_defects(contour, hull, threshold=80):
    length = 0.01*cv.arcLength(contour, True)
    lengthSQR = length * length
    thresh = math.pi * threshold / 180
    defects = cv.convexityDefects(contour, hull)
    result = []
    for i in range(defects.shape[0]):
        defect = defects[i]
        s, e, f, d = defect[0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        mDist = max( [distance_sqr(start,far), distance_sqr(end,far)] )

        ang = angle(far, start, end)

        if(mDist >= lengthSQR and ang < thresh):
            result.append(defect)
    
    result = np.array(result)
    return result

# Calculate the center of the gull
def calculate_center(hull):
    M = cv.moments(hull)
    x = int(M["m10"] / M["m00"]) 
    y = int(M["m01"] / M["m00"])

    return (x,y)

# Calculates the hull and the respective hand points (Change the calculation to use the point average)
def calculate_hand_points(contours):
    hands = []
    hull_list = []
    defect_list = []
    centers = []
    used_contour = []
    for contour in contours:
        handPoints = []
        length = 0.01*cv.arcLength(contour, True)
        hull = cv.convexHull(contour)

        defects = compute_convexity_defects(contour, cv.convexHull(contour, returnPoints=False))

        for i in range(defects.shape[0]) :
            s, e, _, _ = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            add_point_to_hand(handPoints, start, length)
            add_point_to_hand(handPoints, end, length)

        if(len(handPoints) == 0):
            continue
        
        used_contour.append(contour)
        hull_list.append(hull)
        centers.append(calculate_center(hull))
        defect_list.append(defects)
        handPoints = np.array(handPoints, np.int32)
        hands.append(handPoints)

    return (hands, hull_list, defect_list, centers, used_contour)

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

def nothing(x):
    pass

def processImage(image):
    global mouseX, mouseY
    # Apply Gaussian blur
    image = cv.GaussianBlur(image, (5, 5), 5)
    imageArea = image.shape[0] * image.shape[1]
    print('Area:', imageArea)

 
    # Convert to HSV color-space
    hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #hsvImage = clustering(hsvImage)
    
    # Create a black image, a window and bind the function to window
    cv.namedWindow('Hand')
    cv.setMouseCallback('Hand',show_svg)

    # Calculate the lower and upper HS values
    minH = 0
    maxH = 70

    minS = 100
    maxS = 250

    # minH = cv.getTrackbarPos('minH','Hand')
    # minH = cv.getTrackbarPos('minH','Hand')
    # maxH = cv.getTrackbarPos('maxH','Hand')
    # minS = cv.getTrackbarPos('minS','Hand')
    # maxS = cv.getTrackbarPos('maxS','Hand')

    lower = (minH,minS, 0)
    upper = (maxH, maxS, 255)

    

    # Mask HS Values
    mask = cv.inRange(hsvImage, lower, upper)

    #cv.erode(mask, (5,5), iterations=2)
    cv.dilate(mask, (5,5), iterations=3)

    # Find the contours of different hands
    ret, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contours = filter_contour_size(contours, imageArea)

    # Calculate points closest to a Point of Interest
    hands, hull_list, defect_list, centers, contours = calculate_hand_points(contours)

    # Draw the original contours and their respective hulls
    cv.drawContours(image, contours, -1, (255, 0, 0), 2)
    cv.drawContours(image, hull_list, -1, (0, 0, 255), 2)

    # Draw the convex hand shape using points of interest
    if(len(hands) != 0):
        cv.drawContours(image, hands, -1, (200, 200, 200), 2)

    # Draw a point in each POI
    for index, hand in enumerate(hands):
        center = centers[index]
        for point in hand:
            x,y = point[0]
            cv.circle(image, (x, y), 5, (0, 255, 0), thickness=5)
            cv.line(image, center, (x,y), (226,194,65), 2)
    
    # Draw Defects Found
    for index, defects in enumerate(defect_list):
        print(index)
        center = centers[index]
        contour = contours[index]
        text = "Defects: {0} Estimated: {1}".format(len(defects), 1 + len(defects))
        cv.putText(image, text, (center[0] - 0, center[1] + 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            cv.circle(image, far, 5, (0,255,255), -1)
    
    for center in centers:
        cv.circle(image, center, 10, (226,194,65), -1)

    # Debug Tool To log the HSV Values
    if (mouseX < image.shape[0] and mouseX > 0 and mouseY > 0 and mouseY < image.shape[1]):
        h,s,v = hsvImage[mouseY, mouseX]
        value = "H:{0} S:{1} V:{2}".format(h,s,v)
        print(value)
        cv.putText(image, value, (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv.LINE_AA)
        cv.circle(image, (mouseX, mouseY), 10, (0,255,255), 10, cv.LINE_4)

    # Show each mask used
    cv.imshow("Hand", image)
    cv.imshow("HSV", hsvImage)
    cv.imshow("Mask", mask)
    return image

def findHSVValues(image):
    image = cv.GaussianBlur(image, (5,5), 2) 
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    color = ('b', 'r')

    for i, col in enumerate(color):
        hist = cv.calcHist([hsv_image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0,256])
    
    plt.show()


def test():
    # cv.namedWindow('Hand')
    # cv.createTrackbar('minH','Hand',0,255,nothing)
    # cv.createTrackbar('maxH','Hand',0,255,nothing)
    # cv.createTrackbar('minS','Hand',0,255,nothing)
    # cv.createTrackbar('maxS','Hand',0,255,nothing)


    video = cv.VideoCapture(0)
    while(True):
        _, img = video.read()
        # processImage(img)
        findHSVValues(img)

        if(cv.waitKey(10000) == 27):
            break

    # Image read
    image = cv.imread('hands.jpg', cv.IMREAD_COLOR)

    result = processImage(image)

    hsvImage = cv.cvtColor(result, cv.COLOR_BGR2HSV)

    # plt.xlabel('Hue') 
    # plt.ylabel('Saturation') 
    # (h, s, v) = cv.split(hsvImage)
    # plt.scatter(h, s, label = "test")
    # plt.legend()
    # plt.show()
# 
 #
test()

while(cv.waitKey(0) != 27): continue
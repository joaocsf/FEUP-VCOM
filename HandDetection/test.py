import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pprint
import math

from math import sqrt


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
def addPointToHand(handPoints, newPoint, minDistance=20):
    canAdd = True
    x2, y2 = newPoint[0]
    for point in handPoints:
        x1, y1 = point[0]
        v1, v2 = (x1-x2, y1-y2)
        distance = sqrt(v1 * v1 + v2 * v2)
        if distance < minDistance:
            canAdd = False
            break

    if canAdd:
        handPoints.append(newPoint)

# Calculates the hull and the respective hand points (Change the calculation to use the point average)
def calculateHandPoints(contours):
    hands = []
    hull_list = []
    for contour in contours:
        handPoints = []
        hull = cv.convexHull(contour)
        hull_list.append(hull)
        for point in hull:
            addPointToHand(handPoints, point, 100)
        handPoints = np.array(handPoints)
        hands.append(handPoints)

    return (hands, hull_list)


def processImage(image):
    # Apply Gaussian blur
    image = cv.GaussianBlur(image, (5, 5), 1)

    # image = clustering(image)
    # Convert to HSV color-space
    hsvImage = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # Calculate the lower and upper HS values
    minH = 90
    maxH = 120

    minS = 20
    maxS = 150

    lower = (minH,minS, 0)
    upper = (maxH, maxS, 255)

    # Mask HS Values
    mask = cv.inRange(hsvImage, lower, upper)

    # cv.erode(mask, (5,5))
    # cv.dilate(mask, (5,5))

    # Find the contours of different hands
    ret, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    pp = pprint.pprint
    pprint.pformat(pprint.PrettyPrinter, indent=4, depth=20)
    pp(contours)

    # Calculate points closest to a POI
    hands, hull_list = calculateHandPoints(contours)

    # Draw a point in each POI
    for hand in hands:
        for point in hand:
            x,y = point[0]
            cv.circle(image, (x, y), 5, (0, 255, 0), thickness=5)

    # Draw the original contours and their respective hulls
    cv.drawContours(image, contours, -1, (255, 0, 0), 2)
    cv.drawContours(image, hull_list, -1, (0, 0, 255), 2)

    # Draw the convex hand shape using POIs
    cv.drawContours(image, np.array(hands), -1, (0, 255, 255), 2)

    # Show each mask used
    cv.imshow("Hand", image)
    cv.imshow("HSV", hsvImage)
    cv.imshow("Mask", mask)
    return image



def processImage1(image):
    # Apply Gaussian blur
    image = cv.GaussianBlur(image, (3, 3), 0)

    # Convert to HSV color-space
    hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Calculate the lower and upper HS values
    minH = 0
    maxH = 30

    minS = 0
    maxS = 255

    lower = (minH, minS, 0)
    upper = (maxH, maxS, 255)

    # Mask HS Values to create a binary image
    mask = cv.inRange(hsvImage, lower, upper)

    # Morphological transformations to filter
    erode = cv.erode(mask, np.ones((5, 5)), iterations=1)
    dilation = cv.dilate(erode, np.ones((5, 5)), iterations=1)

    filtered = cv.GaussianBlur(dilation, (5, 5), 0)
    ret, threshold = cv.threshold(filtered, 120, 255, 0)

    # Find the contours
    ret, contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Find contour with maximum area
    contour = max(contours, key=lambda x: cv.contourArea(x))

    # Create bounding rectangle around the contour
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # Find convex hull
    hull = cv.convexHull(contour)

    # Draw contour
    drawing = np.zeros(image.shape, np.uint8)
    cv.drawContours(drawing, [contour], -1, (0, 255, ), 0)
    cv.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

    # Find convexity defects
    hull = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull)

    count_defects = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14

        # if angle > 90 draw a circle at the far point
        if angle <= 90:
            count_defects += 1
            cv.circle(image, far, 1, [0, 0, 255], -1)

        cv.line(image, start, end, [0, 255, 0], 2)

    if count_defects == 0:  # Differentiate when is a 0 or 1
        print(1)
    elif count_defects == 1:
        print(2)
    elif count_defects == 2:
        print(3)
    elif count_defects == 3:
        print(4)
    elif count_defects == 4:
        print(5)
    else:
        pass


    """ret, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    pp = pprint.pprint
    pprint.pformat(pprint.PrettyPrinter, indent=4, depth=20)
    pp(contours)

    # Calculate points closest to a POI
    hands, hull_list = calculateHandPoints(contours)

    # Draw a point in each POI
    for hand in hands:
        for point in hand:
            x,y = point[0]
            cv.circle(image, (x, y), 5, (0, 255, 0), thickness=5)

    # Draw the original contours and their respective hulls
    cv.drawContours(image, contours, -1, (255, 0, 0), 2)
    cv.drawContours(image, hull_list, -1, (0, 0, 255), 2)

    # Draw the convex hand shape using POIs
    cv.drawContours(image, np.array(hands), -1, (0, 255, 255), 2)
"""
    # Show each mask used
    cv.imshow("Hand", image)
    cv.imshow("HSV", hsvImage)
    cv.imshow("Mask", mask)
    cv.imshow("Thresholded", threshold)


def test():
    # video = cv.VideoCapture(0)
    # while(True):
    #     rect, img = video.read()
    #     processImage(img)

    #     if(cv.waitKey(1) == 27):
    #         break

    # Image read
    image = cv.imread('../handsPosesDataSet/3/3.jpg', cv.IMREAD_COLOR)

    processImage1(image)

    #result = processImage(image)

    # hsvImage = cv.cvtColor(result, cv.COLOR_RGB2HSV)

    # plt.xlabel('Hue')
    # plt.ylabel('Saturation')
    # (h, s, v) = cv.split(hsvImage)
    # plt.scatter(h, s, label = "test")
    # plt.legend()
    # plt.show()


test()

while(cv.waitKey(0) != 27): continue
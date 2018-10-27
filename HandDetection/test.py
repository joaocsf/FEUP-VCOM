import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pprint
import math

def processImage(image):
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

    # Show each mask used
    cv.imshow("Hand", image)
    while(cv.waitKey(0) != 27): continue
    #cv.imshow("HSV", hsvImage)
    #cv.imshow("Mask", mask)
    #cv.imshow("Thresholded", threshold)
    num_fingers = count_defects + 1
    hands = []
    hands.append(num_fingers)
    return hands

def test():
    image = cv.imread('hands.jpg', cv.IMREAD_COLOR)
    result = processImage(image)
    print(result)
    while(cv.waitKey(0) != 27): continue

#test()


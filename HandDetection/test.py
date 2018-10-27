import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pprint
import math

def processImage(image):
    # Apply Gaussian blur
    image = cv.GaussianBlur(image, (9, 9), 4)

    # Convert to HSV color-space
    hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Calculate the lower and upper HS values
    minH = 0
    maxH = 50

    minS = 255 * 0.23
    maxS = 255 * 0.68

    lower = (minH, minS, 0)
    upper = (maxH, maxS, 255)

    # Mask HS Values to create a binary image
    mask = cv.inRange(hsvImage, lower, upper)

    # Morphological transformations to filter
    erode = cv.erode(mask, np.ones((5, 5)), iterations=2)
    dilation = cv.dilate(erode, np.ones((5, 5)), iterations=3)

    #filtered = cv.GaussianBlur(dilation, (5, 5), 0)
    #ret, threshold = cv.threshold(filtered, 120, 255, 0)

    # Find the contours
    ret, contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

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

        cv.line(image, start, end, [0, 255, 0], 2)

        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14

        # far point is below the start and end points
        if far[1] < start[1] or far[1] < end[1]:
            continue
        
        # if angle > 90 draw a circle at the far point
        if angle <= 90:
            count_defects += 1
            print(start, end, far)
            cv.circle(image, far, 5, [0, 0, 255], -1)
            cv.circle(image, end, 5, [255, 0, 255], -1)
            cv.circle(image, start, 5, [255, 0, 255], -1)

    # Show each mask used
    image = cv.resize(image, (500, 500)) 
    mask = cv.resize(mask, (500, 500)) 
    cv.imshow("Hand", image)
    cv.imshow("Mask", mask)
    #cv.imshow("HSV", hsvImage)
    #cv.imshow("Thresholded", threshold)
    while(cv.waitKey(0) != 27): continue
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


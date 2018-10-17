import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pprint

from math import sqrt

def clustering(image):
    Z = image.reshape((-1,3))
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
    image = cv.GaussianBlur(image, (5,5), 1)
    #image = clustering(image)    
    hsvImage = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    #Calculate the lower and upper HS values
    minH = 90
    maxH = 120

    minS = 20
    maxS = 150

    lower = (minH,minS, 0)
    upper = (maxH, maxS, 255)

    #Mask HS Values
    mask = cv.inRange(hsvImage, lower, upper)

    #cv.erode(mask, (5,5))
    # cv.dilate(mask, (5,5))

    #Find the contours of different hands
    ret, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    pp = pprint.pprint
    pprint.pformat(pprint.PrettyPrinter, indent=4, depth=20)
    pp(contours)

    #Calculate points closest to a POI
    hands, hull_list = calculateHandPoints(contours)

    #Draw a point in each POI
    for hand in hands:
        for point in hand:
            x,y = point[0]
            cv.circle(image, (x,y), 5, (0,255,0), thickness=5)
 
    #Draw the original contours and their respective hulls
    cv.drawContours(image, contours, -1, (255,0,0), 2)
    cv.drawContours(image, hull_list, -1, (0,0,255), 2)

    #Draw the convex hand shape using POIs
    cv.drawContours(image, np.array(hands), -1, (0,255,255), 2)

    #Show each mask used
    cv.imshow("Hand", image)
    cv.imshow("HSV", hsvImage)
    cv.imshow("Mask", mask)
    return image

 

def test():
    # video = cv.VideoCapture(0)
    # while(True):
    #     rect, img = video.read()
    #     processImage(img)

    #     if(cv.waitKey(1) == 27):
    #         break

    image = cv.imread('hands.jpg', cv.IMREAD_COLOR)

    image = cv.GaussianBlur(image, (5,5), 1)
    result = processImage(image)

    # hsvImage = cv.cvtColor(result, cv.COLOR_RGB2HSV)

    # plt.xlabel('Hue')
    # plt.ylabel('Saturation')
    # (h, s, v) = cv.split(hsvImage)
    # plt.scatter(h, s, label = "test")
    # plt.legend()
    # plt.show()


test()

while(cv.waitKey(0) != 27): continue
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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

def processImage(image):
    image = cv.GaussianBlur(image, (5,5), 1)
    image = clustering(image)    
    hsvImage = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    maxH = 125
    minH = 80

    minS = 20
    maxS = 150



    lower = (minH,minS, 0)

    upper = (maxH, maxS, 255)
    mask = cv.inRange(hsvImage, lower, upper)


    contours= cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    hull_list = []
    for contour in contours[1]:
        hull = cv.convexHull(contour)
        hull_list.append(hull)


    cv.drawContours(image, contours[1], -1, (255,0,0), 2)
    cv.drawContours(image, hull_list, -1, (0,0,255), 2)
#    cv.erode(mask, (5,5))
    # cv.dilate(mask, (5,5))

    cv.imshow("Hand", image)
    cv.imshow("HSV", hsvImage)
    cv.imshow("Window", mask)
    return image

 

def test():
    video = cv.VideoCapture(0)
    while(True):
        rect, img = video.read()
        processImage(img)

        if(cv.waitKey(1) == 27):
            break

    image = cv.imread('hands.jpg', cv.IMREAD_COLOR)

    image = cv.GaussianBlur(image, (5,5), 1)
    result = processImage(image)

    hsvImage = cv.cvtColor(result, cv.COLOR_BGR2HSV)

    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    (h, s, v) = cv.split(hsvImage)
    plt.scatter(h, s, label = "test")
    plt.legend()
    plt.show()


test()

while(cv.waitKey(0) != 27): continue
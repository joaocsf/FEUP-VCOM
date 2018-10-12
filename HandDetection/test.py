import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def processImage(image):
    image = cv.GaussianBlur(image, (5,5), 1)
    hsvImage = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    maxH = 125
    minH = 80

    minS = 40
    maxS = 150

    


    lower = (minH,minS, 0)

    upper = (maxH, maxS, 255)
    mask = cv.inRange(hsvImage, lower, upper)

    contours= cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


    cv.drawContours(image, contours[1], -1, (255,0,0), 2)
#    cv.erode(mask, (5,5))
    # cv.dilate(mask, (5,5))

    cv.imshow("Hand", image)
    cv.imshow("Window", mask)

 

def test():
    video = cv.VideoCapture(0)
    while(True):
        rect, img = video.read()
        processImage(img)

        if(cv.waitKey(1) == 27):
            break

    image = cv.imread('hand.jpg', cv.IMREAD_COLOR)

    image = cv.GaussianBlur(image, (5,5), 1)
    processImage(image)
    #plt.xlabel('Hue')
    #plt.ylabel('Saturation')
    #(h, s, v) = cv.split(hsvImage)
    #plt.scatter(h, s, label = "test")
    #plt.legend()
    #plt.show()


test()

while(cv.waitKey(0) != 27): continue
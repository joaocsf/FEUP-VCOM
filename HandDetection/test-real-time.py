import cv2 as cv
import numpy as np
import copy
import math
from handDetection import processImageRealTime

cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv.erode(fgmask, kernel, iterations=1)
    res = cv.bitwise_and(frame, frame, mask=fgmask)
    return res

# Camera
camera = cv.VideoCapture(0)
camera.set(10,200)

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv.flip(frame, 1)  # flip the frame horizontally
    cv.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv.imshow('mask', img)

        # convert the image into binary image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv.imshow('blur', blur)
        ret, thresh = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)
        cv.imshow('ori', thresh)

        res = processImageRealTime(thresh)
        print('Result: ', res)

    k = cv.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    elif k == ord('b'):
        bgModel = cv.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( 'Background Model created.')
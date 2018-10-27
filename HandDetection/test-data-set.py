import os
import cv2 as cv
from handDetection import processImage as processImg1
from test import processImage as processImg2

def testOneHandImg(pathImg):
    image = cv.imread(pathImg, cv.IMREAD_COLOR)
    hands = processImg2(image) # CHANGE FUNCTION HERE!!!
    
    if len(hands) != 1:
        print('Error: found', len(hands), "hands, when only exists one hand", pathImg)
        return

    fingersFound = hands[0];
    fingersAns = int(pathImg[9])
    
    if fingersFound != fingersAns:
        print('Error: found', fingersFound, "fingers, when exists", fingersAns, "fingers.", pathImg)
        return
    
    print('Success:', pathImg)

def testAllImgs():
    rootDir = 'data-set/'
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            pathFile = dirName + '/' + fname
            testOneHandImg(pathFile)

testAllImgs()
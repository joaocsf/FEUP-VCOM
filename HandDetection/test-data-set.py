import os
import traceback
import logging
import cv2 as cv
from handDetection import processImage as processImg1
from test import processImage as processImg2

def testOneHandImg(pathImg):
    image = cv.imread(pathImg, cv.IMREAD_COLOR)
    hands = processImg2(image) # CHANGE FUNCTION HERE!!!
    
    if len(hands) != 1:
        print('Error: found', len(hands), "hands, when only exists one hand", pathImg)
        return -1

    fingersFound = hands[0];
    fingersAns = int(pathImg[18])
    
    if fingersFound != fingersAns:
        print('Error: found', fingersFound, "fingers, when exists", fingersAns, "fingers.", pathImg)
        return -1
    
    print('Success:', pathImg)
    return 0


def testAllImgs():
    totalTests = 0;
    correctAns = 0;
    rootDir = 'data-set/'
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            totalTests += 1
            pathFile = dirName + '/' + fname
            print('----------------------------------------------------')
            try:
                if testOneHandImg(pathFile) == 0:
                    correctAns += 1
            except:
                logging.error(traceback.format_exc())
    
    print('***************')
    print('Total tests:', totalTests)
    print('Correct answers:', correctAns)
    print('Success ratio:', correctAns / totalTests)
    print('***************')        

testAllImgs()
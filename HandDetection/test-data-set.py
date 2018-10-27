import os
import traceback
import logging
import cv2 as cv
from handDetection import processImage as processImg1
from test import processImage as processImg2

def testOneHandImg(pathImg):
    image = cv.imread(pathImg, cv.IMREAD_COLOR)
    hands = processImg1(image) # CHANGE FUNCTION HERE!!!
    
    if len(hands) != 1:
        print('Error: found', len(hands), "hands, when only exists one hand")
        return -1

    fingersFound = hands[0]
    fingersAns = int(pathImg[18])
    
    if fingersFound != fingersAns:
        print('Error: found', fingersFound, "fingers, when exists", fingersAns, "fingers.")
        return -1
    
    print('Success')
    return 0


def testAllImgs():
    totalTests = 0;
    correctAns = 0;
    rootDir = 'data-set/'
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            pathFile = dirName + '/' + fname
            if pathFile[18] == '0' or pathFile[18] == '1':
                continue
            totalTests += 1
            print('----------------------------------------------------')
            print(pathFile)
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

# 0.65 (test), 0.85 (handDetection)
import os
import traceback
import logging
import cv2 as cv
from handDetection import processImage as processImg1

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

def testManyHandsImg(pathImg):
    image = cv.imread(pathImg, cv.IMREAD_COLOR)
    hands = processImg1(image) # CHANGE FUNCTION HERE!!!
    print(hands)
    return 0

def testAllManyHandsImgs():
    rootDir = 'data-set/multiple-hands'
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            pathFile = dirName + '/' + fname
            print('----------------------------------------------------')
            print(pathFile)
            try:
                testManyHandsImg(pathFile)
            except:
                logging.error(traceback.format_exc())


def testAllOneHandImgs():
    totalTests = 0;
    correctAns = 0;
    rootDir = 'data-set/one-hand'
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            pathFile = dirName + '/' + fname
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

testAllManyHandsImgs()

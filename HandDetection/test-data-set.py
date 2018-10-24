import os

# TODO: import some function that do this
def getNumFingersOfImg(pathImg):
    return 1;

def testImg(pathImg):
    fingersFound = getNumFingersOfImg(pathImg)
    fingersAns = int(pathImg[9])
    if fingersFound != fingersAns:
        print('Error: found', fingersFound, "fingers, when exists", fingersAns, "fingers.", pathImg)
    else:
        print('Success:', pathImg)

def testAllImgs():
    rootDir = 'data-set/'
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            pathFile = dirName + '/' + fname
            testImg(pathFile)

testAllImgs()
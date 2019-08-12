# import the necessary packages
import pickle
import re
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)

import numpy
from imutils import paths
from multiprocessing import Pool
from collections import namedtuple
from skimage import data, io
import argparse
from exif import Image
from datetime import datetime
import cv2
import sys

blurryOrNot = {}
threshold = 8
PictureStruct = namedtuple("PictureStruct", "blurryness path date_taken color sunset_metric")

# added regex filter to cut down the number of images for testing
#photos_ = [x for x in paths.list_images("/media/hamish/Elements/photos/photos/") if re.search("349\d\d\d", x)]#
currentNumber = 350
photos_ = [x for x in paths.list_images("/media/hamish/Elements/photos/photos/") if re.search(repr(currentNumber)+"\d\d\d", x)]
#TODO: make this take off a 1000 image chunk at a time

#photos_ = [x for x in paths.list_images("/media/hamish/Elements/photos/photos/") if re.search("34911\d", x)]
#photos_ = [x for x in paths.list_images("/media/hamish/Elements/photos/photos/") if re.search("3520\d\d", x)]#

#photos_ = [x for x in paths.list_images("/media/hamish/Elements/photos/photos/") ]

middle_lines = []

debug = False
addDebugInfoToImages = False
loadDataFromExistingFile = False

#todo refactor this so it's using the same algorithim for analysis and actually processing
def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sortingFunction(pictureStruct: PictureStruct):
    return pictureStruct.date_taken


def doit():
    # loop over the input images

    #serial impl (for debugging)
    # listOfPhotosWithBlurryness = sorted(
    #     filter(None, [determineIfBlurry(photo) for photo in photos_]
    #            )
    #     , key=sortingFunction)
    if(loadDataFromExistingFile):
        debugLog("loading from existing blurryness data file")
        with open("/home/hamish/buffer/blurryness_data", 'rb') as data_backup: listOfPhotosWithBlurryness =  pickle.load(data_backup)
    else:
        debugLog("analysing")
        with Pool(16) as p:
            listOfPhotosWithBlurryness = sorted(filter(None,p.map(determineIfBlurry, photos_)),key=sortingFunction )
        debugLog("removing blurry photos and processing")
    # with Pool(16) as p:
    non_blurry_image_paths = [thresholdImages(index, listOfPhotosWithBlurryness) for index in range(len(listOfPhotosWithBlurryness))]
    if(not loadDataFromExistingFile):
        with open("./blurryness_data", 'wb') as data_backup:
            pickle.dump(listOfPhotosWithBlurryness, data_backup)


def checkIfBlurry(metric, i, index, listOfPhotosWithBlurryness):
    if(metric <1.0):
        if(not checkNextAndPreviousIfPossible(i, index, listOfPhotosWithBlurryness, metric)):
            return False
    return True

def thresholdImages(index, listOfPhotosWithBlurryness):
    if index != 0:
        i = listOfPhotosWithBlurryness[index]


        if(i.sunset_metric<-100):
            # adjustImage(index, listOfPhotosWithBlurryness)
            debugLog("sunset image")

        metric = compute_metric(i.color, i.blurryness)
        if(not checkIfBlurry(metric, i, index, listOfPhotosWithBlurryness)):
                return

        adjustImageAndPrintFilename(index, listOfPhotosWithBlurryness, metric)

#daytime and nighttime metric thresholds used to be 1.1 and 1 respectively

def checkNextAndPreviousIfPossible(i, index, listOfPhotosWithBlurryness, metric):
    if (notFirstOrLastIndex(index, listOfPhotosWithBlurryness) and not compareWithPrevious(index, metric,
                                                                                           listOfPhotosWithBlurryness)):
        debugLog("too blurry")
        if (debug):
            showImage(i, metric)
        debugLog(index)
        return False
    return True

def compareWithPrevious(i, metric, allPhotos):
    debugLog("falling back to compare with previous")

    threshold_blurryness = 0.9 if allPhotos[i].sunset_metric>-40 else 0.92

    if ((metric / compute_metric(allPhotos[i-1].color, allPhotos[i-1].blurryness) < threshold_blurryness) or (metric / compute_metric(allPhotos[i + 1].color, allPhotos[i + 1].blurryness) < threshold_blurryness)):
        if(debug):
            print("this vs previous: "+ repr(metric / compute_metric(allPhotos[i-1].color, allPhotos[i-1].blurryness)))
            print("sunset metric" + repr(allPhotos[i].sunset_metric))
            showImage(allPhotos[i], "blurry")
        return False
    return True


def notFirstOrLastIndex(index, listOfPhotosWithBlurryness):
    return not( index==1 or index == len(listOfPhotosWithBlurryness)-1)


def isNighttime(color):
    return color < 70


def adjustImageAndPrintFilename(index, listOfPhotosWithBlurryness, metric):
    path = "/media/hamish/Elements/Timelapse/reprocessed_images/" + repr(index) + ".jpg"
    print(path)

    determineRelativeBrigtness(index, listOfPhotosWithBlurryness, metric, path)


def determineRelativeBrigtness(index, listOfPhotosWithBlurryness, metric, path):
    imagestruct = listOfPhotosWithBlurryness[index]
    output1  = readAndResize(imagestruct.path)
    if (not (index == 1 or index == len(listOfPhotosWithBlurryness) - 1)):

        #this kinda works, but makes it take a lot longer for not a lot of benefit
        # if(isNighttime(imagestruct.color)):
        #     listToMean = []
        #     last_good_image = getLastGoodImage(index, listOfPhotosWithBlurryness)
        #     listToMean.append(readAndResize(listOfPhotosWithBlurryness[last_good_image].path)[:][0:1750])
        #     next_good_image = getNextGoodImage(index, listOfPhotosWithBlurryness)
        #
        #     listToMean.append(readAndResize(listOfPhotosWithBlurryness[next_good_image].path)[:][0:1750])
        #
        #     finalistMeans = []
        #     finalistMeans.append(output1[:][0:1750])
        #     finalistMeans.append(numpy.mean(listToMean, axis=0))
        #     output1[:][0:1750]= numpy.mean(finalistMeans, axis=0)

        color = imagestruct.color

        # todo, could try this with averaging, with the main image having an outsized impact compared to the other images
        # todo, brightness normalisation has to be done before comparing blurryness values, since the phone seems to be alternating between brightness levels resulting in false negatives

        # images = []
        # images.append(cv2.imread(listOfPhotosWithBlurryness[index-1].path))
        # images.append(imread)
        # images.append(cv2.imread(listOfPhotosWithBlurryness[index+1].path))
        #
        # output1 = cv2.fastNlMeansDenoisingColoredMulti(images, 1,3)

        averageBrightness = (listOfPhotosWithBlurryness[index - 1].color + listOfPhotosWithBlurryness[
            index + 1].color) / 2
        ratioBrightness = averageBrightness / color
    else:
        # output1= cv2.imread(listOfPhotosWithBlurryness[index].path)

        ratioBrightness = 1.0
        # adjust brightness
    hsv = cv2.cvtColor(output1, cv2.COLOR_BGR2HSV)  # convert it to hsv
    hsv[:, :, 2] = numpy.clip(hsv[:, :, 2] * ratioBrightness, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if (addDebugInfoToImages):
        if (isNighttime(imagestruct)):
            stampText("nighttime" + repr(metric), bgr)
        else:
            stampText("daytime" + repr(metric), bgr)
    cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 99])



def getNextGoodImage(index, listOfPhotosWithBlurryness):
    i = index+1
    metric = compute_metric(listOfPhotosWithBlurryness[i].color, listOfPhotosWithBlurryness[i].blurryness)

    while(not checkIfBlurry(metric, listOfPhotosWithBlurryness[i],index, listOfPhotosWithBlurryness) and
          not (i == 1 or i == len(listOfPhotosWithBlurryness) - 1)):
        metric = compute_metric(listOfPhotosWithBlurryness[i].color, listOfPhotosWithBlurryness[i].blurryness)
        i+=1

    return i



def getLastGoodImage(index, listOfPhotosWithBlurryness):
    i = index-1
    metric = compute_metric(listOfPhotosWithBlurryness[i].color, listOfPhotosWithBlurryness[i].blurryness)
    while(not checkIfBlurry(metric, listOfPhotosWithBlurryness[i],index, listOfPhotosWithBlurryness) and
          not (i == 1 or i == len(listOfPhotosWithBlurryness) - 1)):
        metric = compute_metric(listOfPhotosWithBlurryness[i].color, listOfPhotosWithBlurryness[i].blurryness)
        i-=1
    return i

def readAndResize(path):
    imread = cv2.imread(path)
    return cv2.resize(imread, (4000, 3000), interpolation=cv2.INTER_CUBIC)


def showImage(i, metric):
    path = i.path
    cv2.namedWindow(path, cv2.WINDOW_AUTOSIZE)
    image = cv2.imread(path)
    render(image, metric, path)


def render(image, metric, path):
    resize = cv2.resize(image, (1200, 1000))
    stampText(metric, resize)
    cv2.imshow(path, resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stampText(text, resize):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resize, repr(text), (10, 800), font, 4, (255, 255, 255), 2, cv2.LINE_AA)


def renderNoText(image):
    cv2.namedWindow("", cv2.WINDOW_AUTOSIZE)
    resize = cv2.resize(image, (2400, 2000))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.imshow("", resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def determineIfBlurry(imagePath):
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method

    imageA = cv2.imread(imagePath)


    #Rejects images already seen
    #currently doesn't seem necessary so leaving it commented out
    # middle_line = [lambda x: (x[0],x[1],x[2]) for x in imageA[2000:2001][0]]
    # if (middle_line in middle_lines):
    #     # rejecting as a dupe
    #     print("rejected as dupe")
    #     return None
    # else:
    #     print("hi")
    #     middle_lines.append(middle_line)


    #determines blurryness
    return calculateBlurryNess(imageA, imagePath)

def compute_metric(color_, fm):
    if (isNighttime(color_)):
        return fm / numpy.math.sqrt(color_)
    else:
        return fm/ 4.0



def calculateBlurryNess(imageA, imagePath):
    imageB = cv2.resize(imageA,(4000,3000), interpolation = cv2.INTER_CUBIC)
    image = cv2.GaussianBlur(imageB, (15, 15), 0)  # This removes some of the noise
    # Writes a sample with the blur applied
    # cv2.imwrite("/home/hamish/buffer/cropped" + imagePath.split("/")[-1]+".jpg", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # remove the sky since during the day it'll trick it into thinking it's out of focus
    fm = variance_of_laplacian(gray[:][2200:2400])*2
    avg_color_per_row = numpy.average(gray[:][1800:2000], axis=0)
    color = numpy.average(avg_color_per_row, axis=0)


    sunset_metric = numpy.average(numpy.average(gray[1800:2000, 0:400], axis=0), axis=0) - numpy.average(numpy.average(gray[1800:2000, -400:], axis=0),axis=0)

    with open(imagePath, 'rb') as image_file: date = datetime.strptime(Image(image_file).datetime,"%Y:%m:%d %H:%M:%S").timestamp()

    return PictureStruct(fm, imagePath, date, color, sunset_metric)


def debugLog(message):
    if(debug):
        print(message)
    pass


if __name__ == "__main__":
    doit()

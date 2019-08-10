# import the necessary packages
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
photos_ = [x for x in paths.list_images("/media/hamish/Elements/photos/photos/") if re.search("35005\d", x)]#

#photos_ = [x for x in paths.list_images("/media/hamish/Elements/photos/photos/") ]

middle_lines = []


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
    debugLog("analysing")
    with Pool(16) as p:
        listOfPhotosWithBlurryness = sorted(filter(None,p.map(determineIfBlurry, photos_)),key=sortingFunction )
    debugLog("removing blurry photos and processing")
    # with Pool(16) as p:
    non_blurry_image_paths = [thresholdImages(index, listOfPhotosWithBlurryness) for index in range(len(listOfPhotosWithBlurryness))]



def thresholdImages(index, listOfPhotosWithBlurryness):
    if index != 0:
        i = listOfPhotosWithBlurryness[index]
        prevI = listOfPhotosWithBlurryness[index-1]


        if(i.sunset_metric<-100):
            adjustImage(index, listOfPhotosWithBlurryness)

            # To make this output only the non-blurry ones (e.g for the input to a mencoder run comment out the print statements
        if (i.color < 70):
            debugLog("nighttime")
            metric = compute_metric(i.color, i.blurryness)
            #print(metric)
            if (metric < 1):
                if(index != 1 and not compareWithPrevious(i, metric, prevI)):
                    debugLog("too blurry")
                    debugLog(index)
                    return
        else:
            debugLog("daytime")
            metric = i.blurryness / 4.0
            if (metric < 1.2):
                if(index != 1 and not compareWithPrevious(i, metric, prevI)):
                    debugLog("too blurry")
                    showImage(i,metric)
                    debugLog(index)
                    return
        adjustImage(index, listOfPhotosWithBlurryness)

def adjustImage(index, listOfPhotosWithBlurryness):

    if(not( index==1 or index == len(listOfPhotosWithBlurryness)-3)):
        images = []
        imread = cv2.imread(listOfPhotosWithBlurryness[index].path)
        images.append(cv2.imread(listOfPhotosWithBlurryness[index-1].path))
        images.append(imread)
        images.append(cv2.imread(listOfPhotosWithBlurryness[index+1].path))

        output1 = cv2.fastNlMeansDenoisingColoredMulti(images, 1,3)


        averageBrightness = (listOfPhotosWithBlurryness[index-1].color+listOfPhotosWithBlurryness[index+1].color)/2
        ratioOfThisImageBrightnessToAverage = averageBrightness/listOfPhotosWithBlurryness[index].color
    else:
        output1 = cv2.fastNlMeansDenoisingColored(cv2.imread(listOfPhotosWithBlurryness[index].path))

        ratioOfThisImageBrightnessToAverage= 1.0

    #adjust brightness
    hsv = cv2.cvtColor(output1, cv2.COLOR_BGR2HSV) #convert it to hsv
    hsv[:,:,2] = numpy.clip(hsv[:,:,2] *ratioOfThisImageBrightnessToAverage,0,255)

    path = "/server_share/Timelapse/reprocessed_images/" + repr(index) + ".jpg"
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 99])
    print(path)




def compareWithPrevious(i, metric, prevI):
    if (metric / compute_metric(prevI.color, prevI.blurryness) < 0.9):
        #showImage(i, "blurry")
        return False
    return True

def showImage(i, metric):
    path = i.path
    cv2.namedWindow(path, cv2.WINDOW_AUTOSIZE)
    image = cv2.imread(path)
    render(image, metric, path)


def render(image, metric, path):
    resize = cv2.resize(image, (1200, 1000))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resize, repr(metric), (10, 800), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(path, resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    return fm / numpy.math.sqrt(color_)


def calculateBlurryNess(imageA, imagePath):
    image = cv2.GaussianBlur(imageA, (15, 15), 0)  # This removes some of the noise
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
    print(message)
    pass


if __name__ == "__main__":
    doit()

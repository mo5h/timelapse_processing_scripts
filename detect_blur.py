# import the necessary packages
import re

from imutils import paths
from multiprocessing import Pool
from collections import namedtuple
from skimage import data, io
import argparse
import cv2
import sys

blurryOrNot = {}
threshold = 8
PictureStruct = namedtuple("PictureStruct", "blurryness path")

#added regex filter to cut down the number of images for testing
photos_ = [x for x in paths.list_images("/server_share/Photos/timelapse_photos/") if re.search("6[5-9]\d\d", x)]

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sortingFunction(pictureStruct: PictureStruct):

    return int(pictureStruct.path.split("/")[-1][5:-4])# gets the 100 from "photo100.jpg" and converts it to an int

def doit():
    # loop over the input images
    with Pool(16) as p:
        listOfPhotosWithBlurryness = sorted(filter(None,p.map(determineIfBlurry, photos_)),key=sortingFunction )
    for index in range(len(listOfPhotosWithBlurryness)):
        if index!=0:
            i = listOfPhotosWithBlurryness[index]

            #To make this output only the non-blurry ones (e.g for the input to a mencoder run comment out the print statements
            if i.blurryness < threshold:
                pass
                debugLog(repr(i.blurryness))
                debugLog(i.path + " is blurry")
            else:
                debugLog(  i.path +" not blurry")
                print(i.path)


def determineIfBlurry(imagePath):
    #print("processing"+ imagePath)
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    try:
        imageA = cv2.imread(imagePath)
        image = cv2.GaussianBlur(imageA, (7,7),0)#This removes some of the noise

        #Writes a sample with the blur applied
        #cv2.imwrite("/home/hamish/buffer/cropped" + imagePath.split("/")[-1]+".jpg", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #remove the sky since during the day it'll trick it into thinking it's out of focus
        fm = variance_of_laplacian(gray[2000:3936])
        return PictureStruct(fm, imagePath)

    except(Exception, RuntimeError, OSError):
        pass


def debugLog(message):
    pass
    #print(message);


if __name__ == "__main__":
    doit()



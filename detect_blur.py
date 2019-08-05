# import the necessary packages
import re

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
PictureStruct = namedtuple("PictureStruct", "blurryness path date_taken")

# added regex filter to cut down the number of images for testing
photos_ = [x for x in paths.list_images("/server_share/Photos/timelapse_photos/") if re.search("6[5-9]\d\d", x)]
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

    with Pool(16) as p:
        listOfPhotosWithBlurryness = sorted(filter(None,p.map(determineIfBlurry, photos_)),key=sortingFunction )

    # with Pool(16) as p:
    non_blurry_image_paths = [thresholdImages(index, listOfPhotosWithBlurryness) for index in range(len(listOfPhotosWithBlurryness))]



def thresholdImages(index, listOfPhotosWithBlurryness):
    if index != 0:
        i = listOfPhotosWithBlurryness[index]

        # To make this output only the non-blurry ones (e.g for the input to a mencoder run comment out the print statements
        if i.blurryness < threshold:
            pass
            debugLog(repr(i.blurryness))
            debugLog(i.path + " is blurry")
        else:
            print(i.path)
            debugLog(i.path + " not blurry")
        return i

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


def calculateBlurryNess(imageA, imagePath):
    image = cv2.GaussianBlur(imageA, (7, 7), 0)  # This removes some of the noise
    # Writes a sample with the blur applied
    # cv2.imwrite("/home/hamish/buffer/cropped" + imagePath.split("/")[-1]+".jpg", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # remove the sky since during the day it'll trick it into thinking it's out of focus
    fm = variance_of_laplacian(gray[2000:3936])

    with open(imagePath, 'rb') as image_file: date = datetime.strptime(Image(image_file).datetime,"%Y:%m:%d %H:%M:%S").timestamp()

    return PictureStruct(fm, imagePath, date)


def debugLog(message):
    print(message)
    pass


if __name__ == "__main__":
    doit()

import re

from imutils import paths
from multiprocessing import Pool
from collections import namedtuple
from skimage import data, io
import argparse
import cv2
import sys

"""
Example output:
processing /home/hamish/buffer/sample_photos/blurry_daytime.jpg
variance of laplacian:2.2534889963999993
processing /home/hamish/buffer/sample_photos/blurry_nighttime_image.jpg
variance of laplacian:5.616592129208896
processing /home/hamish/buffer/sample_photos/non_blurry_daytime.jpg
variance of laplacian:10.009019993063378
processing /home/hamish/buffer/sample_photos/non_blurry_nighttime_image.jpg
variance of laplacian:11.800400890986758
"""

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def determineIfBlurry(imagePath):
    #print("processing"+ imagePath)
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    #print("processing "+ imagePath)

    imageA = cv2.imread(imagePath)
    image = cv2.GaussianBlur(imageA, (7,7),0)#This removes some of the noise

    #Writes a sample with the blur applied
    #cv2.imwrite("/home/hamish/buffer/cropped" + imagePath.split("/")[-1]+".jpg", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #remove the sky since during the day it'll trick it into thinking it's out of focus
    fm = variance_of_laplacian(gray[2000:3936])
    #print("variance of laplacian:" + repr(fm))


if __name__ == "__main__":
    photos_ = [x for x in paths.list_images("/home/hamish/buffer/sample_photos/")]
    for photo in photos_:
        determineIfBlurry(photo)

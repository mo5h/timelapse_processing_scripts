from datetime import datetime
import re

import numpy
from exif import Image
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


def determineIfBlurry(imagePath, blurry="blurry"):
    # print("processing"+ imagePath)
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    with open(imagePath, 'rb') as image_file:
        date = datetime.strptime(Image(image_file).datetime, "%Y:%m:%d %H:%M:%S").hour

    imageA = cv2.imread(imagePath)
    image = cv2.GaussianBlur(imageA, (7, 7), 0)  # This removes some of the noise

    # Writes a sample with the blur applied
    cv2.imwrite("/home/hamish/buffer/cropped/cropped" + imagePath.split("/")[-1] + ".jpg", image[:][1800:2000])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # remove the sky since during the day it'll trick it into thinking it's out of focus
    fm = variance_of_laplacian(gray[:][1800:2000])
    avg_color_per_row = numpy.average(gray[:][1800:2000], axis=0)
    color_ = numpy.average(avg_color_per_row, axis=0)

    # if "non" not in imagePath:
    #     print("blurry image")
    # else:
    #     print("non blurry image")



    if (color_ < 70):
        print("nighttime")
        if (compute_metric(color_, fm) < 0.9):
            output_debug_stats(color_,fm,imagePath,"")

            return
    else:
        print("daytime")
        if (fm < 3):
            output_debug_stats(color_,fm,imagePath,"")












            return

        output_debug_stats(color_,fm,imagePath,"non")



def output_debug_stats(color_, fm, imagePath, blurry=""):
    print("file: " + imagePath)
    if(blurry in imagePath):
        return
    else:
        incorrecty = "incorrectly"
    print("matched %s as %s blurry" % (incorrecty, blurry))
    print("color" + repr(color_))
    print("variance of laplacian:" + repr(fm))
    print("metric:" + repr(compute_metric(color_, fm)))
    print("================")


def compute_metric(color_, fm):
    return fm /color_


if __name__ == "__main__":
    #photos_ = [x for x in paths.list_images("/home/hamish/buffer/incorrectly_discarded")]
    photos_ = [x for x in paths.list_images("/home/hamish/buffer/sample_photos/")]
    with Pool(16) as p:
        p.map(determineIfBlurry, photos_)


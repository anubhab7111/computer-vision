import cv2 as cv
import numpy as np

blank = np.zeros((500,500, 3), dtype='uint8')
cv.imshow('Blank', blank)
# random = np.random.random((500, 500))
# print(random)
# cv.imshow('Blank', random)

# 1. Paint the image a certain color
blank[:] = 0,255,0
cv.imshow('Blank after painting green', blank)
img = cv.imread('Photos/cat.jpg')

# Painting to other colors
blank[:] = 155,0,0
cv.imshow('Other colors', blank)
# cv.imshow('Cat', img)

cv.waitKey(0)
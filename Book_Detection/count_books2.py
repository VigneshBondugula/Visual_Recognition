import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt

img = cv.imread('images/CountBooks_BookShelf.jpg')

(height, width) = img.shape[:2]
half = cv.resize(img, (0, 0), fx = 0.1, fy = 0.1)
gray = cv.cvtColor(half, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (29,29), 0)
# edges = cv.Canny(blur, 100, 150)
# dilated = cv.dilate(edges, (7, 7), iterations=3)
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
# closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)
ret, thresh = cv.threshold(blur, 125, 255, cv.THRESH_BINARY)

cimg,cnt, hierarchy = cv.findContours(
    thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
rgb = half.copy()
cv.drawContours(rgb, cnt, -1, (0, 0, 255), 2)

cv.imshow('half', half)
cv.imshow('gray', gray)
cv.imshow('thresh', thresh)

# cv.imshow('blur', blur)
# cv.imshow('edges', edges)
# cv.imshow('dilated', dilated)
# cv.imshow('closed', closed)
cv.imshow('contour', rgb)

cv.waitKey(0)

cv.destroyAllWindows()

print("Number of vertical books : ", len(cnt))

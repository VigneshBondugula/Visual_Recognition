import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt

img = cv.imread('images/CountBooks_BookShelf.jpg')

(height, width) = img.shape[:2]
half = cv.resize(img, (0, 0), fx = 0.1, fy = 0.1)
gray = cv.cvtColor(half, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (15,15), 0)
edges = cv.Canny(blur, 100, 150, 5)
dilated = cv.dilate(edges, (7, 7), iterations=3)


cimg,cnt, hierarchy = cv.findContours(
    dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
rgb = half.copy()
cv.drawContours(rgb, cnt, -1, (0, 0, 255), 2)

cv.imwrite('half.jpg', half)
cv.imwrite('gray.jpg', gray)
cv.imwrite('blur.jpg', blur)
cv.imwrite('edges.jpg', edges)
cv.imwrite('dilated.jpg', dilated)
cv.imwrite('contour.jpg', rgb)

cv.waitKey(0)

cv.destroyAllWindows()

print("Number of vertical books : ", len(cnt))

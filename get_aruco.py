import cv2 as cv
from cv2 import aruco


img = cv.imread('marker33.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
markerCorners, markerIds, rejectedCandidates = aruco.detectMarkers(gray, dictionary, parameters=parameters)

print(rejectedCandidates)
# 33

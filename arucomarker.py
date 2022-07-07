import cv2 as cv
from cv2 import aruco
import numpy as np

# Load the predefined dictionary
dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)

# Generate the marker
markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage = aruco.drawMarker(dictionary, 33, 200, markerImage, 1)

cv.imwrite("marker33.png", markerImage)

import cv2 as cv

frame = cv.imread('marker33.png')
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# Load the dictionary that was used to generate the markers.
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Initialize the detector parameters using default values
parameters = cv.aruco.DetectorParameters_create()

# Detect the markers in the image
markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(gray, dictionary, parameters=parameters)


print(markerIds)

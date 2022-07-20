# 打开摄像头并灰度化显示
import cv2
from cv2 import aruco
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # 读取摄像头画面

    # 灰度化，检测aruco标签，所用字典为6×6——250
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    # 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    print(ids)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
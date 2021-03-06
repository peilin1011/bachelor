# 打开摄像头并灰度化显示
import numpy as np
import cv2 as cv2
from cv2 import aruco
import math
import pyrealsense2 as rs

camera_matrix = np.array([[641.616, 0., 623.077], [0., 3639.931, 402.488],
                          [0., 0., 1.]])
dist_matrix = np.array(
    ([[-0.0538169, 0.0606288, 7.8376e-05, -0.000260899, -0.018944]]))

font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    h1, w1 = frame.shape[:2]
    # 读取摄像头画面

    # 灰度化，检测aruco标签，所用字典为6×6——250
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    # 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    #    如果找不打id
    if ids is not None:
        # 获取aruco返回的rvec旋转矩阵、tvec位移矩阵
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners, 0.05, camera_matrix, dist_matrix)
        # 估计每个标记的姿态并返回值rvet和tvec ---不同
        # rvec为旋转矩阵，tvec为位移矩阵
        # from camera coeficcients
        (rvec - tvec).any()  # get rid of that nasty numpy value array error
        # print(rvec)

        # 在画面上 标注auruco标签的各轴
        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, camera_matrix, dist_matrix, rvec[i, :, :],
                           tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)

        # 显示id标记 #
        cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # 角度估计 #
        # print(rvec)
        # 考虑Z轴（蓝色）的角度
        # 本来正确的计算方式如下，但是由于蜜汁相机标定的问题，实测偏航角度能最大达到104°所以现在×90/104这个系数作为最终角度
        deg = rvec[0][0][2] / math.pi * 180
        # deg=rvec[0][0][2]/math.pi*180*90/104
        # 旋转矩阵到欧拉角
        R = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(rvec, R)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:  # 偏航，俯仰，滚动
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        # 偏航，俯仰，滚动换成角度
        rx = x * 180.0 / 3.141592653589793
        ry = y * 180.0 / 3.141592653589793
        rz = z * 180.0 / 3.141592653589793

        cv2.putText(frame, 'deg_z:' + str(ry) + str('deg'), (0, 140), font, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)
        # print("偏航，俯仰，滚动",rx,ry,rz)

        # 距离估计 #
        distance = ((tvec[0][0][2] + 0.02) * 0.0254) * 100  # 单位是米
        # distance = (tvec[0][0][2]) * 100  # 单位是米

        # 显示距离
        cv2.putText(frame, 'distance:' + str(round(distance, 4)) + str('m'),
                    (0, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 真实坐标换算####（to do）
        # print('rvec:',rvec,'tvec:',tvec)
        # # new_tvec=np.array([[-0.01361995],[-0.01003278],[0.62165339]])
        # # 将相机坐标转换为真实坐标
        # r_matrix, d = cv2.Rodrigues(rvec)
        # r_matrix = -np.linalg.inv(r_matrix)  # 相机旋转矩阵
        # c_matrix = np.dot(r_matrix, tvec)  # 相机位置矩阵

    else:
        # DRAW "NO IDS" #
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

    # 显示结果画面
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

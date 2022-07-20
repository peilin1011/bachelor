import numpy as np
import cv2
import myutils as my
import imutils as im
from imutils import perspective

img_color = cv2.imread('11807.JPG')
ratio = img_color.shape[0] / 500.0
orig = img_color.copy()
img_color = im.resize(img_color, height=500)
my.cv_show('origi', img_color)
gray = img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
'''
# 增加对比度
a = 2.5
imag = float(a) * gray
imag[imag > 255] = 255  # 大于255要截断为255
# 数据类型的转换
imag = np.round(imag)
imag = imag.astype(np.uint8)
'''
# 预处理。找出轮廓
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blur, 75, 200)
# my.cv_show('edged', edged)
contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
img_color_copy = img_color.copy()
cv2.drawContours(img_color_copy, contours, -1, (0, 0, 255), 2)
# my.cv_show('contours', img_color_copy)
# print(len(contours))

locs = []
cnts = []
# 遍历每个轮廓，根据面积找出数字界面
for (i, c) in enumerate(contours):
    # 计算矩形
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    img_color_copy2 = img_color.copy()
    if 0.8 < h / w < 1.2 and x < 0.5 * img_color.shape[
            1] and y < 0.5 * img_color.shape[
                0] and h > 100 and w > 100 and area > 1:
        print(f'w={w}, h={h}')
        print(f'x={x}, y={y}')
        cv2.drawContours(img_color_copy, c, -1, (0, 0, 255), 2)
        locs.append((x, y, w, h))
        cnts.append(c)
print(len(cnts))
print(f'aruco左上角坐标为{locs[0][0]},{locs[0][1]}')
roi = img_color[locs[0][1]:locs[0][1] + locs[0][3],
                locs[0][0]:locs[0][0] + locs[0][2]]
# my.cv_show('roi', roi)

#  求出aruco四角准确坐标
for c in cnts:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)
    print(peri)
    # c表示输入的点集，epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # 4个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        break
print(approx)
warped = perspective.four_point_transform(orig,
                                          screenCnt.reshape(4, 2) * ratio)
# my.cv_show("Scanned", im.resize(warped, height=650))
cv2.destroyAllWindows()

nummer_roi = img_color[196:256, 502:553]
# my.cv_show('Nummer_roi', nummer_roi)

print(type(approx[1]))
radio = int((approx[1][1] - approx[0][1]) / 40)

unit_upperleft_x = approx[0][0] + 43 * radio
unit_upperleft_y = approx[0][1] + 16 * radio
unit_lowerright_x = approx[0][0] + 68 * radio
unit_lowerright_y = approx[0][1] + 26 * radio

unit_roi = img_color[unit_upperleft_y:unit_lowerright_y, unit_upperleft_x,
                     unit_lowerright_x]
my.cv_show('unit_roi', unit_roi)

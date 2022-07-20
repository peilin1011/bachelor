import cv2 as cv
import myutils as my
import numpy as np
import imutils as im

img = cv.imread("interface.jpg")
img = im.resize(img, height=500)
# 创建列表locs储存轮廓的形状坐标，列表cnts储存lunkuo
locs = []
cnts = []
thresh, contours = my.preprocess(img)
img_copy = img.copy()
cv.drawContours(img_copy, contours, -1, (0, 0, 255), 2)
print(len(np.array(contours)))
my.cv_show('contours', img_copy)

# 遍历每个轮廓，根据面积找出数字界面
for (i, c) in enumerate(contours):
    # 计算矩形
    x, y, w, h = cv.boundingRect(c)
    area = cv.contourArea(c)
    print(area)
    # 符合的留下来
print(len(np.array(cnts)))

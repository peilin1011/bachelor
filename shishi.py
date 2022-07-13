import cv2 as cv
import myutils as my

img = cv.imread("balance135624.jpg")
# 创建列表locs储存轮廓的形状坐标，列表cnts储存lunkuo
locs = []
cnts = []
thresh, contours = my.preprocess(img)
for (i, c) in enumerate(contours):
    # 计算矩形
    M = cv.moments(c)
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    print(f'cx:{cx} cy:{cy}')
    if 1000 < cx < 1200 and 200 < cy < 400:
        cnts.append(c)
print(len(cnts))
img_copy = img.copy()
cv.drawContours(img_copy, contours, -1, (0, 0, 255), 3)
my.cv_show('contours', img_copy)
cv.destroyAllWindows()
# 遍历每个轮廓，根据面积找出数字界面

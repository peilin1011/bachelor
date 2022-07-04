import cv2 as cv
import myutils as my
import numpy as np

# ***************** 第一步：处理模版 ******************
number = cv.imread('1to0.png')
thresh, contours = my.preprocess(number)
number_copy = number.copy()
cv.drawContours(number_copy, contours, -1, (0, 255, 0), 2)
# print(len(np.array(contours)))
my.cv_show('contours', number_copy)

contours = my.sort_contours(contours)[0]
digits = {}

# 单个轮廓提取到字典中
for (i, c) in enumerate(contours):
    (x, y, w, h) = cv.boundingRect(c)
    roi = number[y:y + h, x:x + w]  # 在模板中复制出轮廓
    roi = cv.resize(roi, (57, 88))  # 改成相同大小的轮廓
    if i == 9:
        digits[0] = roi
    else:
        digits[i + 1] = roi  # 此时字典键对应的轮廓即为对应数字。
    # my.cv_show('roi', roi)

# ***************** 第二步：获取界面ROI *************************
# 读取原图
img = cv.imread("balance.jpg")
# 创建列表locs储存轮廓的形状坐标，列表cnts储存lunkuo
locs = []
cnts = []
thresh, contours = my.preprocess(img)
# 遍历每个轮廓，根据面积找出数字界面
for (i, c) in enumerate(contours):
    # 计算矩形
    x, y, w, h = cv.boundingRect(c)
    area = cv.contourArea(c)
    # print(area)
    # 符合的留下来
    if 47000 > area > 46000:
        locs.append((x, y, w, h))
        cnts.append(c)
# print(len(np.array(cnts)))
# 求ROI
roi = img[locs[0][1]:locs[0][1] + locs[0][3],
          locs[0][0]:locs[0][0] + locs[0][2]]
my.cv_show('roi', roi)

roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
# my.cv_show('roi_gray', roi_gray)
roi_thresh = cv.threshold(roi_gray, 0, 255,
                          cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
# my.cv_show('roi_thresh', roi_thresh)

roi_median = cv.medianBlur(roi_thresh, 5)  # 中值滤波
i = 0
while i < 13:
    roi_median = cv.medianBlur(roi_median, 5)
    i += 1
# my.cv_show('roi_median', roi_median)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 矩形结构
roi_dilation = cv.dilate(roi_median, kernel)  # 膨胀
roi_dilation = cv.dilate(roi_dilation, kernel)  # 膨胀
# my.cv_show('dilation', roi_dilation)

# 提取小轮廓，即每个数字
roi_contours = cv.findContours(roi_dilation.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[0]
roi_contours = my.sort_contours(roi_contours, method="left-to-right")[0]

roi_cnts = []
for (i, c) in enumerate(roi_contours):
    # 计算矩形
    x, y, w, h = cv.boundingRect(c)
    perimeter = cv.arcLength(c, True)
    if perimeter > 250:
        roi_cnts.append(c)

roi_copy = roi.copy()
cv.drawContours(roi_copy, roi_cnts, -1, (0, 0, 255), 2)
my.cv_show('roi_contours', roi_copy)

# *************** 第三步：模版匹配提取数据 *******************+*****+

# print(len(np.array(roi_locs)))
groupOutput = []
for (i, c) in enumerate(roi_cnts):  # c表示每个小轮廓的终点坐标
    (x, y, w, h) = cv.boundingRect(c)
    roi_number = roi[y:y + h, x:x + w]
    roi_number = cv.resize(roi_number, (57, 88))
    my.cv_show('roi', roi_number)

    scores = []
    # 在模板章中计算每一个得分
    for (digit, digiROI) in digits.items():
        # 模板匹配
        result = cv.matchTemplate(roi_number, digiROI, cv.TM_CCOEFF)
        score = cv.minMaxLoc(result)[1]
        scores.append(score)
    # 得到合适的数字
    groupOutput.append(str(np.argmax(scores)))
print(groupOutput)
cv.destroyAllWindows()

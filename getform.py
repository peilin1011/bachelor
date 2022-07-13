import cv2 as cv
import myutils as my
import imutils as im
'''第一部分：探测边缘'''

img = cv.imread('origin.jpg')
ratio = img.shape[0] / 500.0
img_copy = img.copy()
imag = im.resize(img_copy, height=500)

# ************ 预处理 *********************
gray = cv.cvtColor(imag, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(blur, 75, 200)
edged = cv.GaussianBlur(edged, (5, 5), 0)
my.cv_show('edged', edged)

# ************* 轮廓检测 *******************
# 轮廓检测
contours = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, 2)[0]

'''
cv.drawContours(img_copy, contours, -1, (0, 255, 0), 2)
my.cv_show('contours', img_copy)
cnts = sorted(contours, key=cv.contourArea, reverse=True)[:5]
print(len(contours))
'''
cnts = sorted(contours, key=cv.contourArea, reverse=True)[:5]
# 遍历轮廓
for c in cnts:
    # 计算轮廓近似
    peri = cv.arcLength(c, True)
    # c表示输入的点集，epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
    approx = cv.approxPolyDP(c, 0.02*peri, True)

    # 4个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        break

cv.drawContours(img_copy, screenCnt, -1, (0, 255, 0), 2)
my.cv_show('contours', img_copy)
# 第二部分：提取菜单矩阵轮廓四点进行透视变换
# 第三部分：应用一个透视的转换去获取一个文档的自顶向下的一个正图

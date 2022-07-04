import cv2 as cv
import myutils as my
'''第一部分：探测边缘'''

img = cv.imread('balance.jpg')
ratio = img.shape[0] / 500.0

# ************ 预处理 *********************
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(blur, 75, 200)
edged = cv.GaussianBlur(edged, (5, 5), 0)
my.cv_show('edged', edged)

# ************* 轮廓检测 *******************
# 轮廓检测
contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key=cv.contourArea, reverse=True)[:5]
print(len(contours))

# 第二部分：提取菜单矩阵轮廓四点进行透视变换
# 第三部分：应用一个透视的转换去获取一个文档的自顶向下的一个正图

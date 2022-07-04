import cv2 as cv


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv.boundingRect(c)
                     for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(
        zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes


#  自定义改变图片大小的函数
def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv.resize(image, dim, interpolation=inter)
    return resized


#  自定义的图片顺时针旋转angle度，缩放比例我哦inter函数
def rotated(image, angle, inter):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, inter)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated


#  自定义的展示函数
def cv_show(name, image):
    cv.imshow(name, image)
    cv.waitKey(0)


# 自定义预处理图像函数
def preprocess(img):
    # 灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # my.cv_show('gray', gray)

    # 阈值化
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # my.cv_show('thresh', thresh)

    # 滤波
    median = cv.medianBlur(thresh, 5)  # 中值滤波
    i = 0
    while i < 13:
        median = cv.medianBlur(median, 5)
        i += 1

    # my.cv_show('msdian', median)

    # 初始化卷积核
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 40))

    # 通过闭操作（先膨胀，再腐蚀）将数字连在一起.  将本是4个数字的4个框膨胀成1个框,就腐蚀不掉了
    gradX = cv.morphologyEx(median, cv.MORPH_CLOSE, rect_kernel)
    gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rect_kernel)
    gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rect_kernel)
    gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rect_kernel)
    # my.cv_show('close1', gradX)

    # 二值化
    thresh = cv.threshold(gradX, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    # 计算轮廓
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[0]
    img_copy = img.copy()
    cv.drawContours(img_copy, contours, -1, (0, 0, 255), 2)
    # print(len(np.array(contours)))
    # my.cv_show('contours', img_copy)
    return thresh, contours

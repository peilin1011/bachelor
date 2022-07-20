import cv2 as cv
import numpy as np

#  自定义的展示函数
def cv_show(name, image):
    cv.imshow(name, image)
    cv.waitKey(0)

def nummer_segments(img, trans):

    cv_show('img', img)

    if trans == 1:
        O = cv.equalizeHist(img)
    else: 
        O = img

    # 腐蚀
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(O, kernel)
    cv_show('erosion', erosion)
    # 滤波
    gaussian = cv.GaussianBlur(erosion, (5, 5), 1)  # 高斯滤波
    # 阈值化
    thresh= cv.threshold(gaussian, 160, 255, cv.THRESH_BINARY)[1]
    cv_show('thresh', thresh)
    # canny边缘检测
    

    edges = cv.Canny(thresh, 30, 70)
    cv_show('edges', edges)

    contours = cv.findContours(edges.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[0]
    segment_number = len(contours)
    print(f'Segment 数量为:{segment_number}')
    img_copy = img.copy()
    cv.drawContours(img_copy, contours, -1, (0, 0, 255), 2)
    '''
    for (i,c) in enumerate(contours):
        x, y, w, h = cv.boundingRect(c)  # 外接矩形
        cx = x + 0.5*w
        cy = y + 0.5*h
        print(f'中心点为{cx},{cy}')
    '''
    cv_show('contours', img_copy)
    cv.destroyAllWindows()
    return thresh, segment_number


def contrast_radio(img):
    i = 0
    trans = 0
    for x in range(181):
        for y in range(100):
            pixel = img[x, y]
            if 80 < pixel < 120:
                i = i + 1
    print(f'i= {i}')
    if i > 1400:
        trans = 1
    return trans


img = cv.imread('number3.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (100,181))
trans = contrast_radio(img)
print(trans) 
thresh, segment_number = nummer_segments(img, trans)

correct_number = 0
location_segment = []
if segment_number == 6:
    upper_left = thresh[49,15]
    location_segment.append(upper_left)
    print(upper_left)
    upper_right = thresh[49,85]
    location_segment.append(upper_right)
    print(upper_right)
    lower_left = thresh[129,15]
    location_segment.append(lower_left)
    print(lower_left)
    lower_right = thresh[129,85]
    location_segment.append(lower_right)
    print(lower_right)
    print(location_segment)
    if location_segment == [255, 0, 0, 255]:
        correct_number = 5
    elif location_segment == [0, 255, 255, 0]:
        correct_number = 2
    elif location_segment == [0, 255, 0, 255]:
        correct_number = 3
    elif location_segment == [255, 255, 255, 255]:
        correct_number = 0
elif segment_number == 7:
     upper_left = thresh[49,15]
     location_segment.append(upper_left)
     print(upper_left)
     upper_right = thresh[49,85]
     location_segment.append(upper_right)
     print(upper_right)
     lower_left = thresh[129,15]
     location_segment.append(lower_left)
     print(lower_left)
     lower_right = thresh[129,85]
     location_segment.append(lower_right)
     print(lower_right)
     print(location_segment)
     if location_segment == [255, 0, 255, 255]:
        correct_number = 6
     elif location_segment == [255, 255, 0, 255]:
        correct_number = 9
elif segment_number == 2:
     correct_number = 1
elif segment_number == 5:
     correct_number = 4
elif segment_number == 3:
     correct_number = 7
elif segment_number == 8:
     correct_number = 8
print(f'The numbers identified as {correct_number}')
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


image_processing_option={
    'id': 'myung9',
    'dataset_name': 'idonkown',
    'order': 0,
    'image_path': '../images/cat.49.jpg',
    'image_folder_path': '../iamges',
    ################################
    'processing_mode': 1,

}


# before train -> for train
# after train -> for prediction

class ImageProcessing(object): # 오로지 이미지에 단순 수학적 연산만하는 클래스

    MIN_THRESH = 120
    MAX_THRESH = 255

    def __init__(self, processing_option):
        print('preprocessing')
        self.order = processing_option['order']

    def gray(self, img_param):
        return cv2.cvtColor(img_param, cv2.COLOR_BGR2GRAY)

    def binary_threshold(self, img_param):
        ret, dst = cv2.threshold(img_param, self.MIN_THRESH, self.MAX_THRESH,
                                 cv2.THRESH_BINARY)
        return dst

    def binary_inv_threshold(self, img_param):
        ret, dst = cv2.threshold(img_param, self.MIN_THRESH, self.MAX_THRESH,
                                 cv2.THRESH_BINARY_INV)
        return dst

    def adaptive_threshold(self, img_param):
        adapt_th = cv2.adaptiveThreshold(img_param, 250,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         3, 20)
        # cv2.ADAPTIVE_THRESH_MEAN_C == 0
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C == 1
        return adapt_th

    def blur(self, img_param): # gaussian blur
        #가우시안 커널 사이즈 5 / 0은 알아보기
        return cv2.GaussianBlur(img_param, (5, 5), 0)

    def morph_gradient(self, img_param):
        kernel = np.ones((3, 3), np.uint8)
        morph_g = cv2.morphologyEx(img_param, cv2.MORPH_GRADIENT, kernel)
        return morph_g

    def morph_close(self,img_param):
        morph_kernel = np.ones((10, 10), np.uint8)
        morph_c = cv2.morphologyEx(img_param, cv2.MORPH_CLOSE, morph_kernel)
        return morph_c

    def canny(self, img_param):
        edges = cv2.Canny(img_param, 50, 150, apertureSize=3) # 3 = sobel_kernel_size
        return edges

    def erosion(self, img_param):
        iter_time = 2
        erosion_kernel = np.ones((2, 2), np.uint8)
        return cv2.erode(img_param, erosion_kernel, iterations=iter_time)

    def dilation(self, img_param):
        iter_time = 2
        dilation_kernel = np.ones((2, 2), np.uint8)
        return cv2.dilate(img_param, dilation_kernel, iter_time)




#class SetImage(ImageProcessing):
class SetImage():
    def __init__(self):
        pass

    def seperate(self, img_param):
        result = []
        contours1, hierarchys1 = cv2.findContours(img_param, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        convexhull = []
        for contour in contours1:
            hull = cv2.convexHull(contour)
            convexhull = cv2.drawContours(img_param, [hull], 0, (125, 125, 125), thickness=-1)
        contours2, hierarchys2 = cv2.findContours(convexhull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        img_w = img_param.shape[1]
        for contour in contours2:
            x ,y, w, h = cv2.boundingRect(contour)
            y2 = round(y / 10) * 10
            index = y2 * img_w + x
            rects.append((index, x, y, w, h))
        rects = sorted(rects, key=lambda x:x[0])

        for i, rect in enumerate(rects):
            index, x, y, w, h = rect
            seperted_img = img_param[y:y + h, x:x + w]
            seperted_img = 255 - seperted_img

            ww = round((w if w > h else h)* 1.1)
            spc = np.zeros((ww,ww))
            wy = (ww - h) // 2
            wx = (ww - w) // 2

            spc[wy:wy + h, wx:wx + w] = seperted_img
            ##리사이즈 구역

            result.append(seperted_img)
        return result









if __name__ == '__main__':
    print('main start')
    img = cv2.imread('../images/cat.49.jpg')
    #print(img)
    SetImage.seperate(img_param=img)


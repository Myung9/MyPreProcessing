import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


processing_option = {
    'order': 'a'
}


def print_img(window_name,img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class ImageProcessingModule():
    def __init__(self, processing_option):
        self.order = processing_option['order']
    #이미지 연산 모듈
    def gray(self, img_param):
        if len(img_param.shape) == 2:
            print('Thin image is already gray scale or has no channel')
            return img_param
        else:
            try:
                return cv2.cvtColor(img_param, cv2.COLOR_BGR2GRAY)
            except:
                print('convert gray scale failed')

    '''
    def binary_threshold(self, img_param):
        ret, dst = cv2.threshold(img_param, 127, 255, cv2.THRESH_BINARY)
        return dst

    def binary_threshold_inv(self, img_param):
        ret, dst = cv2.threshold(img_param, 127, 255, cv2.THRESH_BINARY_INV)
        return dst

    def binary_threshold_otsu(self, img_param):
        ret, dst = cv2.threshold(img_param, 127, 255, cv2.THRESH_OTSU)
        return dst

    def binary_threshold_TOZERO(self, img_param):
        ret, dst = cv2.threshold(img_param, 127, 255, cv2.THRESH_TOZERO)
        return dst
    '''

    def binary_threshold(self, img_param, thresh_mode):
        if thresh_mode == None:
            ret, dst = cv2.threshold(img_param, 127, 255, cv2.THRESH_BINARY)
        elif thresh_mode == 'BINARY_INV':
            ret, dst = cv2.threshold(img_param, 127, 255, cv2.THRESH_BINARY_INV)
        elif thresh_mode == 'OTSU':
            ret, dst = cv2.threshold(img_param, 127, 255, cv2.THRESH_OTSU)
        elif thresh_mode == 'TOZERO':
            ret, dst = cv2.threshold(img_param, 127, 255, cv2.THRESH_TOZERO)
        elif thresh_mode == 'TRUNC':
            ret, dst = cv2.threshold(img_param, 127, 255, cv2.THRESH_TRUNC)
        else:
            ret, dst = cv2.threshold(img_param, 127, 255, cv2.THRESH_BINARY)
        return dst

    def adaptive_threshold(self, img_param, mask_mode):
        if mask_mode == 'MEAN_C':
            return cv2.adaptiveThreshold(img_param, 250,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         3, 20)
        elif mask_mode == 'GAUS':
            return cv2.adaptiveThreshold(img_param, 250,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         3, 20)
        elif mask_mode == None:
            return cv2.adaptiveThreshold(img_param, 250,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         3, 20)

    def blur(self, img_param):
        return cv2.GaussianBlur(img_param, (5, 5), 0)

    def morph_gradient(self, img_param):
        kernel = np.ones((10, 10), np.uint8)
        return cv2.morphologyEx(img_param, cv2.MORPH_CLOSE, kernel)

    def morph_close(self, img_param):
        kernel = np.ones((10, 10), np.uint8)
        return cv2.morphologyEx(img_param, cv2.MORPH_GRADIENT, kernel)

    def canny(self, img_param):
        #apertureSize : sobel kernel 사이즈
        return cv2.Canny(img_param, 100, 150, apertureSize=3)

    def erosion(self, img_param):
        iter_time = 1
        erosion_kernel = np.ones((2, 2), np.uint8)
        return cv2.erode(img_param, erosion_kernel, iterations=iter_time)

    def dilation(self, img_param):
        iter_time = 1
        dilation_kernel = np.ones((2, 2), np.uint8)
        return cv2.dilate(img_param, dilation_kernel, iter_time)




class SeperateImage():
    def __init__(self, processing_option):
        super().__init__(processing_option)
        self.img_list = []

        #이미지 자르고하는거

    def seperate(self, img_param):

        ip = ImageProcessing(processing_option)
        img = cv2.cvtColor(img_param, cv2.COLOR_BGR2GRAY)
        print_img('원본이미지', img_param)
        bt = ip.binary_threshold(img_param)
        blur = ip.blur(bt)
        mg = ip.morph_gradient(blur)
        canny = ip.canny(mg)
        print_img('canny edge', canny)

        contours_1, hierarchy_1 = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        convexHull = np.empty(())
        for i, contour_1 in enumerate(contours_1):
            x, y, w, h = cv2.boundingRect(contour_1)
            print(i+1, '번째 좌표(1) ', cv2.boundingRect(contour_1))
            hull = cv2. convexHull(contour_1)
            convexHull = cv2.drawContours(canny, [hull], 0, (125, 125, 125))
            print(type(convexHull))
            print(convexHull.shape)
        print_img('convexhull', convexHull)
        rects = []
        #img_h, img_w = img_param.shape[:2]
        contours_2, hierarchy_2 = cv2.findContours(convexHull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour_2 in enumerate(contours_2):
            x, y, w, h = cv2.boundingRect(contour_2)
            print(i+1, '번째 컨투어스 좌표(2) ', cv2.boundingRect(contour_2))
            cv2.rectangle(img_param, (x, y), (x + w, y + h), (255, 0, 0), 1)
            blocking = cv2.drawContours(img, contours_2, -1, (0, 0, 255), 2)
            #cv2.line(blocking, (x,y),(x+1,y+1), (0,100,255),3)
            #print_img(blocking)
            rects.append([i, x, y, w, h])
            # 여기서 인덱스를 어떡게할지나 생각해보기 / 그냥놔둘지 바꿀지
            # 인덱스 순서는 왼쪽아래부터 오른쪽으로 확인하면서 위로
        rects = sorted(rects, key=lambda x:x[0])
        shapes = []
        img_list = []
        seperated_img = np.empty(())
        for i, rect in enumerate(rects):
            img_list = self.img_list
            idx, x, y, w, h = rect
            seperated_img = bt[y:y+h, x:x+w]
            seperated_img = 255 - seperated_img
            img_list.append(seperated_img)
            shapes.append(seperated_img.shape)
        print_img('seperated_img', seperated_img)
        return img_param, img_list, rects # 잘린이미지리스트 // 원본에서의 좌표와 w,h 리스트






class ImageProcessing(ImageProcessingModule):
    def __init__(self, processing_option):
        super().__init__(processing_option)
        self.ipm = ImageProcessingModule(processing_option=processing_option)
        #엣지옵션 -> canny / adat_gausi / adat_meanc



SeperateImage(processing_option=processing_option).seperate(cv2.imread('../images/cat.57.jpg'))
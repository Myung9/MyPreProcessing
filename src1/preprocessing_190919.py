import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

###############################################
###############전역 변수 설정###################
##############################################

############모폴리지##########################
MORPH_KERNEL_SIZE = 10  # 모폴로지 커널 사이즈
morph_kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)  # 모폴리지처리용 커널 선언

###########스레숄드##########################
MIN_THRESH = 100  # 스레숄드 연산에 사용될 최소값
MAX_THRESH = 255  # 스레숄드 연산에 사용될 최대값

########적응형스레숄드#########################
ADPT_THRESH = 250  # adaptiveThreshold에 의해 계산된 문턱값과
# thresholdType에 의해 픽셀에 적용될 최대값
ADPT_BLOCKSIZE = 3
WEIGHTED_C = 20


#############CANNY##########################
MIN_CANNY = 50  # MIN_CANNY 이하에 포함된 가장자리에서 제외
MAX_CANNY = 150  # MAX_CANNY 이상에 포함된 가장자리는 가장자리로 간주
APERTURE_SIZE = 3 #canny aperture size
SOBEL_KERNEL_SIZE = 3  # Canny에서의 커널 크기 / Sobel마스크의 Aperture Size를 의미
# == apertureSize

###########BLUR###########################
GAUSSIAN_KERNEL_SIZE = 5  # 가우시안블러의 커널 크기 / 보통 5를 사용


##########erosion##########################
EROSION_ITER1 = 1  # erosion 반복횟수
EROSION_ITER2 = 2
EROSION_ITER3 = 3
EROSION_ITER4 = 4
EROSION_ITER5 = 5
EROSION_KERNEL_SIZE = 2 #erosion 커널사이즈
erosion_kernel = np.ones((EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE), np.uint8)

##########dilation#########################
DILATION_ITER1 = 1  # dilation 반복횟수
DILATION_ITER2 = 2
DILATION_ITER3 = 3
DILATION_ITER4 = 4
DILATION_ITER5 = 5
DILATION_KERNEL_SIZE = 2 #dilation 커널사이즈
dilation_kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
###########################################
MARGIN_FOR_SLICEDIMG = 1.15



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
        kernel = np.ones((10, 10), np.uint8)
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



imgList = []
#class SetImage(ImageProcessing):
class SetImage(ImageProcessing):
    def __init__(self):
        pass
    def seperate(self, img_param):
        ip = ImageProcessing(image_processing_option)
        #img = cv2.cvtColor(img_param, cv2.COLOR_BGR2GRAY)
        print_img(img_param)
        bt = ip.binary_threshold(img_param)
        blur = ip.blur(bt)
        mg = ip.morph_gradient(blur)
        canny = ip.canny(mg)
        print_img(canny)

        contours_1, hierarchy_1 = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        convexHull = np.empty(())
        for i, contour_1 in enumerate(contours_1):
            x, y, w, h = cv2.boundingRect(contour_1)
            print(i+1, '번째 좌표(1) ', cv2.boundingRect(contour_1))
            hull = cv2. convexHull(contour_1)
            convexHull = cv2.drawContours(canny, [hull], 0, (125, 125, 125))
            print(type(convexHull))
            print(convexHull.shape)
        print_img(convexHull)
        rects = []
        img_h, img_w = img_param.shape[:2]
        contours_2, hierarchy_2 = cv2.findContours(convexHull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour_2 in enumerate(contours_2):
            x, y, w ,h = cv2.boundingRect(contour_2)
            print(i+1, '번째 컨투어스 좌표(2) ', cv2.boundingRect(contour_2))
            cv2.rectangle(img_param, (x, y), (x + w, y + h), (255, 0, 0), 1)
            blocking = cv2.drawContours(img, contours_2, -1, (0, 0, 255), 2)
            #cv2.line(blocking, (x,y),(x+1,y+1), (0,100,255),3)
            print_img(blocking)
            rects.append([i, x, y, w, h])
            # 여기서 인덱스를 어떡게할지나 생각해보기 / 그냥놔둘지 바꿀지
            # 인덱스 순서는 왼쪽아래부터 오른쪽으로 확인하면서 위로
        rects = sorted(rects, key=lambda x:x[0])
        print(rects)
        for i, rect in enumerate(rects):
            idx, x, y, w, h = rect
            print('wow', rect)
            seperated_img = bt[y:y+h, x:x+w]
            print_img(seperated_img)


def print_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)




if __name__ == '__main__':
    print('main start')
    #img = cv2.imread('../images/cat.49.jpg')
    img = cv2.imread('../images/wow.JPG')
    #print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('aaa', img)
    #cv2.waitKey(0)
    #print(img)
    a = SetImage()
    a.seperate(img)


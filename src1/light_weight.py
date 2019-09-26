import cv2
import numpy as np
import math
from scipy import ndimage

import sys
import matplotlib.pyplot as plt
from PIL import Image

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



######################################################
###################전처리 함수 영역#####################
######################################################
def Gray(img_param):
    gray = cv2.cvtColor(img_param, cv2.COLOR_BGR2GRAY)
    print('Gray 진행')
    return gray

def binary_Threshold(img_param):
    ret, dst = cv2.threshold(img_param, MIN_THRESH, MAX_THRESH, cv2.THRESH_BINARY)
    print('이진화 진행')
    return dst

def Blur(img_param):
    blur = cv2.GaussianBlur(img_param, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)
    print('Blur 진행')
    return blur

def morph_GRADIENT(img_param):
    morph_G = cv2.morphologyEx(img_param, cv2.MORPH_GRADIENT, morph_kernel)
    print('Morphology Gradient 진행')
    return morph_G

def adaptive_Threshold(img_param):
    adapt_th = cv2.adaptiveThreshold(img_param, ADPT_THRESH, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ADPT_BLOCKSIZE, WEIGHTED_C)
    # cv2.ADAPTIVE_THRESH_MEAN_C == 0
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C == 1
    print('adaptive threshold')
    return adapt_th

def morph_CLOSE(img_param):
    morph_C = cv2.morphologyEx(img_param, cv2.MORPH_CLOSE, morph_kernel)
    print('Morphology CLOSE 진행')
    return morph_C

def Canny(img_param):
    edges = cv2.Canny(img_param, MIN_CANNY, MAX_CANNY, apertureSize=APERTURE_SIZE)
    print('CANNY EDGES 진행')
    return edges

def Erosion(img_param, iter):
    erode = cv2.erode(img_param, erosion_kernel, iterations=iter)
    #print('erosion', iter, '번 진행')
    return erode

def Dilatation(img_param, iter):
    dil = cv2.dilate(img_param, dilation_kernel, iter)
    #print('dilation', iter, '번 진행')
    return dil

###################################################################

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def Display(argv):
    count = 0
    nrows = 6
    ncols = 6
    plt.figure(figsize=(8, 8))
    for n in range(len(argv)):
        count += 1
        plt.subplot(nrows, ncols, count)
        #plt.title(img_array_name[n])
        #plt.imshow(argv[n], cmap='Greys_r')
        plt.imshow(argv[n])
    plt.tight_layout()
    plt.show()
img_array_name = []



def main(argv):
    img = cv2.imread(argv)
    gray = Gray(img)
    bt = binary_Threshold(gray)
    blur = Blur(bt)
    mg = morph_GRADIENT(blur)
    at = adaptive_Threshold(mg)
    mc = morph_CLOSE(at)
    canny = Canny(mg)
    cv2.imshow('daa', mg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    plt.imshow(img)
    plt.show()
    c = 0
    HorW = []
    print('\n')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print(c + 1, '번째 컨투어스 좌표(1)', cv2.boundingRect(cnt))
        c = c + 1
        if w >= h:
            HorW.append(w)
        else:
            HorW.append(h)
        hull = cv2.convexHull(cnt)
        convexHull = cv2.drawContours(canny, [hull], 0, (125, 125, 125), thickness=-1)
    print('넓이와 높이 중 최대값 : ', max(HorW))
    print('contours 개수(1) : ', c)

    #plt.imshow(canny)
    #plt.show()
    plt.imshow(convexHull, cmap='Greys_r')
    plt.show()
    print('fill convexhull')

    n_contours, n_hierarchy = cv2.findContours(convexHull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_c = 0
    n_HorW = []
    rects = []
    im_w = canny.shape[1]


    for cnt in n_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print(n_c + 1, '번째 컨투어스의 좌표', cv2.boundingRect(cnt))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        blocking = cv2.drawContours(img, n_contours, -1, (0, 0, 255), 1)
        n_c = n_c + 1
        if w >= h:
            n_HorW.append(w)
        else:
            n_HorW.append(h)

        y2 = round(y/10)*10
        index = y2 * im_w + x
        rects.append((index, x, y, w, h))


    rects = sorted(rects, key=lambda x:x[0])

    spcc = np.zeros((28, 28))
    X = []
    imgList = []
    count = 0
    for i, r in enumerate(rects):
        index, x, y, w, h = r
        num = bt[y:y+h, x:x+w]
        num = 255 - num
        #print('flag1aaaa', i, r)
        count = count + 1

        #imgList.append(num)


        ww = round((w if w > h else h) * MARGIN_FOR_SLICEDIMG)
        #margin_for_slicedImg
        spc = np.zeros((ww, ww))
        wy = (ww-h)//2
        wx = (ww-w)//2

        spc[wy:wy+h, wx:wx+w] = num
        print(spc.shape)
        if h or w < 28:
            num = cv2.resize(spc, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            num = cv2.resize(spc, (28, 28), interpolation=cv2.INTER_LINEAR)



        imgList.append(num)

        #print('list(1)', np.nditer(imgList))
        #print('num(1)', np.nditer(num))
        #print('rects(1)', np.nditer(rects))
    print(count)


    for i in range(count):

        #np.ravel(num, order='C')
        #dd = np.array(num)
        #num = np.arange(784).reshape(28, 28)

        num = Dilatation(imgList[i], DILATION_ITER1)

        #print('flag1', i, r)
        #print('rect=', rects)

        imgList.append(num)
        #print('list(2)', np.nditer(imgList))
        #print('num(2)', np.nditer(  num))
        #print('rects(2)', np.nditer(rects))

        #num = num.reshape(28 * 28)
        #num = num.astype("float32") / 255.0
        X.append(num)


    param = np.array(X)
    Display(imgList)
    #for i in len(np.ndenumerate(imgList)):
    #    param = Dilatation(imgList, 1)

    return param
'''
    def main2(num):
        for i in num:
            X = []
            X.append(num)
            param = np.array(X)
        return param
'''

main('C:\DEV\PycharmProjects\MyPreProcessing\images\wow.JPG')
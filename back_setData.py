import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import log
log = log.Logger('setData')

#######################
X = []
imgList = []
imgList2 = []
cnt_imgList = 0
img_array_name = []
#######################

def PreProcessing(pre_option):

    argv = pre_option['img_PATH']
    pre_img_mode = pre_option['pre_img_mode']
    after_img_mode = pre_option['after_img_mode']

    ##############################################
    ###############################################
    ###############전역 변수 설정###################
    ##############################################

    ############MINST_SIZE#######################
    MNIST_IMAGE_FORMAT_SIZE = pre_option['MNIST_IMAGE_FORMAT_SIZE']

    ############모폴리지##########################
    MORPH_KERNEL_SIZE = pre_option['MORPH_KERNEL_SIZE']  # 모폴로지 커널 사이즈
    morph_kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)  # 모폴리지처리용 커널 선언

    ###########스레숄드##########################
    MIN_THRESH = pre_option['MIN_THRESH']  # 스레숄드 연산에 사용될 최소값
    MAX_THRESH = pre_option['MAX_THRESH']  # 스레숄드 연산에 사용될 최대값

    ########적응형스레숄드#########################
    ADPT_THRESH = pre_option['ADPT_THRESH']  # adaptiveThreshold에 의해 계산된 문턱값과
    # thresholdType에 의해 픽셀에 적용될 최대값
    ADPT_BLOCKSIZE = pre_option['ADPT_BLOCKSIZE']
    WEIGHTED_C = pre_option['WEIGHTED_C']

    #############CANNY##########################
    MIN_CANNY = pre_option['MIN_CANNY']  # MIN_CANNY 이하에 포함된 가장자리에서 제외
    MAX_CANNY = pre_option['MAX_CANNY']  # MAX_CANNY 이상에 포함된 가장자리는 가장자리로 간주
    SOBEL_KERNEL_SIZE = pre_option['SOBEL_KERNEL_SIZE']  # Canny에서의 커널 크기 / Sobel마스크의 Aperture Size를 의미
    # == apertureSize

    ###########BLUR###########################
    GAUSSIAN_KERNEL_SIZE = pre_option['GAUSSIAN_KERNEL_SIZE']  # 가우시안블러의 커널 크기 / 보통 5를 사용

    ##########erosion##########################
    EROSION_ITER1 = pre_option['EROSION_ITER1']  # erosion 반복횟수
    EROSION_ITER2 = pre_option['EROSION_ITER2']
    EROSION_ITER3 = pre_option['EROSION_ITER3']
    EROSION_ITER4 = pre_option['EROSION_ITER4']
    EROSION_ITER5 = pre_option['EROSION_ITER5']
    EROSION_KERNEL_SIZE = pre_option['EROSION_KERNEL_SIZE']  # erosion 커널사이즈
    erosion_kernel = np.ones((EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE), np.uint8)

    ##########dilation#########################
    DILATION_ITER1 = pre_option['DILATION_ITER1']  # dilation 반복횟수
    DILATION_ITER2 = pre_option['DILATION_ITER2']
    DILATION_ITER3 = pre_option['DILATION_ITER3']
    DILATION_ITER4 = pre_option['DILATION_ITER4']
    DILATION_ITER5 = pre_option['DILATION_ITER5']
    DILATION_KERNEL_SIZE = pre_option['DILATION_KERNEL_SIZE']  # dilation 커널사이즈
    dilation_kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
    ###########################################
    MARGIN_FOR_SLICEDIMG = pre_option['MARGIN_FOR_SLICEDIMG']

    ######################################################
    ###################전처리 함수 영역#####################
    ######################################################

    def Gray(img_param):
        gray = cv2.cvtColor(img_param, cv2.COLOR_BGR2GRAY)
        return gray

    def binary_Threshold(img_param):
        ret, dst = cv2.threshold(img_param, MIN_THRESH, MAX_THRESH, cv2.THRESH_BINARY)
        return dst

    def Blur(img_param):
        blur = cv2.GaussianBlur(img_param, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)
        return blur

    def morph_GRADIENT(img_param):
        morph_G = cv2.morphologyEx(img_param, cv2.MORPH_GRADIENT, morph_kernel)
        return morph_G

    def adaptive_Threshold(img_param):
        adapt_th = cv2.adaptiveThreshold(img_param, ADPT_THRESH, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                         ADPT_BLOCKSIZE, WEIGHTED_C)
        # cv2.ADAPTIVE_THRESH_MEAN_C == 0
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C == 1
        return adapt_th

    def morph_CLOSE(img_param):
        morph_C = cv2.morphologyEx(img_param, cv2.MORPH_CLOSE, morph_kernel)
        return morph_C

    def Canny(img_param):
        edges = cv2.Canny(img_param, MIN_CANNY, MAX_CANNY, apertureSize=SOBEL_KERNEL_SIZE)
        return edges

    def Erosion(img_param, iter):
        erode = cv2.erode(img_param, erosion_kernel, iterations=iter)
        return erode

    def Dilatation(img_param, iter):
        dil = cv2.dilate(img_param, dilation_kernel, iter)
        return dil

    ###################################################################
    ##################convexhull & slicing IMG & resizing##############
    ###################################################################
    ###################################################################

    def Slicing_Resizing(img_param, pre_mode):  #
        contours, hierarchy = cv2.findContours(img_param, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = 0
        for cnt in contours:
            # x, y, w, h = cv2.boundingRect(cnt)
            c = c + 1
            hull = cv2.convexHull(cnt)
            convexHull = cv2.drawContours(img_param, [hull], 0, (125, 125, 125), thickness=-1)

        n_contours, n_hierarchy = cv2.findContours(convexHull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_c = 0
        rects = []
        im_w = img_param.shape[1]
        for cnt in n_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            blocking = cv2.drawContours(img_param, n_contours, -1, (0, 0, 255), 1)
            n_c = n_c + 1
            y2 = round(y / 10) * 10
            index = y2 * im_w + x
            rects.append((index, x, y, w, h))
        rects = sorted(rects, key=lambda x: x[0])

        for i, r in enumerate(rects):
            index, x, y, w, h = r
            if pre_mode == 'bt':
                sliced_img = bt[y:y + h, x:x + w]
            elif pre_mode == 'bt_dil1' or pre_mode == 'imgmodel1':
                sliced_img = bt_dil1[y:y + h, x:x + w]

            elif pre_mode == 'bt_dil1_ero2' or pre_mode == 'imgmodel2':
                sliced_img = bt_dil1_ero2[y:y + h, x:x + w]
            else:
                sliced_img = bt[y:y + h, x:x + w]
            sliced_img = 255 - sliced_img
            global cnt_imgList
            cnt_imgList = cnt_imgList + 1

            ww = round((w if w > h else h) * MARGIN_FOR_SLICEDIMG)
            spc = np.zeros((ww, ww))
            wy = (ww - h) // 2
            wx = (ww - w) // 2

            spc[wy:wy + h, wx:wx + w] = sliced_img
            if h or w < MNIST_IMAGE_FORMAT_SIZE:
                sliced_img = cv2.resize(spc, (MNIST_IMAGE_FORMAT_SIZE, MNIST_IMAGE_FORMAT_SIZE),
                                        interpolation=cv2.INTER_AREA)
            else:
                sliced_img = cv2.resize(spc, (MNIST_IMAGE_FORMAT_SIZE, MNIST_IMAGE_FORMAT_SIZE),
                                        interpolation=cv2.INTER_LINEAR)
            imgList.append(sliced_img)

        #Display(imgList)
        return imgList

    def After_processing(imgList_param, after_mode):
        for i in range(cnt_imgList):
            if after_mode == 'dil1':
                img_Sliced = Dilatation(imgList_param[i], DILATION_ITER1)
            elif after_mode == 'dil2':
                img_Sliced = Dilatation(imgList_param[i], DILATION_ITER2)
            elif after_mode == 'dil3':
                img_Sliced = Dilatation(imgList_param[i], DILATION_ITER3)
            elif after_mode == 'ero1':
                img_Sliced = Erosion(imgList_param[i], EROSION_ITER1)
            elif after_mode == 'ero2':
                img_Sliced = Erosion(imgList_param[i], EROSION_ITER2)
            elif after_mode == 'ero3':
                img_Sliced = Erosion(imgList_param[i], EROSION_ITER3)
            elif after_mode == 'blur':
                img_Sliced = Blur(imgList_param[i])
            else:
                print("there's nothing mode about your input")
            imgList2.append(img_Sliced)
        return imgList2

    def imgData_Nomalization(imgList_param):
        for i in range(cnt_imgList):
            imgList_param[i] = imgList_param[i].reshape(MNIST_IMAGE_FORMAT_SIZE * MNIST_IMAGE_FORMAT_SIZE)
            imgList_param[i] = imgList_param[i].astype("float32") / 255.0
            X.append(imgList_param[i])
        param = np.array(X)
        return param

    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]

    def Display(argv):
        count = 0
        nrows = 6
        ncols = 5

        plt.figure(figsize=(8, 8))

        for n in range(len(argv)):
            count += 1
            plt.subplot(nrows, ncols, count)
            # plt.title(img_array_name[n])
            # plt.imshow(argv[n], cmap='Greys_r')
            plt.imshow(argv[n])

        plt.tight_layout()
        plt.show()
    ###################################################################
    ####################실제 이미지 처리 영역############################
    ###################################################################

    img = cv2.imread(argv)
    if img is None:
        log.debug('Cannot load image: ' + argv)
        exit()

    gray = Gray(img)
    bt = binary_Threshold(gray)
    blur = Blur(bt)
    mg = morph_GRADIENT(blur)
    at = adaptive_Threshold(mg)
    canny = Canny(mg)

    ############pre img model#######################################
    bt_dil1 = Dilatation(bt, DILATION_ITER1)  # img model 1
    bt_dil1_ero2 = Erosion(bt_dil1, EROSION_ITER2)  # img model 2
    ################################################################
    Slicing_Resizing(canny, pre_img_mode)
    #Slicing_Resizing(param1, param2)
    #param1 = 자를 이미지 영역을 정할 이미지(canny엣지)
    #param2 = 사용할 전처리 이미지

    After_processing(imgList, after_img_mode)
    #After_processing(param1, param2)
    #param1 = 자르고 리사이징된 이미지가 들어간 리스트
    #param2 = 자르고 리사이징된 이미지를 다시 전처리할 모드

    #Display(imgList2)

    return imgData_Nomalization(imgList2)
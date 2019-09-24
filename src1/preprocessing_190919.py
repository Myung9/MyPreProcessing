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
MIN_THRESH = 150  # 스레숄드 연산에 사용될 최소값
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
    'img_color_option': 'gray',
    'dataset': 'mnist',
    #'dataset': 'catsdogs',
    ################################
    'processing_mode': 1,

}

w = 100
h = 100



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
        edges = cv2.Canny(img_param, 100, 150, apertureSize=3) # 3 = sobel_kernel_size
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
    def __init__(self, processing_option):
        super().__init__(processing_option)
        self.img_list = []

    def seperate(self, img_param):
        ip = ImageProcessing(image_processing_option)
        #img = cv2.cvtColor(img_param, cv2.COLOR_BGR2GRAY)
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
            x, y, w ,h = cv2.boundingRect(contour_2)
            print(i+1, '번째 컨투어스 좌표(2) ', cv2.boundingRect(contour_2))
            cv2.rectangle(img_param, (x, y), (x + w, y + h), (255, 0, 0), 1)
            blocking = cv2.drawContours(img, contours_2, -1, (0, 0, 255), 2)
            #cv2.line(blocking, (x,y),(x+1,y+1), (0,100,255),3)
            #print_img(blocking)
            rects.append([i, x, y, w, h])
            # 여기서 인덱스를 어떡게할지나 생각해보기 / 그냥놔둘지 바꿀지
            # 인덱스 순서는 왼쪽아래부터 오른쪽으로 확인하면서 위로
        rects = sorted(rects, key=lambda x:x[0])
        print(rects)
        shapes = []
        img_list = []
        seperated_img = np.empty(())
        for i, rect in enumerate(rects):
            img_list = self.img_list
            idx, x, y, w, h = rect
            print('wow', rect)
            seperated_img = bt[y:y+h, x:x+w]
            seperated_img = 255 - seperated_img
            img_list.append(seperated_img)
            print('seperated_img.shape', seperated_img.shape)
            shapes.append(seperated_img.shape)
            print(x, y, w, h)
        print_img('seperated_img', seperated_img)
        #print(img_list)
        #print(len(img_list))
        #print('shapes\n', shapes)
        #print(rects)
        return img_param, img_list, rects # 잘린이미지리스트 // 원본에서의 좌표와 w,h 리스트

    def equalize_width_height(self, ori_image, rect_list): # 그래이일 경우에만 되려나..? # not resize
        #아직은 고려하지않는부분
        #print(ori_image.shape)
        #print(rect_list)
        for rect in rect_list:
            print(rect[-4:])
            x, y, w, h = rect[-4:]
            plate_wh = max([h, w])
            '''
            이 함수는 아직 그대로 두고 나중에 고려하자
            '''
    def image_resize(self, img):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


    def set_image(self, img, depth_option): # image read with depth
        if depth_option == 'gray':
            try:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        elif depth_option == 'RGB':
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # 그냥 default로 색상 읽기
        else: #others
            return cv2.cvtColor(img)  # 그냥 default로 색상 읽기

    def seperate_preprocessing(self, img):
        ip = ImageProcessing(image_processing_option)

        src = '../images'
        img_paths = os.listdir(src)
        img_paths = [os.path.abspath(os.path.join(src, i))
                   for i in img_paths]
        print(img_paths)
        #print(os.path.basename(img_paths[0]))
        for img_path in img_paths:
            #img = cv2.imread('../images/dog.3261.jpg')
            img = cv2.imread(img_path)
            img_tmp = img.copy()
            img_tmp2 = img.copy()
            basename = os.path.basename(img_path)

            print(img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #gray = ip.erosion(gray)
            #gray = ip.erosion(gray)
            #gray = ip.erosion(gray)
            bt = ip.binary_threshold(gray)


            for i in range(20):
                gray = ip.blur(gray)

            for i in range(50):
                gray = ip.erosion(gray)

            for i in range(10):
                gray = ip.dilation(gray)



            #bt = ip.adaptive_threshold(gray)
            edge = ip.canny(bt)
            print('aaa', edge.shape)
            #print_img('edge', edge)
            #print_img('bt', bt)



            red_mask = np.zeros(img.shape, img.dtype)
            red_mask[:,:] = [0, 0, 255]

            blue_mask = np.zeros(img.shape, img.dtype)
            blue_mask[:,:] = [255, 0, 0]




            bt_mask = cv2.bitwise_and(red_mask, red_mask, mask=bt)
            bt_result = cv2.addWeighted(bt_mask, 1, img, 1, 0, img)
            cv2.imwrite('../result_images/bt/' + basename, bt_result)


            edge_mask = cv2.bitwise_and(blue_mask, blue_mask, mask=edge)
            edge_result = cv2.addWeighted(edge_mask, 1, img_tmp, 1, 0, img_tmp)
            cv2.imwrite('../result_images/edge/' + basename, edge_result)

            #################################

            #print_img('a', img_tmp2)


            img_b, img_g, img_r = cv2.split(img_tmp2)
            zeros = np.zeros((img_tmp2.shape[0], img_tmp2.shape[1]), dtype=img.dtype)
            img_b = cv2.merge([img_b, zeros, zeros])
            img_g = cv2.merge([zeros, img_g, zeros])
            img_r = cv2.merge([zeros, zeros, img_r])

            cv2.imwrite('../result_images/b/' + basename, img_b)
            cv2.imwrite('../result_images/g/' + basename, img_g)
            cv2.imwrite('../result_images/r/' + basename, img_r)

            b_gray = ip.binary_threshold(img_b)
            g_gray = ip.binary_threshold(img_g)
            r_gray = ip.binary_threshold(img_r)

            cv2.imwrite('../result_images/b_gray/' + basename, b_gray)
            cv2.imwrite('../result_images/g_gray/' + basename, g_gray)
            cv2.imwrite('../result_images/r_gray/' + basename, r_gray)

            img_tmp = img
            img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
            guassi = cv2.adaptiveThreshold(img_tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            guassi_canny = ip.canny(guassi)
            meanc = cv2.adaptiveThreshold(img_tmp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            meanc_canny = ip.canny(meanc)

            cv2.imwrite('../result_images/adt_gau/' + basename, guassi)
            cv2.imwrite('../result_images/adt_meanc/' + basename, meanc)

            cv2.imwrite('../result_images/adt_gau_canny/' + basename, guassi_canny)
            cv2.imwrite('../result_images/adt_meanc_canny/' + basename, meanc_canny)

    '''
        for i in range(10):
            lines = cv2.HoughLines(edge, 1, np.pi/((i+10)*13), 180)
            for rho,theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 2000 * (-b))
                y1 = int(y0 + 2000 * (a))
                x2 = int(x0 - 2000 * (-b))
                y2 = int(y0 - 2000 * (a))

                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
            res = np.vstack((img))
            print_img('aa', img)
        pass
        '''




def print_img(window_name,img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass




if __name__ == '__main__':
    print('main start')
    img = cv2.imread('../images/cat.49.jpg')
    #img = cv2.imread('../images/wow.JPG')
    #img = cv2.imread('../images/cat.60.jpg')

    #cv2.imshow('aaa', img)
    #cv2.waitKey(0)
    #print(img)
    a = SetImage(image_processing_option)
    a.seperate_preprocessing(image_processing_option)
    exit()

    a_img, a_img_list, a_rects = a.seperate(img)
    print('a_img', a_img)
    print('a_img_list',len(a_img_list))
    print('a_rects', a_rects)

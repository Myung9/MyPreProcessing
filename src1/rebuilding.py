import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

MIN_THRESH = 127
MAX_THRESH = 255




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

    def threshold(self, img_param, thresh_mode):
        if thresh_mode == 'BINARY':
            ret, dst = cv2.threshold(img_param, MIN_THRESH, MAX_THRESH, cv2.THRESH_BINARY)
        elif thresh_mode == 'BINARY_INV':
            ret, dst = cv2.threshold(img_param, MIN_THRESH, MAX_THRESH, cv2.THRESH_BINARY_INV)
        elif thresh_mode == 'OTSU':
            ret, dst = cv2.threshold(img_param, MIN_THRESH, MAX_THRESH, cv2.THRESH_OTSU)
        elif thresh_mode == 'TOZERO':
            ret, dst = cv2.threshold(img_param, MIN_THRESH, MAX_THRESH, cv2.THRESH_TOZERO)
        elif thresh_mode == 'TRUNC':
            ret, dst = cv2.threshold(img_param, MIN_THRESH, MAX_THRESH, cv2.THRESH_TRUNC)
        else:
            ret, dst = cv2.threshold(img_param, MIN_THRESH, MAX_THRESH, cv2.THRESH_BINARY)
        return dst

    def adaptive_threshold(self, img_param, mask_mode):
        if mask_mode == 'MEAN_C':
            return cv2.adaptiveThreshold(img_param, 250,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         3, 20)
        elif mask_mode == 'GAUSSIAN':
            return cv2.adaptiveThreshold(img_param, 250,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY,
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




class ImageProcessing(ImageProcessingModule):
    def __init__(self):
        super().__init__(processing_option)
        self.ipm = ImageProcessingModule(processing_option)
        #엣지옵션 -> canny / adat_gausi / adat_meanc
        print('in img prcessing')
        print('##image depth->>', processing_option['DTS_IMG_DEPTH_CD'])
        self.img_depth = processing_option['DTS_IMG_DEPTH_CD']


    def seperate_preprocessing(self, img_param, mode): # need grayscale image
        ipm = self.ipm
        print(img_param.shape)
        #print_img('aa', img_param)

        img = cv2.cvtColor(img_param, cv2.COLOR_BGR2GRAY)

        img_ori = img.copy()
        #현재는 한가지 모드만 / 차후에 2부터 모드..
        if mode == 1:
            bt = ipm.threshold(img_param=img, thresh_mode='BINARY')
            blur = ipm.blur(bt)
            dil = ipm.dilation(blur)
            ero = ipm.erosion(dil)
            canny = ipm.canny(ero)
        else: # 입력이 잘못됬을때 default로
            bt = ipm.threshold(img_param=img, thresh_mode='BINARY')
            blur = ipm.blur(bt)
            dil = ipm.dilation(blur)
            ero = ipm.erosion(dil)
            canny = ipm.canny(ero)
        #여기에 추가적인 seperte용 처리 연산을 elif구문으로
        return canny

    def cover_edge(self, img_folder_path, cover_mode):
        ipm = self.ipm
        #나중에는 csv가 있을테니 그것을 이용함
        img_paths = []
        #csv일경우 PATH column을 리스트로 읽어와서 img_path_list로
        #폴더명을 입력받을 경우 하위폴더구조 혹은 바로 하위 폴더에 이미지들은 검색해서
        #list로 반환하여 img_path_list로

        #폴더입력인경우 / 폴더인경우중에서도 한폴더에 다있는경우
        folder_path = '../images'
        '''
        img_paths = os.listdir(folder_path)
        img_paths = [os.path.abspath(os.path.join(folder_path, i))
                     for i in img_paths]
        '''
        for (path, dir, files) in os.walk('../images/mnist_sample'):
            for filename in files:
                ext = os.path.splitext(filename)[-1]

                print(ext)
                if ext == '.csv':
                    print('ext is csv')
                else:
                    print(ext)


                if ext == '.png' or '.jpg':
                    print(ext)

                if ext == 'png':
                    print("설마 피엔지도?")

                #if ext == '.png':
                if ext == '.png' or ext == '.jpg':
                    print('%s/%s' % (path, filename))

                    img_paths.append(str(path) + '/' + str(filename))
                    print('in 구문자 ', ext)

                else:
                    print('else')

        for img_path in img_paths:
            img = cv2.imread(img_path)
            ori_img = img.copy()
            ##########
            #여기서부터 모드 분기?
            ##########

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bt = ipm.threshold(gray, 'binary')
            blur = ipm.blur(bt)
            eros = ipm.erosion(blur)
            dil = ipm.dilation(eros)
            edge = ipm.canny(dil)

            red_mask = np.zeros(img.shape, img.dtype)
            blue_mask = np.zeros(img.shape, img.dtype)

            red_mask[:, :] = [0, 0, 255]
            blue_mask[:, :] = [255, 0, 0]

            mask = cv2.bitwise_and(red_mask, red_mask, mask=bt)

            ############여기까지가 기본 default 엣지





            ###이부분은 채널나누는부분##
            img_b, img_g, img_r = cv2.split(ori_img)
            zeros = np.zeros((ori_img.shape[0],ori_img.shape[1]), dtype=ori_img.dtype)
            img_b = cv2.merge([img_b, zeros, zeros])
            img_g = cv2.merge([zeros, img_g, zeros])
            img_r = cv2.merge([zeros, zeros, img_r])



            #다음에는 AdaptiveThreshold의 gaussian / meanc 를 모드로 분기하여
            print_img()


class SeperateImage(ImageProcessing):
    def __init__(self):
        super().__init__()
        self.img_list = []
        print('__init__ serperateimage')

    def seperate(self, img_param): # seperate preprocessing된 gray 스케일
        ip = ImageProcessing()
        #processing_img = ip.seperate_preprocessing(img_param, 'BINARY')

        #img = cv2.cvtColor(img_param, cv2.COLOR_BGR2GRAY)
        #img = ip.seperate_preprocessing(img_param, 'BINARY')
        print('seperate in img_param', img_param)

        #print_img('원본이미지', img_param)
        bt = ip.threshold(img_param, 'BINARY')
        blur = ip.blur(bt)
        mg = ip.morph_gradient(blur)
        canny = ip.canny(mg)
        #print_img('canny edge', canny)

        contours_1, hierarchy_1 = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        convexHull = np.empty(())
        for i, contour_1 in enumerate(contours_1):
            x, y, w, h = cv2.boundingRect(contour_1)
            print(i+1, '번째 좌표(1) ', cv2.boundingRect(contour_1))
            hull = cv2. convexHull(contour_1)
            convexHull = cv2.drawContours(canny, [hull], 0, (125, 125, 125))
            print(type(convexHull))
            print(convexHull.shape)
        #print_img('convexhull', convexHull)
        rects = []
        #img_h, img_w = img_param.shape[:2]
        contours_2, hierarchy_2 = cv2.findContours(convexHull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour_2 in enumerate(contours_2):
            x, y, w, h = cv2.boundingRect(contour_2)
            print(i+1, '번째 컨투어스 좌표(2) ', cv2.boundingRect(contour_2))
            cv2.rectangle(img_param, (x, y), (x + w, y + h), (255, 0, 0), 1)
            #blocking = cv2.drawContours(img, contours_2, -1, (0, 0, 255), 2)
            #cv2.line(blocking, (x,y),(x+1,y+1), (0,100,255),3)
            #print_img(blocking)
            rects.append([i, x, y, w, h])
            # 여기서 인덱스를 어떡게할지나 생각해보기 / 그냥놔둘지 바꿀지
            # 인덱스 순서는 왼쪽아래부터 오른쪽으로 확인하면서 위로
        rects = sorted(rects, key=lambda x:x[0])
        shapes = []
        img_list = []

        for i, rect in enumerate(rects):
            img_list = self.img_list
            idx, x, y, w, h = rect
            seperated_img = bt[y:y+h, x:x+w]
            seperated_img = 255 - seperated_img
            img_list.append(seperated_img)
            shapes.append(seperated_img.shape)
            #print_img('ttt', seperated_img)
        return img_param, img_list, rects # 잘린이미지리스트 // 원본에서의 좌표와 w,h 리스트



#SeperateImage(processing_option=processing_option).seperate(cv2.imread('../images/cat.57.jpg'))
if __name__ == '__main__':
    processing_option = {
        'order': 'a',
        'DTS_IMG_DEPTH_CD': 1
    }
    test_img = cv2.imread('../images/5.png')
    for_sep_img = ImageProcessing().seperate_preprocessing(img_param=test_img, mode=1)
    SeperateImage().seperate(for_sep_img) # 현라인 포함 위에까지 두줄은 하나로...?
    '''
    for (path, dir, files) in os.walk('../images/mnist_sample'):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png' or '.jpg':
                print('%s/%s'%(path,filename))
    '''
    ImageProcessing().cover_edge(img_folder_path='../images', cover_mode=1)









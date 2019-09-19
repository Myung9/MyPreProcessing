import cv2
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import shutil

width = 28
height = 28
batch_size = 1000
save_npy = True

label_dict = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9
}

#DTS_IMG_DEPTH = 0
DTS_IMG_DEPTH = 1

class Convert:
    def __init__(self, dts):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        dts = '..' + dts
        dts_folder_name = '.'.join(dts.split('.')[:-1])
        self.input_dataset_path = '.'.join(dts.split('.')[:-1])
        self.input_dataset_name = self.input_dataset_path.split('/')[-1]
        self.output_dataset_path = '../pretrain/' + self.input_dataset_name + '_' + str(batch_size) + '_' + str(
            width) + '_' + str(height)

        self.dataset_name = self.input_dataset_name

        self.rawdata_dataset_path = '../rawdata/' + dts_folder_name
        self.pretrain_dataset_path = '../pretrain/' + self.output_dataset_path

        self._one_hot_label_set = self.one_hot_matrix(label_dict)


        ##################





        ##################



    def confirm_fileform(self, folder_path, fileform):  # return file list
        fileform = '.' + fileform
        result_file_list = []
        file_list = os.listdir(folder_path)
        for file in file_list:
            tmp = file.split('/')
            for i in tmp[-1:]:
                filename, ff = os.path.splitext(i)
                if ff == fileform:
                    result_file_list.append(i)
        return result_file_list

    def read_csv_label(self, flag):  # flag = train/ test
        tmp_dict = {}
        if flag == 'train':
            csv_list = ['train.csv']
        elif flag == 'test':
            csv_list = ['test.csv']
        else:
            csv_list = self.confirm_fileform(self.output_dataset_path + '/', 'csv')
        for csv in csv_list:
            csv_path = self.output_dataset_path + '/' + csv
            r_csv = pd.read_csv(csv_path)
            labels = r_csv['label']
            for i in range(len(labels)):
                if not labels[i] in tmp_dict:
                    tmp_dict[labels[i]] = 1
                else:
                    tmp_dict[labels[i]] += 1
        return tmp_dict

    def get_csv_col(self, flag, col_name):  # csv col - > list[]
        csv_path = self.output_dataset_path + '/' + flag + '.csv'
        read_csv = pd.read_csv(csv_path)
        col = read_csv[col_name]
        return col

    def get_label_from_path(self, path):  # img를 읽을때 img의 path를 통해 폴더명으로 라벨지정
        return int(path.split('/')[-2])

    def one_hot_matrix(self, label_dict):
        labels = [x for x in range(len(label_dict))]
        C = len(labels)
        C = tf.constant(C, name='C')
        one_hot_matrix1 = tf.one_hot(indices=labels, depth=C, axis=0)
        with tf.Session() as sess:
            one_hot = sess.run(one_hot_matrix1)
            sess.close()
        return one_hot

    def get_one_hot_label(self, label):
        for i, one_hot in enumerate(self._one_hot_label_set):
            if i == label:
                return one_hot

    def read_np_label(self, flag, step): # 스텝에 맞게 label batchsize만큼 읽어오기
        #step은 0부터 들어온다
        batch_size = self.batch_size
        label_list = self.get_csv_col(flag, 'label')
        print(label_list)
        label_amount = len(label_list)
        start = (step - 1)
        tmp = label_list[step * batch_size:step * batch_size + batch_size] # 나누어떨어질때
        tmp = label_list[step * batch_size:(step * batch_size) + (label_amount % batch_size)]
        print(len(tmp))



    def read_np_image(self, flag, step): # 스텝에 맞는 npy파일 읽어오기
        pass

    def set_value_about_channel(self): # 이미지 cvt값 설정 / nparray 채널값 설정
        self.width = width
        self.height = height
        print('DTS_IMG_DEPTH', DTS_IMG_DEPTH)
        if DTS_IMG_DEPTH == 0: # 이상한놈
            print('DTS_IMG_DEPTH', DTS_IMG_DEPTH)
        elif DTS_IMG_DEPTH == 1: # gray
            print('DTS_IMG_DEPTH', DTS_IMG_DEPTH)
        elif DTS_IMG_DEPTH == 3 or 4: # RGB or RGBA -> RGB로하기
            print('DTS_IMG_DEPTH', DTS_IMG_DEPTH)





    def save_image_npy(self, flag, seperate):
        img_path_list = self.get_csv_col(flag, 'resized_image_path')
        img_amount = len(img_path_list)
        if img_amount % batch_size == 0:
            total_batch_len = img_amount // batch_size
        else:
            total_batch_len = img_amount // batch_size + 1
        for batch_cnt in range(total_batch_len):
            if seperate == True:
                if (batch_cnt + 1) * batch_size > img_amount:  #
                    converted = np.empty(((img_amount % batch_size), width, height, 3))
                    #label_list = [None] * (img_amount % batch_size)
                    batch_list = img_path_list[batch_cnt * batch_size:batch_cnt * batch_size + (img_amount % batch_size)]
                else:
                    converted = np.empty((batch_size, width, height, 3))
                    #label_list = [None] * batch_size
                    batch_list = img_path_list[batch_cnt * batch_size:batch_cnt * batch_size + batch_size]
                for i, img_path in enumerate(batch_list):
                    #print(i, ' / ', batch_cnt, ' / ',img_path)
                    img = cv2.imread(img_path)
                    img = img.astype(np.float32)
                    img = np.multiply(img, 1.0 / 255.0)
                    converted[i] = img
                    #label_list[i] = self.get_one_hot_label(self.get_label_from_path(img_path))
                np.save(self.output_dataset_path + '/temp/data/'
                        + flag + '_img_' + str(batch_cnt+1).zfill(5)
                        + '.npy', converted)
            elif seperate == False:
                converted = np.empty((img_amount, width, height, 3))
                #label_list = [None] * img_amount
                batch_list = img_path_list
                for i, img_path in enumerate(batch_list):
                    img = cv2.imread(img_path)
                    img = img.astype(np.float32)
                    img = np.multiply(img, 1.0 / 255.0)
                    converted[i] = img
                    #label_list[i] = self.get_one_hot_label(self.get_label_from_path(img_path))
                np.save(self.output_dataset_path + '/temp/data/'
                        + flag + '_img.npy', converted)

    def test(self):
        #self.save_image_npy(flag='train', seperate=True)
        self.read_np_label('train', 1)
        self.set_value_about_channel()



if __name__ == '__main__':
    print('hi im test')
    # Convert(dts='asefas/feag33w/fsssa/rawdata/mnist_low.zip').get_np_image('train', True, 10)
    # Convert(dts='/rawdata/mnist_low.zip').get_batch('train', True, 10)
    Convert(dts='/rawdata/mnist_low.zip').test()
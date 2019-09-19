import tensorflow as tf
import numpy as np

batch_size = 64
height = 28
width = 28
channel_n = 1 #gray scale

num_classes = 10

batch_image = np.empty((batch_size, height, width))
batch_label = np.empty((batch_size, num_classes))


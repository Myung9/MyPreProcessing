import tensorflow as tf
import numpy as np

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

def one_hot_matrix(label_dict):
    labels = [x for x in range(len(label_dict))]
    C = len(labels)
    C = tf.constant(C, name='C')
    one_hot_matrix1 = tf.one_hot(indices=labels, depth=C, axis=0)
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix1)
        sess.close()
    return one_hot

result = one_hot_matrix(label_dict)
print(result)

def get_onehot_label(label):
    for i, one_hot in enumerate(result):
        if i == label:
            return one_hot
####################################################################


import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam

def preprocess(img):
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 64 # max length of predicted labels


def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1: 
            break
        else:
            ret+=alphabets[ch]
    return ret

model = tf.keras.models.load_model('model/handwriting.h5')

# model.summary()

# test = pd.read_csv('label/written_name_test_v2.csv')
# D:\CODE\ml_app\test_v2\test\TEST_0001.jpg

# plt.figure(figsize=(15, 10))
# for i in range(6):
#     ax = plt.subplot(2, 3, i+1)
#     img_dir = 'test_v2/test/'+test.loc[i, 'FILENAME']
#     image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
#     plt.imshow(image, cmap='gray')

#     # print(img_dir)
    
#     image = preprocess(image)
#     image = image/255.
#     pred = model.predict(image.reshape(1, 256, 64, 1))
#     decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
#                                        greedy=True)[0][0])
#     print(num_to_label(decoded[0]))
#     plt.title(num_to_label(decoded[0]), fontsize=12)
#     plt.axis('off')
    
# plt.subplots_adjust(wspace=0.2, hspace=-0.8)

#tes 1 gambar

img_dir = 'test_v2/test/TEST_0009.jpg'
image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
# plt.imshow(image, cmap='gray')

# print(img_dir)

image = preprocess(image)
image = image/255.
pred = model.predict(image.reshape(1, 256, 64, 1))
decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                    greedy=True)[0][0])
print(num_to_label(decoded[0]))


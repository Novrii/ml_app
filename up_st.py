import cv2
import streamlit as st
import numpy as np

from PIL import Image

import tensorflow as tf
from keras import backend as K

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


def main_loop():
    st.title("Handwriting Recognition Demo App")
    st.subheader("This app allows you to recognize handwriting image !")
    st.text("We use Tensorflow, OpenCV and Streamlit for this demo")

    model = tf.keras.models.load_model('temp_model/hrv1.h5')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    image = preprocess(image)
    image = image/255.
    
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0])

    st.text("Image Handwriting")
    st.image([image_file])
    st.text(f"Result: {num_to_label(decoded[0])}")

if __name__ == '__main__':
    main_loop()
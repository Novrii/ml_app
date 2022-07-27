import cv2
import streamlit as st
import numpy as np

from PIL import Image

import tensorflow as tf
from keras import backend as K
from streamlit_drawable_canvas import st_canvas

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


def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "Upload Image": up_img,
        "Canvas": canvas_app
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://github.com/kurniawan2805/handwriting-recognition">@Tim HR 4</a> </h6>',
            unsafe_allow_html=True,
        )


def up_img():

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

    st.text(f"Image Handwriting")
    st.image([image_file])
    st.text(f"Probably the result: {num_to_label(decoded[0])}")

def canvas_app():
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=64,
        width=256,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        model = tf.keras.models.load_model('temp_model/hrv1.h5')

        imagefile = canvas_result.image_data

        image = imagefile.copy()
        image = image.astype('uint8')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = preprocess(image)
        image = image/255.

        # image.resize(1, 256, 64, 1)
        
        pred = model.predict(image.reshape(1, 256, 64, 1))
        decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                        greedy=True)[0][0])

        st.text("Image Handwriting")
        st.image([imagefile])
        st.text(f"Probably the result : {num_to_label(decoded[0])}")

if __name__ == '__main__':
    st.set_page_config(
        page_title="Streamlit Handwriting Recognition Demo", page_icon="🖊️"
    )
    st.title("Handwriting Recognition Demo App")
    st.subheader("This app allows you to recognize handwriting image !")
    st.text("We use Tensorflow and Streamlit for this demo")
    st.sidebar.subheader("Configuration")
    main()
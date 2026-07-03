import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elyazami_model_cats_dogs.h5')

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

HEADER_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DL_SAID.png')
if os.path.exists(HEADER_IMAGE):
    st.image(HEADER_IMAGE, use_container_width=True)

st.title('Classification Chiens vs Chats')
st.write("Importez une image de chien ou de chat pour lancer la prédiction de mon modèle.")

uploaded_file = st.file_uploader("Cliquez ici pour télécharger", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image téléchargée', use_container_width=True)
    img = image.resize((150, 150))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0][0]
    if prediction > 0.5:
        st.success(f"Prediction: **Dog** ({prediction*100:.1f}% confiance)")
    else:
        st.success(f"Prediction: **Cat** ({(1-prediction)*100:.1f}% confiance)")

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import os
import warnings

st.title("Crop Classification")
from langchain_groq import ChatGroq
model = load_model('DLproject.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

llm = ChatGroq(
    temperature=0,
    groq_api_key = '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',
    model_name = "llama-3.1-70b-versatile"
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_dir = "./temp_uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    img_cv2 = cv2.imread(file_path)
    
    if img_cv2 is not None:
        resized_array = cv2.resize(img_cv2, (224,224))
        img_array = resized_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        output = labels[np.argmax(predictions)]
        st.write(output)

        res = llm.invoke(output)
        st.write(res.content)

        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        st.image(img_rgb, caption='Image read by OpenCV', use_column_width=True)
        
        st.write('Image shape:', img_rgb.shape)
    else:
        st.write("Error: Could not load the image using OpenCV.")
    
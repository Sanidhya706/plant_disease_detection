import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from model.preprocessing import preprocess_image

# Load the trained model
model = load_model('model/model.h5')

# Title of the app
st.title("Plant Disease Detection System")

# Image upload functionality
uploaded_image = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Model prediction
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class = np.argmax(prediction, axis=1)

    # Display prediction results
    st.subheader(f"Predicted Disease: {predicted_class}")
    st.write(f"Confidence: {max(prediction[0]) * 100:.2f}%")

# Optionally display further results like disease treatment
